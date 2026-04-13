#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D-CNN（CBAM+ResBlock）完整训练与评测脚本（输入维度固定为 [B,1,1,150]）
- 数据：仅支持 .txt，每个文件是一条 SNV+高斯加权处理后的 150 维光谱
- 预处理：在本脚本中不再做 SNV / 高斯，直接读取 150 维向量
- 输入给模型：Dataset 输出 [1,1,150]，DataLoader 叠成 [B,1,1,150]
- KS 划分：对 150 维特征做 Kennard–Stone (默认 75/25)
- 训练：5 次 run，每次 200 epoch
- 评测：PreciseBN（可开关）+ EMA（修复为使用不带采样器的 loader 刷新 BN）
- 保存：每个 run 保存最佳 EMA 准确率权重
- 日志：控制台 + 文件 logs/<脚本名_时间>.txt
"""

import os
import sys
import time
import copy
import hashlib
import logging
import multiprocessing
from collections import Counter
from typing import Optional, Tuple, Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# ============================== 全局配置 ==============================
# 注意：这里的 DATA_FOLDER 目录结构必须是：
# DATA_FOLDER/
#   female/*.txt
#   male/*.txt
# 每个 txt 文件为一行 150 个浮点数（SNV+高斯离线处理后的特征）
# DATA_FOLDER      = r"hy_txt_snv_gauss_1113/train_val"
DATA_FOLDER      = r"hy_txt_snv_gauss_1120_no_clear/train_val"
ROOT_DIR         = r"hy_model/hy_model_1dcnn_v7_f1.5_m1.0-1224_no_clear_suspect_patchs"
STR_TIME         = time.strftime("%Y%m%d")
NUM_RUNS         = 5
NUM_EPOCHS       = 200
BATCH_SIZE       = 128
INIT_LR          = 2e-4
WEIGHT_DECAY     = 1e-4
BN_MOMENTUM      = 0.005
USE_BALANCED_SAMPLER = True
GRAD_CLIP_NORM   = 5.0

USE_EMA          = True
EMA_DECAY_BASE   = 0.999                 # 前 5 epoch 用 0.99

USE_PRECISE_BN   = True
PRECISE_BN_BATCHES = 200

# 是否将 150 维特征预先读入内存，加速训练
PRECOMPUTE_FEATURES_IN_MEMORY = True

torch.set_num_threads(multiprocessing.cpu_count())

# ============================== 怀疑样本降权配置 ==============================
# suspect_paths.txt：每行一个路径，和 Dataset 里的路径一致（或同样的相对路径）
SUSPECT_PATHS_TXT = r"suspect_paths.txt"   # 如果不在同目录，改成你的实际路径
SUSPECT_LOSS_FACTOR = 0.3                 # 对怀疑样本的 loss 乘系数（0.1~0.5 都可以）

def _load_suspect_paths(txt_path: str):
    """
    读取怀疑样本列表，返回一个规范化后的路径 set（全局只读）。
    """
    if not os.path.isfile(txt_path):
        print(f"[SUSPECT] 没找到 {txt_path}，不做怀疑样本降权处理。")
        return set()

    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 规范化路径，兼容 / 和 \
            paths.append(os.path.normpath(line))

    suspects = set(paths)
    print(f"[SUSPECT] 已加载怀疑样本 {len(suspects)} 条，自 {txt_path}")
    return suspects

SUSPECT_PATH_SET = _load_suspect_paths(SUSPECT_PATHS_TXT)

# ============================== 日志（控制台+文件） ==============================
filename = os.path.basename(__file__) if "__file__" in globals() else "train_eval_1dcnn_hw1.py"
LOG_FILE = os.path.join("logs", f"{filename}_{time.strftime('%Y%m%d-%H%M%S')}.txt")

def _setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("train_1dcnn_hw1")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="[%H:%M:%S]")
    sh = logging.StreamHandler(stream=sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

def export_torchscript(model: nn.Module, save_path_pt: str, device: torch.device):
    """
    将当前模型导出为 TorchScript .pt 文件（用于 C# / 部署）
    """
    # 拷贝一份，移到 CPU 上再导出（避免和训练过程的 GPU 状态纠缠）
    model_cpu = copy.deepcopy(model).to('cpu')
    model_cpu.eval()
    with torch.no_grad():
        scripted = torch.jit.script(model_cpu)
        scripted.save(save_path_pt)
    print(f"[TorchScript] Saved: {save_path_pt}")

LOGGER = _setup_logger(LOG_FILE)

import builtins as _builtins
_builtin_print = _builtins.print
def print(*args, **kwargs):
    if 'file' in kwargs and kwargs['file'] is not sys.stdout:
        _builtin_print(*args, **kwargs)
    else:
        msg = " ".join(str(a) for a in args)
        LOGGER.info(msg)

# ============================== 数据处理（.txt -> 150维向量） ==============================
def txt_to_vec150(path: str) -> np.ndarray:
    """
    读取一个 txt 文件，并解析为长度 150 的向量。
    默认你的离线预处理脚本写的是：每个文件一行 150 个浮点数。
    为保险起见，这里允许多行，会把所有 token 拼在一起再检查长度。
    """
    tokens: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(line.split())

    if len(tokens) != 150:
        raise AssertionError(f"[{path}] 期望 150 个数，实际 {len(tokens)}")

    arr = np.array(list(map(float, tokens)), dtype=np.float32)  # (150,)
    return arr

def to_model_input_hw1(vec150: np.ndarray) -> torch.Tensor:
    """
    将 (150,) 向量转成 [1,1,150]。
    DataLoader 会自动在第 0 维叠成 batch -> [B,1,1,150]
    """
    return torch.tensor(vec150, dtype=torch.float32).reshape(1, 1, 150)

# ============================== 数据集（仅从 .txt 读，可选内存预取） ==============================
class SpectralTxtDataset(torch.utils.data.Dataset):
    """
    输出：(x[1,1,1,150], y[long], path[str])
    目录结构：root/{female,male}/*.txt
    每个 txt 为一条 SNV+高斯处理后的 150 维光谱
    """
    def __init__(self, root_dir: str, precompute: bool = False):
        self.root_dir = root_dir
        self.class_to_idx = {"female": 0, "male": 1}
        self.samples: List[tuple] = []
        for cname, y in self.class_to_idx.items():
            cdir = os.path.join(root_dir, cname)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.lower().endswith(".txt"):
                    self.samples.append((os.path.join(cdir, fn), y))
        if not self.samples:
            raise RuntimeError(f"未在 {root_dir} 下找到 .txt（需要 female/ 与 male/ 子目录）")

        self.precompute = precompute
        self.features = None  # (N,150)
        if self.precompute:
            print("[INFO] Precomputing 150-d features into memory ...")
            vecs = []
            for (p, _) in self.samples:
                vecs.append(txt_to_vec150(p))
            self.features = np.stack(vecs).astype(np.float32)  # (N,150)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        if self.precompute and self.features is not None:
            vec150 = self.features[idx]
        else:
            vec150 = txt_to_vec150(p)  # (150,)
        x = to_model_input_hw1(vec150)  # [1,1,150]
        return x, torch.tensor(int(y), dtype=torch.long), p

# ============================== KS 划分（Kennard–Stone） ==============================
def kennard_stone(X: np.ndarray, n_selected: int) -> np.ndarray:
    n = X.shape[0]
    selected = np.zeros(n, dtype=bool)
    dist = pairwise_distances(X)
    idx = np.unravel_index(np.argmax(dist), dist.shape)
    selected[idx[0]] = True
    selected[idx[1]] = True
    while np.sum(selected) < min(n_selected, n):
        unselected = np.where(~selected)[0]
        sub_dist = dist[unselected][:, selected]
        min_dist = np.min(sub_dist, axis=1)
        new_idx = unselected[np.argmax(min_dist)]
        selected[new_idx] = True
    return selected

class KSHyperspectralDataset:
    """
    对 SpectralTxtDataset 做 KS 划分，保证 (x,y,path) 输出结构不变。
    这里的特征是离线 SNV+高斯处理后的 150 维向量。
    """
    def __init__(self, data_folder: str, train_ratio: float = 0.75):
        # 先只构造不预取的 base 来做 KS 特征
        base_for_ks = SpectralTxtDataset(data_folder, precompute=False)
        if len(base_for_ks) == 0:
            raise ValueError("基础数据集为空！")

        # 为 KS 生成所有样本的 150 向量（每个文件读一次）
        X = np.stack([txt_to_vec150(p) for (p, _) in base_for_ks.samples])
        selected = kennard_stone(X, n_selected=int(train_ratio * len(X)))

        # 训练/测试子集索引
        self.train_indices = np.where(selected)[0]
        self.test_indices  = np.where(~selected)[0]

        # 重新构造一个“可选预取”的 base 数据集用于实际训练/评估
        self.base_dataset = SpectralTxtDataset(data_folder, precompute=PRECOMPUTE_FEATURES_IN_MEMORY)

        print("\n===== KSHyperspectralDataset 数据集划分统计 =====")
        print(f"总样本数: {len(X)}")
        print(f"训练集数量: {len(self.train_indices)} (占比: {len(self.train_indices) / len(X):.1%})")
        print(f"测试集数量: {len(self.test_indices)} (占比: {len(self.test_indices) / len(X):.1%})")

    @property
    def train(self):
        return torch.utils.data.Subset(self.base_dataset, self.train_indices)

    @property
    def test(self):
        return torch.utils.data.Subset(self.base_dataset, self.test_indices)

# ============================== 模型（CBAM + ResBlock，接收 [B,1,1,150]） ==============================
class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        red = max(1, channels // reduction)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, red),
            nn.ReLU(inplace=True),
            nn.Linear(red, channels)
        )
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=2)
        mx, _ = torch.max(x, dim=2)
        ca = self.channel_fc(avg) + self.channel_fc(mx)
        ca = torch.sigmoid(ca).unsqueeze(2)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.spatial_conv(sa))
        return x * sa

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=True),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

class HyperspectralCNN(nn.Module):
    """
    接收输入 [B,1,1,150]：
      - 在 forward 开头展平成 [B,1,150] 做 1D 卷积
      - 最后全局池化 -> [B,256] -> fc
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.cbam1 = CBAM1D(64, reduction=8)
        self.res1 = ResidualBlock1D(64, 128)
        self.res2 = ResidualBlock1D(128, 256)
        self.cbam2 = CBAM1D(256, reduction=8)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里不再用含 tuple(x.shape) 的 assert，避免 TorchScript 报错
        if x.dim() != 4 or x.size(1) != 1 or x.size(2) != 1 or x.size(3) != 150:
            # 用固定字符串，TorchScript 可以接受
            raise RuntimeError("HyperspectralCNN expects input shape [B,1,1,150].")

        B = x.size(0)
        # [B,1,1,150] -> [B,1,150]
        x = x.view(B, 1, 150)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.cbam2(x)
        x = self.pool(x)

        x = self.dropout(x)
        x = self.avg_pool(x)   # [B,256,1]
        x = x.view(B, -1)      # [B,256]
        return self.fc(x)      # [B,2]

# ============================== 工具 ==============================
def ensure_input_hw1(x: torch.Tensor) -> torch.Tensor:
    """
    统一输入为 [B,1,1,150]，兼容：
    - [B,150]
    - [B,1,150]
    - [B,1,1,150]
    """
    if x.dim() == 2 and x.shape[1] == 150:
        x = x.unsqueeze(1).unsqueeze(1)  # [B,1,1,150]
    elif x.dim() == 3 and x.shape[1] == 1 and x.shape[2] == 150:
        x = x.unsqueeze(1)               # [B,1,1,150]
    elif x.dim() == 4 and x.shape[1:] == (1,1,150):
        pass
    else:
        raise AssertionError(f"bad input shape to model: {tuple(x.shape)}")
    return x

def set_bn_momentum(model: nn.Module, momentum: Optional[float] = 0.01):
    if momentum is None: return
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = momentum

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

@torch.no_grad()
def precise_bn(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 200):
    was_training = model.training
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    model.train()
    for d in dropouts: d.eval()  # 关闭 Dropout
    seen = 0
    for inputs, _, _ in loader:
        inputs = ensure_input_hw1(inputs).to(device, non_blocking=True)
        model(inputs); seen += 1
        if seen >= max_batches: break
    if was_training:
        for d in dropouts: d.train(True)
        model.train()
    else:
        for d in dropouts: d.train(False)
        model.eval()

from torch.amp import autocast, GradScaler

def dataset_signature(ds: torch.utils.data.Dataset, k: int = 200) -> Tuple[str, list]:
    paths = []
    n = min(k, len(ds))
    for i in range(n):
        _, _, p = ds[i]; paths.append(str(p))
    s = ";".join(paths).encode("utf-8", errors="ignore")
    return hashlib.sha1(s).hexdigest(), paths[:10]

def print_split_report(train_ds, test_ds):
    ytr, yte = [], []
    for i in range(len(train_ds)): _, y, _ = train_ds[i]; ytr.append(int(y))
    for i in range(len(test_ds)):  _, y, _ = test_ds[i];  yte.append(int(y))
    ctr, cte = Counter(ytr), Counter(yte)
    def ratio(c: Dict[int,int]):
        tot = sum(c.values()); return {k: f"{v} ({v/tot*100:.1f}%)" for k, v in c.items()}
    print("\n===== KS 划分统计（分层检查）=====")
    print("Train label ratio:", ratio(ctr))
    print("Test  label ratio:", ratio(cte))

@torch.no_grad()
def quick_eval_stats(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labs, p0s = [], [], []
    for xb, yb, _ in loader:
        xb = ensure_input_hw1(xb).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb); p = torch.softmax(out, dim=1)
        preds.append(out.argmax(1).cpu()); labs.append(yb.cpu()); p0s.append(p[:,0].detach().cpu())
    pred = torch.cat(preds); lab = torch.cat(labs); p0 = torch.cat(p0s)
    acc = (pred == lab).float().mean().item()
    from collections import Counter as C2
    return acc, dict(C2(pred.tolist())), float(p0.mean().item())

# def train_one_epoch(model: nn.Module,
#                     train_loader: DataLoader,
#                     optimizer: optim.Optimizer,
#                     criterion: nn.Module,
#                     device: torch.device,
#                     use_amp: bool,
#                     ema: Optional[EMA]) -> Tuple[float, float]:
#     model.train()
#     scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
#     running_loss, correct, total = 0.0, 0, 0
#     dev_type = 'cuda' if device.type == 'cuda' else 'cpu'
#     for inputs, labels, _ in train_loader:
#         inputs = ensure_input_hw1(inputs).to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#         optimizer.zero_grad(set_to_none=True)
#         with autocast(dev_type, enabled=(use_amp and device.type == 'cuda')):
#             outputs = model(inputs)
#
#             # ========= Hard Example Mining =========
#             # 找到每个样本的预测结果
#             probs = torch.softmax(outputs, dim=1)  # [B,2]
#             male_conf = probs[:, 1]  # p(male)
#
#             # 对 male(1) 类被模型判断成 female 的“困难样本”放大 loss
#             # 条件：label=1（真公蛋） 且 p_male < 0.5（模型认为是母）
#             hard_mask = (labels == 1) & (male_conf < 0.5)
#
#             if hard_mask.any():
#                 # 使用与 criterion 相同的 class weight（避免额外全局变量）
#                 loss_each_fn = nn.CrossEntropyLoss(
#                     reduction='none',
#                     weight=criterion.weight  # 关键：这里改掉
#                 )
#                 loss_each = loss_each_fn(outputs, labels)  # [B]
#                 loss_each[hard_mask] = loss_each[hard_mask] * 2.0
#                 loss = loss_each.mean()
#             else:
#                 loss = criterion(outputs, labels)
#
#             # loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
#         scaler.step(optimizer)
#         scaler.update()
#         if ema is not None: ema.update(model)
#         running_loss += loss.item()
#         _, pred = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (pred == labels).sum().item()
#     return running_loss / max(1, len(train_loader)), correct / max(1, total)

def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    use_amp: bool,
                    ema: Optional[EMA]) -> Tuple[float, float]:
    model.train()
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')
    running_loss, correct, total = 0.0, 0, 0
    dev_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for inputs, labels, paths in train_loader:   # ⭐ 这里要带上 paths
        inputs = ensure_input_hw1(inputs).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dev_type, enabled=(use_amp and device.type == 'cuda')):
            outputs = model(inputs)         # [B,2]
            probs   = torch.softmax(outputs, dim=1)
            male_conf = probs[:, 1]        # p(male)

            # ========= Hard Example Mining（原来的逻辑）=========
            hard_mask = (labels == 1) & (male_conf < 0.5)   # 真公但模型不太像公的

            # ========= 基于路径的怀疑样本降权 =========
            if SUSPECT_PATH_SET:
                # 当前 batch 中哪些样本在怀疑列表里
                suspect_mask_list = [
                    os.path.normpath(p) in SUSPECT_PATH_SET for p in paths
                ]
                suspect_mask = torch.tensor(
                    suspect_mask_list, dtype=torch.bool, device=device
                )
            else:
                suspect_mask = None

            # ========= 统一用 per-sample loss，方便叠加权重 =========
            loss_each_fn = nn.CrossEntropyLoss(
                reduction='none',
                weight=criterion.weight  # 和整体 class weight 一致
            )
            loss_each = loss_each_fn(outputs, labels)    # [B]

            # 对 Hard male 样本放大 loss（原来你的 *2）
            if hard_mask.any():
                loss_each[hard_mask] = loss_each[hard_mask] * 2.0

            # 对怀疑样本降权
            if suspect_mask is not None and suspect_mask.any():
                loss_each[suspect_mask] = loss_each[suspect_mask] * SUSPECT_LOSS_FACTOR

            loss = loss_each.mean()

        # ====== 反向传播 & EMA ======
        scaler.scale(loss).backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    return running_loss / max(1, len(train_loader)), correct / max(1, total)

@torch.no_grad()
def test_eval(model: nn.Module,
              test_loader: DataLoader,
              criterion: nn.Module,
              device: torch.device,
              file,
              str_time: str,
              use_amp: bool) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    dev_type = 'cuda' if device.type == 'cuda' else 'cpu'
    for inputs, labels, file_path in test_loader:
        inputs = ensure_input_hw1(inputs).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(dev_type, enabled=(use_amp and device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs.data, 1)
        for i in range(probs.size(0)):
            sex = "Female" if probs[i, 0].item() >= probs[i, 1].item() else "Male"
            file.write(f"{str_time},"
                       f"{os.path.splitext(os.path.basename(file_path[i]))[0]},"
                       f"{probs[i, 0].item():.5f},"
                       f"{probs[i, 1].item():.5f},"
                       f"{sex}\n")
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return running_loss / max(1, len(test_loader)), correct / max(1, total)

@torch.no_grad()
def eval_with_ema(model: nn.Module,
                  ema: EMA,
                  train_loader_eval_bn: DataLoader,   # 不带采样器的 loader
                  test_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  str_time: str,
                  use_amp: bool,
                  ftest) -> Tuple[float, float]:
    ema_model = copy.deepcopy(model).to(device)
    ema.copy_to(ema_model)
    if USE_PRECISE_BN:
        precise_bn(ema_model, train_loader_eval_bn, device, max_batches=max(50, PRECISE_BN_BATCHES // 2))
    loss_ema, acc_ema = test_eval(ema_model, test_loader, criterion, device, ftest, str_time, use_amp)
    del ema_model
    return loss_ema, acc_ema

def export_predictions_csv(model, loader, device, save_csv_path):
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    model.eval()
    with open(save_csv_path, "w", encoding="utf-8") as f:
        f.write("file,p_female,p_male,pred\n")
        for xb, yb, paths in loader:
            xb = ensure_input_hw1(xb).to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(xb)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = probs.argmax(axis=1)
            for i, p in enumerate(paths):
                f.write(f"{os.path.basename(p)},{probs[i,0]:.6f},{probs[i,1]:.6f},{int(preds[i])}\n")

# ============================== 主流程 ==============================
def main():
    print(f"Log file => {LOG_FILE}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch:", torch.__version__)
    use_amp = (device.type == 'cuda')

    # KS 划分
    ks_dataset = KSHyperspectralDataset(DATA_FOLDER, train_ratio=0.75)
    train_dataset = ks_dataset.train
    test_dataset  = ks_dataset.test

    total_count = len(train_dataset) + len(test_dataset)
    print("\n===== 数据集划分统计（main） =====")
    print(f"总样本数: {total_count}")
    print(f"训练集数量: {len(train_dataset)} (占比: {len(train_dataset) / max(1, total_count) * 100:.1f}%)")
    print(f"测试集数量: {len(test_dataset)} (占比: {len(test_dataset) / max(1, total_count) * 100:.1f}%)")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 分层报告 + 测试集签名
    print_split_report(train_dataset, test_dataset)
    test_sig, test_head = dataset_signature(test_dataset, k=200)
    print("Test head (10):", test_head)
    print("Test signature:", test_sig)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    main_seed = 42

    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n====== Run {run_idx}/{NUM_RUNS} ======")

        task_seed = main_seed + run_idx
        torch.manual_seed(task_seed); np.random.seed(task_seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(task_seed)

        fold_dir = f"{ROOT_DIR}_{timestamp}//run{run_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        # 训练 DataLoader
        if USE_BALANCED_SAMPLER:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, pin_memory=(device.type == 'cuda'))
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, pin_memory=(device.type == 'cuda'))

        # 评估用的“无采样器”train loader（供 EMA 的 PreciseBN 使用）
        train_loader_eval_bn = DataLoader(
            train_dataset,
            batch_size=min(512, BATCH_SIZE*2),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=(device.type == 'cuda'))

        # 首批护栏
        xb, yb, pb = next(iter(train_loader))
        assert xb.dim() == 4 and xb.shape[1:] == (1,1,150), f"bad shape {xb.shape}"
        assert yb.dtype in (torch.long, torch.int64), f"bad dtype {yb.dtype}"
        if torch.isnan(xb).any() or torch.isinf(xb).any():
            raise ValueError("NaN/Inf detected in input batch")
        print("Train label count (first batch):", Counter(yb.tolist()))
        print("Sample path example:", pb[0] if isinstance(pb[0], str) else str(pb[0]))

        # 模型 & 训练器
        model = HyperspectralCNN(num_classes=2).to(device)
        set_bn_momentum(model, BN_MOMENTUM)
        # ===== 加公类权重，提高公蛋准确率 =====
        # female = 0, male = 1
        class_weights = torch.tensor([1.5, 1.0], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        ema = EMA(model, decay=EMA_DECAY_BASE) if USE_EMA else None

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, verbose=False
        )

        max_accuracy = 0.0
        best_epoch   = -1

        for epoch in range(NUM_EPOCHS):
            # 护栏：固定 Test 集
            cur_sig, _ = dataset_signature(test_loader.dataset, k=200)
            assert cur_sig == test_sig, f"[KS ERROR] Test set changed! {cur_sig} != {test_sig}"

            if ema is not None:
                ema.decay = 0.99 if epoch < 5 else EMA_DECAY_BASE

            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, ema)

            # Raw：PreciseBN + 验证
            if USE_PRECISE_BN:
                precise_bn(model, train_loader, device, max_batches=PRECISE_BN_BATCHES)

            with open(os.path.join(fold_dir, 'test.txt'), 'a', encoding='utf-8') as ftest:
                ftest.write(f'Epoch {epoch + 1}/{NUM_EPOCHS} => \n')

                test_loss_raw, test_acc_raw = test_eval(model, test_loader, criterion, device, ftest, STR_TIME, use_amp)

                # EMA 验证（用 train_loader_eval_bn 做 PreciseBN）
                if ema is not None:
                    test_loss_ema, test_acc_ema = eval_with_ema(
                        model, ema, train_loader_eval_bn, test_loader, criterion, device, STR_TIME, use_amp, ftest
                    )
                else:
                    test_loss_ema, test_acc_ema = test_loss_raw, test_acc_raw

                # 训练集快速诊断（Raw）
                acc_tr_raw, cnt_tr_raw, p0_tr_raw = quick_eval_stats(model, train_loader, device)

            # 日志打印
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} => "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%} | "
                  f"Test Raw: {test_acc_raw:.4%} (loss {test_loss_raw:.4f}) | "
                  f"Test EMA: {test_acc_ema:.4%} (loss {test_loss_ema:.4f})")
            print(f"  [Diag] TrainEval Raw: acc={acc_tr_raw:.4%}, pred_cnt={cnt_tr_raw}, p0_mean={p0_tr_raw:.3f}")

            # 可选：EMA 诊断
            if ema is not None:
                ema_model = copy.deepcopy(model).to(device)
                ema.copy_to(ema_model)
                if USE_PRECISE_BN:
                    precise_bn(ema_model, train_loader_eval_bn, device, max_batches=max(50, PRECISE_BN_BATCHES // 2))
                acc_te_ema, cnt_te_ema, p0_te_ema = quick_eval_stats(ema_model, test_loader, device)
                del ema_model
                print(f"  [Diag] TestEval EMA:  acc={acc_te_ema:.4%}, pred_cnt={cnt_te_ema}, p0_mean={p0_te_ema:.3f}")

            # 调度 & 保存（以 EMA 指标为准）
            scheduler.step(test_loss_ema)

            if test_acc_ema > max_accuracy:
                max_accuracy = test_acc_ema
                best_epoch   = epoch + 1
                save_path = os.path.join(fold_dir, f'best_model_val_{max_accuracy:.4f}.pth')
                save_path_pt = os.path.join(fold_dir, f'best_model_val_{max_accuracy:.4f}.pt')

                if ema is not None:
                    ema_model = copy.deepcopy(model).to(device)
                    ema.copy_to(ema_model)
                    torch.save(ema_model.state_dict(), save_path)
                    # 2) 导出 TorchScript .pt（C#/部署直接加载用）
                    export_torchscript(ema_model, save_path_pt, device)
                    del ema_model
                else:
                    torch.save(model.state_dict(), save_path)
                    export_torchscript(model, save_path_pt, device)
                print(f"[RUN {run_idx}] New best acc(EMA): {max_accuracy:.4f}  -> saved: {save_path}")

                # 同步导出该最佳点的逐样本预测（便于你复核）
                try:
                    if ema is not None:
                        ema_model = copy.deepcopy(model).to(device)
                        ema.copy_to(ema_model)
                        if USE_PRECISE_BN:
                            precise_bn(ema_model, train_loader_eval_bn, device, max_batches=max(50, PRECISE_BN_BATCHES // 2))
                        pred_csv = os.path.join(fold_dir, f'pred_best_epoch{best_epoch}.csv')
                        export_predictions_csv(ema_model, test_loader, device, pred_csv)
                        del ema_model
                    else:
                        pred_csv = os.path.join(fold_dir, f'pred_best_epoch{best_epoch}.csv')
                        export_predictions_csv(model, test_loader, device, pred_csv)
                    print(f"[RUN {run_idx}] Exported per-sample predictions: {pred_csv}")
                except Exception as e:
                    print(f"[WARN] export predictions failed: {e}")

        print(model)

if __name__ == '__main__':
    try:
        main()
        print(f"Log file saved at: {LOG_FILE}")
    except Exception:
        LOGGER.exception("Fatal error:")
        raise
