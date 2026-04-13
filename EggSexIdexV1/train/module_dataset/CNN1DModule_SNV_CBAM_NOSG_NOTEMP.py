import os
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from tools.tools import gaussian_weights, snv_normalize

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        # alpha per target
        alpha_t = torch.tensor([self.alpha, 1.0 - self.alpha], device=inputs.device)[targets]
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch_size, num_pixels, input_dim]
        attn_scores = self.attention(x).squeeze(-1)  # [batch_size, num_pixels]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, num_pixels, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch_size, input_dim]
        return pooled

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CBAM1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM1D, self).__init__()
        # 通道注意力
        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.channel_max = nn.AdaptiveMaxPool1d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        # 空间注意力
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        batch, channels, length = x.size()
        # 通道注意力
        avg_pool = self.channel_avg(x).view(batch, channels)
        max_pool = self.channel_max(x).view(batch, channels)
        channel_weight = torch.sigmoid(self.channel_fc(avg_pool) + self.channel_fc(max_pool))
        channel_weight = channel_weight.view(batch, channels, 1)
        x = x * channel_weight
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = torch.sigmoid(self.spatial_conv(spatial_weight))
        x = x * spatial_weight
        return x


class HyperspectralCNN(nn.Module):
    def __init__(self, num_bands=150, use_attention_pooling=False):
        super(HyperspectralCNN, self).__init__()
        self.num_bands = num_bands
        self.use_attention_pooling = use_attention_pooling

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.cbam1 = CBAM1D(64, reduction=8)
        self.res1 = ResidualBlock1D(64, 128)
        self.res2 = ResidualBlock1D(128, 256)
        self.cbam2 = CBAM1D(256, reduction=8)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 2)

        # 新增 attention pooling
        self.attn_pool = SelfAttentionPooling(256)

    def forward(self, x):
        batch_size, height, width, num_bands = x.size()
        num_pixels = height * width

        x = x.view(batch_size * num_pixels, 1, num_bands)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.cbam2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = x.view(batch_size * num_pixels, -1)

        x = x.view(batch_size, num_pixels, -1)  # [batch, pixels, features]

        if self.use_attention_pooling:
            # 使用 self-attention pooling
            x = self.attn_pool(x)  # [batch, features]
        else:
            # 原来的 mean pooling
            x = x.mean(dim=1)  # [batch, features]

        output = self.fc(x)
        return output



# 调整SNV顺序 SNV -> 高斯权重
class CustomDataset_Meng(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(self.classes)
        self.file_list = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
            self.file_list.extend([os.path.join(class_dir, f) for f in files])
            self.labels.extend([i] * len(files))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # --------------- 修改后的处理流程 --------------
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
        data = np.array(data)  # 原始数据形状: [num_points, num_bands]

        # 以下是原来的方法:
        # 步骤1: 先进行SNV归一化（逐样本点）
        # 对每个样本点（每一行）的波段数据做SNV
        snv_data = np.apply_along_axis(snv_normalize, 1, data)  # 输出形状: [num_points, num_bands]

        result = []
        sigma = 1.0
        for col in range(snv_data.shape[1]):
            column_data = snv_data[:, col]  # 提取SNV后的波段列数据
            # 计算高斯权重（基于SNV后的数据）
            weights = gaussian_weights(column_data, sigma=sigma)
            if np.sum(weights) == 0:
                weighted_sum = 0.0
            else:
                weighted_sum = np.sum(column_data * weights) / np.sum(weights)
            result.append(weighted_sum)
        result = np.array(result).reshape(1, -1)  # 形状: [1, num_bands]

        # TODO 这个是新增的
        # result = result - np.mean(result)  # 减去均值，使数据居中于0

        data_tensor = torch.tensor(result, dtype=torch.float32)
        data_tensor = data_tensor.unsqueeze(0)

        # data_tensor = data_tensor.view(1,1,-1)  # 适配1D CNN输入: [batch, channels, length]

        return data_tensor, label, file_path
