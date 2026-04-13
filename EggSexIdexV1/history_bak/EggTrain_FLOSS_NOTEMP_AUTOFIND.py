import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import pandas as pd

from train.module_dataset.CNN1DModule_SNV_CBAM import HyperspectralCNN, CustomDataset_Meng, FocalLoss

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = correct = total = 0
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), correct / total


def evaluate(model, loader, device, thresholds=[0.5]):
    """评估函数，支持多个阈值"""
    model.eval()
    all_results = []

    with torch.no_grad():
        for threshold in thresholds:
            stats = {'correct': 0, 'total': 0,
                     'female_correct': 0, 'female_total': 0,
                     'male_correct': 0, 'male_total': 0,
                     'pred_female_total': 0, 'pred_female_correct': 0,
                     'pred_male_total': 0, 'pred_male_correct': 0}

            for inputs, labels, _ in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                female_probs = probs[:, 0]
                predicted = 1 - (female_probs >= threshold).long()

                stats['total'] += labels.size(0)
                stats['correct'] += (predicted == labels).sum().item()

                female_mask = (labels == 0)
                male_mask = (labels == 1)
                pred_female_mask = (predicted == 0)
                pred_male_mask = (predicted == 1)

                stats['female_total'] += female_mask.sum().item()
                stats['male_total'] += male_mask.sum().item()
                stats['female_correct'] += (female_mask & (predicted == labels)).sum().item()
                stats['male_correct'] += (male_mask & (predicted == labels)).sum().item()
                stats['pred_female_total'] += pred_female_mask.sum().item()
                stats['pred_male_total'] += pred_male_mask.sum().item()
                stats['pred_female_correct'] += (pred_female_mask & (labels == 0)).sum().item()
                stats['pred_male_correct'] += (pred_male_mask & (labels == 1)).sum().item()

            accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] else 0
            female_recall = 100 * stats['female_correct'] / stats['female_total'] if stats['female_total'] else 0
            male_recall = 100 * stats['male_correct'] / stats['male_total'] if stats['male_total'] else 0
            female_precision = 100 * stats['pred_female_correct'] / stats['pred_female_total'] if stats[
                'pred_female_total'] else 0
            male_precision = 100 * stats['pred_male_correct'] / stats['pred_male_total'] if stats[
                'pred_male_total'] else 0

            all_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'female_recall': female_recall,
                'male_recall': male_recall,
                'female_precision': female_precision,
                'male_precision': male_precision
            })

    return all_results


def find_best_threshold(model, loader, device, thresholds=np.linspace(0.1, 0.9, 17)):
    """自动寻找最佳阈值"""
    results = evaluate(model, loader, device, thresholds)

    # 首先尝试寻找同时满足高召回率和高精确度的阈值
    filtered = [r for r in results if r['female_recall'] >= 97 and r['male_precision'] >= 95]

    if filtered:
        # 如果有满足条件的阈值，选择准确率最高的
        best = max(filtered, key=lambda x: x['accuracy'])
    else:
        # 如果没有满足条件的，优先保证女性召回率
        best = max(results, key=lambda x: x['female_recall'])

    return best, results


def validate(model, loader, device):
    """单阈值验证函数"""
    results = evaluate(model, loader, device, thresholds=[0.5])
    if results:
        res = results[0]
        return res['accuracy'], res['female_recall'], res['female_precision'], res['male_precision']
    return 0, 0, 0, 0


def EggNetMain():
    if not os.path.exists(data_folder):
        print(f"数据目录 {data_folder} 不存在")
        return

    dataset = CustomDataset_Meng(data_folder)
    print(f"数据集样本数: {len(dataset)}, CUDA: {torch.cuda.is_available()}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    summary_records = []

    for i in range(20):
        print(f'运行 {i + 1}/20')
        torch.manual_seed(42 + i)

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, len(dataset) - train_size - val_size]
        )

        # WeightedRandomSampler 保证 batch 内 female:male ≈ 6:4
        train_labels = [train_ds.dataset[idx][1] for idx in train_ds.indices]
        weights = [0.6 if l == 0 else 0.4 for l in train_labels]  # female=0, male=1
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8)

        model = HyperspectralCNN(num_bands=n_bands, use_attention_pooling=True).to(device)

        criterion = FocalLoss(alpha=0.7, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)

        best_acc = 0
        best_val_recall = 0
        best_model_path = None  # 保存最佳模型路径
        patience_counter = 0

        fold1 = os.path.join(fold, str(i))
        os.makedirs(fold1, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_acc, female_recall, female_precision, male_precision = validate(model, val_loader, device)

            scheduler.step(1 - val_acc)

            print(f"Epoch {epoch + 1}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.2f}, "
                  f"F-Rec={female_recall:.2f}, M-Prec={male_precision:.2f}")

            # 保存 best acc 模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(fold1, f"best_acc_{val_acc:.2f}.pth")
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            # 保存满足 recall & precision 条件的模型
            elif female_recall >= 97 and male_precision >= 95 and val_acc > 90:
                best_val_recall = female_recall
                torch.save(model.state_dict(),
                           os.path.join(fold1,
                                        f"best_recall_{female_recall:.2f}_mprec_{male_precision:.2f}_acc_{val_acc:.2f}.pth"))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("早停")
                break

        # 加载最佳模型进行阈值搜索
        if best_model_path and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"加载最佳模型进行阈值搜索: {best_model_path}")

            # 在验证集上寻找最佳阈值
            best_val, _ = find_best_threshold(model, val_loader, device)

            # 在测试集上使用最佳阈值评估
            best_test, _ = find_best_threshold(model, test_loader, device, thresholds=[best_val['threshold']])

            # 保存最佳阈值结果
            with open(os.path.join(fold1, "best_threshold.txt"), "w") as f:
                f.write(f"Val Best: {best_val}\nTest Best: {best_test}\n")

            # 记录结果
            summary_records.append({
                "fold": i,
                "best_acc": best_acc,
                "best_val_recall": best_val_recall,
                "val_best_threshold": best_val['threshold'],
                "val_frecall": best_val['female_recall'],
                "val_mprecision": best_val['male_precision'],
                "test_best_threshold": best_test['threshold'],
                "test_frecall": best_test['female_recall'],
                "test_mprecision": best_test['male_precision']
            })
        else:
            summary_records.append({
                "fold": i,
                "best_acc": best_acc,
                "best_val_recall": best_val_recall
            })

    pd.DataFrame(summary_records).to_csv(os.path.join(fold, "summary.csv"), index=False)


if __name__ == '__main__':
    batch_size = 128
    lr = 3e-4
    num_epochs = 200
    patience = 30
    n_bands = 150

    data_folder = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\mixed"
    fold = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\1DCNN_FR97_MP94_Mixed\NO_TEMP"

    EggNetMain()