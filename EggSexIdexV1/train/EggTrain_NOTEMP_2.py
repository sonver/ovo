import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch.optim as optim
from tabulate import tabulate

from train.module_dataset.CNN1DModule_SNV_CBAM import *

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


def test(model, test_loader, criterion, device, thresholds=[0.5], output_file='test.txt', fold_path=None):
    model.eval()
    class_names = ['female', 'male']
    results = []
    if fold_path:
        os.makedirs(fold_path, exist_ok=True)

    for threshold in thresholds:
        running_loss = 0.0
        stats = {'correct': 0, 'total': 0, 'female_correct': 0, 'female_total': 0,
                 'male_correct': 0, 'male_total': 0, 'pred_female_total': 0, 'pred_female_correct': 0,
                 'pred_male_total': 0, 'pred_male_correct': 0}
        y_pred, y_true = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                running_loss += criterion(outputs, labels).item()
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

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(test_loader) if len(test_loader) else float('inf')
        accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] else 0
        female_recall = 100 * stats['female_correct'] / stats['female_total'] if stats['female_total'] else 0
        male_recall = 100 * stats['male_correct'] / stats['male_total'] if stats['male_total'] else 0
        female_precision = 100 * stats['pred_female_correct'] / stats['pred_female_total'] if stats[
            'pred_female_total'] else 0
        male_precision = 100 * stats['pred_male_correct'] / stats['pred_male_total'] if stats['pred_male_total'] else 0
        cls_report = classification_report(y_true, y_pred, digits=8, zero_division=0, target_names=class_names)

        results.append({
            'Threshold': threshold,
            'Test Accuracy (%)': f"{accuracy:.2f} ({stats['correct']}/{stats['total']})",
            'Female Recall (%)': f"{female_recall:.2f} ({stats['female_correct']}/{stats['female_total']})",
            'Male Recall (%)': f"{male_recall:.2f} ({stats['male_correct']}/{stats['male_total']})",
            'Predicted Female Total': stats['pred_female_total'],
            'Female Precision (%)': f"{female_precision:.2f} ({stats['pred_female_correct']}/{stats['pred_female_total']})",
            'Predicted Male Total': stats['pred_male_total'],
            'Male Precision (%)': f"{male_precision:.2f} ({stats['pred_male_correct']}/{stats['pred_male_total']})",
            'Classification Report': cls_report
        })

        if fold_path:
            with open(os.path.join(fold_path, output_file), 'a', encoding='utf-8') as f:
                f.write(f"\nThreshold: {threshold}\nTest Loss: {avg_loss:.6f}\n"
                        f"Test Accuracy: {accuracy:.2f}% ({stats['correct']}/{stats['total']})\n"
                        f"Female Recall: {female_recall:.2f}% ({stats['female_correct']}/{stats['female_total']})\n"
                        f"Male Recall: {male_recall:.2f}% ({stats['male_correct']}/{stats['male_total']})\n"
                        f"Female Precision: {female_precision:.2f}% ({stats['pred_female_correct']}/{stats['pred_female_total']})\n"
                        f"Male Precision: {male_precision:.2f}% ({stats['pred_male_correct']}/{stats['pred_male_total']})\n"
                        f"Classification Report:\n{cls_report}\n")

    if results:
        headers = ['Threshold', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
                   'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
        with open(os.path.join(fold_path, output_file), 'a', encoding='utf-8') as f:
            f.write(f"\nTest Results Table ({output_file}):\n"
                    f"{tabulate([[r[h] for h in headers] for r in results], headers=headers, tablefmt='grid')}")

    return results


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(np.argmax(outputs.cpu().numpy(), axis=1))
            y_true.extend(labels.cpu().numpy())

    cls_dict = classification_report(y_true, y_pred, digits=8, zero_division=0, target_names=['female', 'male'],
                                     output_dict=True)
    cls_report = classification_report(y_true, y_pred, digits=8, zero_division=0, target_names=['female', 'male'])
    return (running_loss / len(val_loader), correct / total, cls_report,
            cls_dict['female']['recall'], cls_dict['female']['f1-score'])


def EggNetMain():
    if not os.path.exists(data_folder):
        print(f"数据目录 {data_folder} 不存在")
        return

    try:
        dataset = CustomDataset_Meng(data_folder)
        print(f"数据集样本数: {len(dataset)}, CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_samples = female_samples + male_samples  # female + male
    female_weight = (total_samples / female_samples) * female_weight_times # 增加 female 权重（例如 1.2 倍）
    male_weight = total_samples / male_samples
    class_weights = torch.tensor([female_weight, male_weight], dtype=torch.float).to(device)
    print(f"类权重: {class_weights}")

    for i in range(20):
        print(f'运行 {i + 1}/20')
        torch.manual_seed(42 + i)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size,
                                                                            len(dataset) - train_size - val_size])

        fold1 = os.path.join(fold, str(i))
        os.makedirs(fold1, exist_ok=True)

        try:
            train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=8)
            test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=8)
        except Exception as e:
            print(f"数据加载器创建失败: {e}")
            continue
        model = HyperspectralCNN(num_bands=n_bands, use_attention_pooling=True).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=7)

        best_val_acc = 0  # 修改：跟踪最佳验证准确率
        max_f1 = 0
        counter = 0
        train_losses, val_losses = [], []
        best_val_loss = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            print(f'轮次 {epoch + 1}/{num_epochs}: 训练损失: {train_loss:.6f}, 准确率: {train_acc:.4%}')

            val_loss, val_acc, cls_report, female_recall, female_f1 = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            with open(os.path.join(fold1, 'val.txt'), 'a', encoding='utf-8') as f:
                f.write(f'轮次 {epoch + 1}: 验证损失: {val_loss:.6f}, 准确率: {val_acc:.4%}, '
                        f'雌性召回率: {female_recall:.6f}, F1: {female_f1:.6f}\n{cls_report}\n')
            print(f'验证损失: {val_loss:.6f}, 准确率: {val_acc:.4%}')

            # 修改：保存验证准确率最高的模型
            if val_acc > best_val_acc and val_acc > 0.9:
                best_val_acc = val_acc
                best_val_loss = val_loss  # 记录对应的验证损失
                torch.save(model.state_dict(),
                           os.path.join(fold1, f'1DCNN_acc_{best_val_acc:.6f}_vloss_{best_val_loss:.6f}.pth'))
                counter = 0
            else:
                counter += 1

            # 保存F1分数最高的模型（保持原逻辑）
            if female_recall > 0.96 and female_f1 > 0.94 and female_f1 > max_f1:
                max_f1 = female_f1
                torch.save(model.state_dict(), os.path.join(fold1, f'1DCNN_f1_{max_f1:.6f}_vacc_{val_acc:.6f}.pth'))

            if counter >= patience:
                print(f"轮次 {epoch + 1} 提前停止")
                break

        try:
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.legend()
            plt.savefig(os.path.join(fold1, 'loss_plot.png'))
            plt.close()
        except Exception as e:
            print(f"保存损失曲线失败: {e}")

        # 修改：测试验证准确率最高和F1分数最高的模型
        for model_type, file_name in [('val_acc', 'test_acc.txt'), ('f1_score', 'test_recall_f1.txt')]:
            model_path = os.path.join(fold1,
                                      f'1DCNN_acc_{best_val_acc:.6f}_vloss_{best_val_loss:.6f}.pth' if model_type == 'val_acc' else
                                      f'1DCNN_f1_{max_f1:.6f}_vacc_{best_val_acc:.6f}.pth' if max_f1 else '')
            if model_path and os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"测试模型: {model_path}")
                    test(model, test_loader, criterion, device, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                         output_file=file_name, fold_path=fold1)
                except Exception as e:
                    print(f"测试模型 {model_path} 失败: {e}")


def to_torchScript():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\1DCNN_200epoch_SNV\jh_delete_gt_4080_70_15_15_ATT\1'
    try:
        model = HyperspectralCNN(num_bands=150, use_attention_pooling=True).to(device)
        model.load_state_dict(torch.load(os.path.join(root_path, '1DCNN_acc_0.939361_vloss_0.303291.pth')))
        torch.jit.script(model).save(os.path.join(root_path, 'JH_93.pt'))
    except Exception as e:
        print(f"模型转换失败: {e}")


if __name__ == '__main__':
    # Dataloader批处理数量
    batch_size = 128
    # 初始学习率
    lr = 3e-4
    # 折数
    num_epochs = 200
    # 忍耐轮数
    patience = 30
    # 公母样本数量
    female_samples, male_samples =  2958, 3095
    # female的权重倍率
    female_weight_times = 1.2
    # 波段数量
    n_bands = 150
    # 注意力池化，加大female权重 1.4倍。
    data_folder = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\huayu"
    fold = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\1DCNN_200epoch_SNV\jh_delete_gt_4080_70_15_15'  # 母多公少，half，16*16
    EggNetMain()

    # to_torchScript()
