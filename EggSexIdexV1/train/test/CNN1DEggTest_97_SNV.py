# 单模型预测

import torch
from torch.utils.data import DataLoader
from train.module_dataset.CNN1DModule_97_VERSION import HyperspectralCNN, CustomDataset_Meng
from tabulate import tabulate


# 测试函数
def test_unseen_data(model_path, test_data_dir, thresholds=[0.5], device='cpu',
                     save_predictions=False):
    # 设备设置
    device = torch.device(device)
    print(f"Using device: {device}")

    # 加载模型
    model = HyperspectralCNN(num_bands=150).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载测试数据
    test_dataset = CustomDataset_Meng(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 类别名称（与模型训练时一致）
    class_names = ['female', 'male']  # female: 0, male: 1

    # 存储所有阈值的结果
    results = []

    for threshold in thresholds:
        print(f"\nTesting with threshold = {threshold}")

        # 统计指标
        correct = 0
        total = 0
        female_correct = 0
        female_total = 0
        male_correct = 0
        male_total = 0
        predicted_female_total = 0
        predicted_female_correct = 0
        predicted_male_total = 0
        predicted_male_correct = 0
        predictions = [] if save_predictions else None

        with torch.no_grad():
            for batch_idx, (data, labels, file_paths) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)

                # 模型预测
                outputs = model(data)  # [batch_size, 2]
                probs = torch.softmax(outputs, dim=1)  # [batch_size, 2]
                female_probs = probs[:, 0]  # 取 female 的概率（类别 0）
                predicted = (female_probs >= threshold).long()  # 0: female, 1: male
                predicted = 1 - predicted  # 反转预测标签：1: female, 0: male（与第一个代码对齐）

                # 总体统计
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 按类别统计（向量化操作）
                female_mask = (labels == 0)  # 真实为 female
                male_mask = (labels == 1)  # 真实为 male
                pred_female_mask = (predicted == 0)  # 预测为 female
                pred_male_mask = (predicted == 1)  # 预测为 male

                female_total += female_mask.sum().item()
                male_total += male_mask.sum().item()
                female_correct += (female_mask & (predicted == labels)).sum().item()
                male_correct += (male_mask & (predicted == labels)).sum().item()

                predicted_female_total += pred_female_mask.sum().item()
                predicted_male_total += pred_male_mask.sum().item()
                predicted_female_correct += (pred_female_mask & (labels == 0)).sum().item()
                predicted_male_correct += (pred_male_mask & (labels == 1)).sum().item()

                # 可选：保存预测结果
                if save_predictions:
                    for i in range(len(file_paths)):
                        pred_label = 'female' if predicted[i].item() == 0 else 'male'
                        true_label = 'female' if labels[i].item() == 0 else 'male'
                        predictions.append({
                            'file': file_paths[i],
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'probability': female_probs[i].item(),
                            'correct': predicted[i].item() == labels[i].item()
                        })

        # 计算指标
        accuracy = 100 * correct / total
        female_recall = 100 * female_correct / female_total if female_total > 0 else 0
        male_recall = 100 * male_correct / male_total if male_total > 0 else 0
        female_precision = 100 * predicted_female_correct / predicted_female_total if predicted_female_total > 0 else 0
        male_precision = 100 * predicted_male_correct / predicted_male_total if predicted_male_total > 0 else 0

        # 存储结果
        results.append({
            'Threshold': threshold,
            'Test Accuracy (%)': f"{accuracy:.2f} ({correct}/{total})",
            'Female Recall (%)': f"{female_recall:.2f} ({female_correct}/{female_total})",
            'Male Recall (%)': f"{male_recall:.2f} ({male_correct}/{male_total})",
            'Predicted Female Total': predicted_female_total,
            'Female Precision (%)': f"{female_precision:.2f} ({predicted_female_correct}/{predicted_female_total})",
            'Predicted Male Total': predicted_male_total,
            'Male Precision (%)': f"{male_precision:.2f} ({predicted_male_correct}/{predicted_male_total})"
        })

    # 打印表格
    headers = ['Threshold', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
               'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
    print("\nResults Table:")
    print(tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid"))

    return results


if __name__ == "__main__":
    # 测试数据目录
    test_data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000"
    model_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\B_TRAIN_RESULT\1DCNN_SNV_HLH_VC_huayu13d12h_8000_715\3\1DCNN_acc_0.979348_vloss_0.011912.pth"

    # 运行测试，测试不同阈值
    thresholds = [0.15, 0.25, 0.35, 0.4,0.45, 0.5, 0.6, 0.7]

    print("now is testing %s", test_data_dir)
    test_unseen_data(model_path, test_data_dir, thresholds=thresholds, save_predictions=False)
