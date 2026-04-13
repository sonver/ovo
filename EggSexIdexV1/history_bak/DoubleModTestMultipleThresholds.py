# 双模型、交叉多阈值预测

import torch
from torch.utils.data import DataLoader
from train.module_dataset.CNN1DModule_SNV_CBAM_SG_NOTEMP import HyperspectralCNN, CustomDataset_Meng
from tabulate import tabulate
from itertools import product


def test_unseen_data(model_path1, model_path2, test_data_dir, thresholds1=[0.5], thresholds2=[0.5],
                     device='cuda' if torch.cuda.is_available() else 'cpu', save_predictions=False):
    # 设备设置
    device = torch.device(device)
    print(f"Using device: {device}")

    # 加载两个模型
    model1 = HyperspectralCNN(num_bands=150).to(device)
    model2 = HyperspectralCNN(num_bands=150).to(device)
    model1.load_state_dict(torch.load(model_path1, map_location=device))
    model2.load_state_dict(torch.load(model_path2, map_location=device))
    model1.eval()
    model2.eval()

    # 加载测试数据
    test_dataset = CustomDataset_Meng(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 类别名称（与模型训练时一致）
    class_names = ['female', 'male']  # female: 0, male: 1

    # 存储所有阈值组合的结果
    results = []

    # 测试所有阈值组合
    for threshold1, threshold2 in product(thresholds1, thresholds2):
        print(f"\nTesting with thresholds = ({threshold1}, {threshold2})")

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

                # 模型1预测
                outputs1 = model1(data)  # [batch_size, 2]
                probs1 = torch.softmax(outputs1, dim=1)
                female_probs1 = probs1[:, 0]  # female 概率（类别 0）
                predicted1 = 1 - (female_probs1 >= threshold1).long()  # 0: female, 1: male

                # 模型2预测
                outputs2 = model2(data)  # [batch_size, 2]
                probs2 = torch.softmax(outputs2, dim=1)
                female_probs2 = probs2[:, 0]  # female 概率（类别 0）
                predicted2 = 1 - (female_probs2 >= threshold2).long()  # 0: female, 1: male

                # 取并集：若任一模型预测为 female（0），则最终预测为 female
                predicted = torch.logical_or(predicted1 == 0, predicted2 == 0).long()  # 0: female, 1: male
                predicted = 1 - predicted  # 反转：1: female, 0: male（与原逻辑对齐）

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
                            'probability': max(female_probs1[i].item(), female_probs2[i].item()),  # 使用最大概率
                            'correct': predicted[i].item() == labels[i].item()
                        })

        # 计算指标
        accuracy = 100 * correct / total if total > 0 else 0
        female_recall = 100 * female_correct / female_total if female_total > 0 else 0
        male_recall = 100 * male_correct / male_total if male_total > 0 else 0
        female_precision = 100 * predicted_female_correct / predicted_female_total if predicted_female_total > 0 else 0
        male_precision = 100 * predicted_male_correct / predicted_male_total if predicted_male_total > 0 else 0

        # 存储结果
        results.append({
            'Thresholds': f"({threshold1}, {threshold2})",
            'Test Accuracy (%)': f"{accuracy:.2f} ({correct}/{total})",
            'Female Recall (%)': f"{female_recall:.2f} ({female_correct}/{female_total})" if female_total > 0 else "undefined (no female samples)",
            'Male Recall (%)': f"{male_recall:.2f} ({male_correct}/{male_total})",
            'Predicted Female Total': predicted_female_total,
            'Female Precision (%)': f"{female_precision:.2f} ({predicted_female_correct}/{predicted_female_total})",
            'Predicted Male Total': predicted_male_total,
            'Male Precision (%)': f"{male_precision:.2f} ({predicted_male_correct}/{predicted_male_total})"
        })

    # 打印表格
    headers = ['Thresholds', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
               'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
    print("\nResults Table:")
    print(tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid"))

    return results


if __name__ == "__main__":
    # 测试数据目录
    test_data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\mixed_fortest"

    # 两个模型路径
    model_path1 = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin\1DCNN_acc_0.956522_vloss_0.287348.pth"
    model_path2 = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin\1DCNN_f1_0.941736_vacc_0.935441.pth"

    # 每个模型的阈值列表
    thresholds1 = [0.2, 0.3, 0.4]
    thresholds2 = [ 0.4]

    print("Now is testing %s with two models and independent thresholds" % test_data_dir)
    test_unseen_data(model_path1, model_path2, test_data_dir,
                     thresholds1=thresholds1, thresholds2=thresholds2,
                     save_predictions=False)
