import os
import torch
from torch.utils.data import DataLoader
from train.module_dataset.CNN1DModule_97_VERSION import HyperspectralCNN, CustomDataset_Meng
from tabulate import tabulate


def test_unseen_data(models_dir, test_data_dir, thresholds=[0.5],
                     device='cuda' if torch.cuda.is_available() else 'cpu',
                     save_predictions=False):
    device = torch.device(device)
    print(f"Using device: {device}")

    # 加载目录下的所有 .pth 模型
    model_paths = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not model_paths:
        raise ValueError(f"No .pth models found in {models_dir}")

    models = []
    for path in model_paths:
        model = HyperspectralCNN(num_bands=150).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded model: {path}")

    test_dataset = CustomDataset_Meng(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    results = []

    for threshold in thresholds:
        print(f"\nTesting with threshold = {threshold}")

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

                # ---------- 多模型：收集每个模型的 female 概率与布尔预测 ----------
                all_probs = []           # list of tensors [batch_size]
                all_female_bool = []     # list of bool tensors [batch_size]
                for model in models:
                    outputs = model(data)            # [batch_size, 2]
                    probs = torch.softmax(outputs, dim=1)
                    female_probs = probs[:, 0]      # female 概率
                    all_probs.append(female_probs)
                    all_female_bool.append(female_probs >= threshold)

                # OR 融合：只要任意模型认为是 female，则最终为 female
                female_any = torch.stack(all_female_bool, dim=0).any(dim=0)  # bool tensor, True = female
                # 统一映射为与标签一致： 0 = female, 1 = male
                predicted = 1 - female_any.long()  # 0: female, 1: male

                # 调试输出：前几条样本的各模型 female probs
                for i in range(len(file_paths)):
                    egg_idx = batch_idx * test_loader.batch_size + i + 1
                    probs_str = " | ".join([f"{p[i].item():.4f}" for p in all_probs])
                    print(f"Egg {egg_idx} (file: {file_paths[i]}): Female probs from models: {probs_str} -> final_pred: {'female' if predicted[i].item()==0 else 'male'} , true: {'female' if labels[i].item()==0 else 'male'}")

                # ---------- 统计 ----------
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                female_mask = (labels == 0)
                male_mask = (labels == 1)
                pred_female_mask = (predicted == 0)
                pred_male_mask = (predicted == 1)

                female_total += female_mask.sum().item()
                male_total += male_mask.sum().item()
                female_correct += (female_mask & (predicted == labels)).sum().item()
                male_correct += (male_mask & (predicted == labels)).sum().item()

                predicted_female_total += pred_female_mask.sum().item()
                predicted_male_total += pred_male_mask.sum().item()
                predicted_female_correct += (pred_female_mask & (labels == 0)).sum().item()
                predicted_male_correct += (pred_male_mask & (labels == 1)).sum().item()

                if save_predictions:
                    for i in range(len(file_paths)):
                        pred_label = 'female' if predicted[i].item() == 0 else 'male'
                        true_label = 'female' if labels[i].item() == 0 else 'male'
                        max_prob = max([p[i].item() for p in all_probs])
                        predictions.append({
                            'file': file_paths[i],
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'probability': max_prob,
                            'correct': predicted[i].item() == labels[i].item()
                        })

                # （可选）在首批次打印一下标签分布，帮助检查是否与预期一致
                if batch_idx == 0:
                    print("First-batch label counts:", torch.bincount(labels.cpu()).tolist())
                    print("First-batch predicted counts:", torch.bincount(predicted.cpu()).tolist())

        # 计算指标
        accuracy = 100 * correct / total if total > 0 else 0
        female_recall = 100 * female_correct / female_total if female_total > 0 else 0
        male_recall = 100 * male_correct / male_total if male_total > 0 else 0
        female_precision = 100 * predicted_female_correct / predicted_female_total if predicted_female_total > 0 else 0
        male_precision = 100 * predicted_male_correct / predicted_male_total if predicted_male_total > 0 else 0

        results.append({
            'Threshold': threshold,
            'Test Accuracy (%)': f"{accuracy:.2f} ({correct}/{total})",
            'Female Recall (%)': f"{female_recall:.2f} ({female_correct}/{female_total})" if female_total > 0 else "undefined (no female samples)",
            'Male Recall (%)': f"{male_recall:.2f} ({male_correct}/{male_total})",
            'Predicted Female Total': predicted_female_total,
            'Female Precision (%)': f"{female_precision:.2f} ({predicted_female_correct}/{predicted_female_total})",
            'Predicted Male Total': predicted_male_total,
            'Male Precision (%)': f"{male_precision:.2f} ({predicted_male_correct}/{predicted_male_total})"
        })

    headers = ['Threshold', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
               'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
    print("\nResults Table:")
    print(tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid"))

    return results


if __name__ == "__main__":
    models_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin"
    test_data_dir = (r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\half\huayu_13d1"
                     r"2h_4000_half")
    thresholds = [0.2,0.3,0.35,0.4,0.45,0.5, 0.6,0.7]
    print(f"Now testing {test_data_dir} with models from {models_dir}")
    test_unseen_data(models_dir, test_data_dir, thresholds=thresholds, save_predictions=False)
