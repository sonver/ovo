import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from train.module_dataset.CNN1DModule_SNV_CBAM import HyperspectralCNN
from tools.tools import snv_normalize

def gaussian_weights(data, sigma):
    mean = np.mean(data)
    weights = np.exp(-((data - mean) ** 2) / (2 * sigma ** 2))
    return weights

class CustomDatasetTxt(Dataset):
    def __init__(self, data_folder, target_height=900, num_bands=150):
        self.data_folder = data_folder
        self.target_height = target_height
        self.num_bands = num_bands
        self.file_paths = []

        for file_name in os.listdir(data_folder):
            if file_name.endswith('.txt'):
                file_path = os.path.join(data_folder, file_name)
                self.file_paths.append(file_path)

        if len(self.file_paths) == 0:
            raise ValueError(f"在 {data_folder} 中未找到 .txt 文件")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
        data = np.array(data)

        if data.shape[1] != self.num_bands:
            raise ValueError(f"文件 {file_path} 的波段数 {data.shape[1]} 与预期 {self.num_bands} 不匹配")
        if data.shape[0] != self.target_height:
            raise ValueError(f"文件 {file_path} 的高度 {data.shape[0]} 与预期 {self.target_height} 不匹配")

        snv_data = np.apply_along_axis(snv_normalize, 1, data)
        result = []
        for col in range(snv_data.shape[1]):
            column_data = snv_data[:, col]
            sigma = np.std(column_data) * 0.5 + 0.5
            weights = gaussian_weights(column_data, sigma=sigma)
            if np.sum(weights) == 0:
                weighted_sum = 0
            else:
                weighted_sum = np.sum(column_data * weights) / np.sum(weights)
            result.append(weighted_sum)
        result = np.array(result).reshape(1, -1)

        data_tensor = torch.tensor(result, dtype=torch.float32)
        data_tensor = data_tensor.view(1, 1, 150)
        return data_tensor, torch.tensor(0, dtype=torch.long), file_path

def predict(model, data_loader, device, female_files, male_files, threshold=0.5):
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    file_paths = []

    with torch.no_grad():
        for inputs, _, file_path in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            female_probs = probs[:, 1]  # 类别 1 为 female
            predicted = (female_probs >= threshold).long()  # 0: female, 1: male
            predicted = 1 - predicted  # 反转：1: female, 0: male

            # 转换为 numpy 数组
            female_probs = female_probs.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()

            batch_labels = []
            for path in file_path:
                file_name = os.path.basename(path)
                if file_name in female_files:
                    batch_labels.append(1)  # female 对应 1
                elif file_name in male_files:
                    batch_labels.append(0)  # male 对应 0
                else:
                    raise ValueError(f"文件 {file_name} 未在 female 或 male 文件夹中找到")

            predictions.extend(predicted)
            probabilities.extend(female_probs)
            true_labels.extend(batch_labels)
            file_paths.extend(file_path)

    return np.array(predictions), np.array(probabilities), np.array(true_labels), file_paths

def main():
    female_folder = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187test\female"
    male_folder = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187test\male"

    female_files = set()
    male_files = set()

    for file_name in os.listdir(female_folder):
        if file_name.endswith('.txt'):
            female_files.add(file_name)

    for file_name in os.listdir(male_folder):
        if file_name.endswith('.txt'):
            male_files.add(file_name)

    overlap = female_files.intersection(male_files)
    if overlap:
        raise ValueError(f"以下文件同时出现在 female 和 male 文件夹中，请检查数据：{overlap}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_root = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\resnet\MQ1215_50_g_400epoch\MQ715_1DCNN_SNV_0'
    model_name = '1DCNN_0.466110_vacc_0.815652.pth'
    model_path = os.path.join(model_root, model_name)
    n_bands = 150
    model = HyperspectralCNN(num_bands=n_bands).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mixed_data_folder = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187mixed"
    mixed_dataset = CustomDatasetTxt(mixed_data_folder, target_height=900, num_bands=n_bands)
    batch_size = 128
    mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # 定义阈值数组
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85]

    output_dir = 'D:/workspace/gdv-egg-model/Code_EggGenderDet/predictions'
    os.makedirs(output_dir, exist_ok=True)

    # 存储表格数据
    results = []

    with open(os.path.join(output_dir, "mixed_predictions_" + model_name + ".txt"), 'w', encoding='utf-8') as f:
        # 对每个阈值进行测试
        for threshold in thresholds:
            y_pred, y_probs, y_true, file_paths = predict(model, mixed_loader, device, female_files, male_files, threshold)

            classification = classification_report(y_true, y_pred, digits=8, zero_division=0,
                                                   target_names=['male', 'female'])
            accuracy = np.mean(y_pred == y_true)
            correct = np.sum(y_pred == y_true)
            total = len(y_true)

            cm = confusion_matrix(y_true, y_pred)
            correct_male = cm[0, 0]  # 类别 0 为 male
            correct_female = cm[1, 1]  # 类别 1 为 female
            total_male = np.sum(y_true == 0)
            total_female = np.sum(y_true == 1)

            predicted_male_total = np.sum(y_pred == 0)
            predicted_female_total = np.sum(y_pred == 1)

            # 计算召回率和精确率
            female_recall = correct_female / total_female if total_female > 0 else 0
            male_recall = correct_male / total_male if total_male > 0 else 0
            female_precision = correct_female / predicted_female_total if predicted_female_total > 0 else 0
            male_precision = correct_male / predicted_male_total if predicted_male_total > 0 else 0

            # 存储表格数据
            results.append({
                'Threshold': threshold,
                'Test Accuracy (%)': f"{accuracy * 100:.2f} ({correct}/{total})",
                'Female Recall (%)': f"{female_recall * 100:.2f} ({correct_female}/{total_female})",
                'Male Recall (%)': f"{male_recall * 100:.2f} ({correct_male}/{total_male})",
                'Predicted Female Total': predicted_female_total,
                'Female Precision (%)': f"{female_precision * 100:.2f} ({correct_female}/{predicted_female_total})",
                'Predicted Male Total': predicted_male_total,
                'Male Precision (%)': f"{male_precision * 100:.2f} ({correct_male}/{predicted_male_total})"
            })

            # 写入文件
            f.write(f"\n=== 阈值: {threshold:.2f} ===\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write("分类报告:\n")
            f.write(classification + "\n")
            f.write(f"male 正确预测数量: {correct_male}/{total_male}\n")
            f.write(f"female 正确预测数量: {correct_female}/{total_female}\n")
            f.write(f"预测为 male 的数量: {predicted_male_total}\n")
            f.write(f"预测为 female 的数量: {predicted_female_total}\n")

        # 打印表格
        headers = ['Threshold', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
                   'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
        print("\nResults Table:")
        print(tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid"))

        # 逐样本预测结果（使用默认阈值 0.5）
        f.write("\n逐样本预测结果 (阈值 0.5):\n")
        y_pred, y_probs, y_true, file_paths = predict(model, mixed_loader, device, female_files, male_files, threshold=0.5)
        for i in range(len(y_true)):
            f.write(
                f"文件: {file_paths[i]}, 真实标签: {y_true[i]}, 预测标签: {y_pred[i]}, 概率: {y_probs[i]:.4f}\n")

if __name__ == "__main__":
    main()