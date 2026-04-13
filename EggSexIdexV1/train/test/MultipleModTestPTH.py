import torch
from torch.utils.data import DataLoader
from train.module_dataset.CNN1DModule_SNV_CBAM_WITH_T import HyperspectralCNN, CustomDataset_Meng, ModelWithTemperature
from tabulate import tabulate
import os
import glob


def load_model_with_temperature(model_path, device, default_temp=1.0):
    """加载模型及其温度参数（若存在 _temperature.txt 文件则使用）"""
    # 加载基础模型
    base_model = HyperspectralCNN(num_bands=150, use_attention_pooling=True).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.eval()

    # 查找温度参数文件
    temp_file = model_path.replace('.pth', '_temperature.txt')
    temperature = default_temp

    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r') as f:
                temperature = float(f.read().strip())
            print(f"[INFO] 模型 {os.path.basename(model_path)} 使用温度 {temperature:.4f}")
        except Exception as e:
            print(f"[WARN] 温度文件读取失败 {temp_file}: {e}，使用默认 {default_temp}")
    else:
        print(f"[WARN] 未找到温度文件 {temp_file}，使用默认 {default_temp}")

    # 包装成带温度的模型
    model_with_temp = ModelWithTemperature(base_model)
    model_with_temp.temperature = torch.nn.Parameter(torch.tensor([temperature], device=device))
    model_with_temp.eval()

    return model_with_temp


def test_unseen_data(model_source, test_data_dir, thresholds=[0.5],
                     device='cpu',
                     save_predictions=False):
    """在测试集上运行推理，支持多模型集成 + 温度缩放"""
    device = torch.device(device)
    print(f"Using device: {device}")

    # 获取模型路径列表
    if isinstance(model_source, list):
        model_paths = model_source
    elif os.path.isdir(model_source):
        model_paths = glob.glob(os.path.join(model_source, '*.pth'))
    else:
        raise ValueError("model_source 必须是目录路径或模型路径列表")

    print(f"[INFO] 加载 {len(model_paths)} 个模型：")
    models = []
    for model_path in model_paths:
        print(f"  -> {model_path}")
        try:
            model = load_model_with_temperature(model_path, device)
            models.append(model)
        except Exception as e:
            print(f"[ERROR] 加载模型失败 {model_path}: {e}")

    if not models:
        raise RuntimeError("未能成功加载任何模型！")

    # 加载测试数据
    test_dataset = CustomDataset_Meng(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    class_names = ['female', 'male']  # female: 0, male: 1
    results = []

    for threshold in thresholds:
        print(f"\n[INFO] 测试阈值 = {threshold}")

        # 初始化统计指标
        stats = {
            'correct': 0, 'total': 0,
            'female_correct': 0, 'female_total': 0,
            'male_correct': 0, 'male_total': 0,
            'pred_female_total': 0, 'pred_female_correct': 0,
            'pred_male_total': 0, 'pred_male_correct': 0
        }

        predictions = [] if save_predictions else None

        with torch.no_grad():
            for batch_idx, (data, labels, file_paths) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)

                all_female_probs = []
                model_predictions = []

                # 多模型预测（带温度缩放）
                for model in models:
                    scaled_logits = model(data)  # 模型里已经带温度缩放
                    probs = torch.softmax(scaled_logits, dim=1)
                    female_probs = probs[:, 0]
                    all_female_probs.append(female_probs)

                    # 二分类预测：0=female, 1=male
                    model_pred = (female_probs < threshold).long()
                    model_predictions.append(model_pred)

                # 堆叠所有模型的预测结果 [num_models, batch_size]
                model_predictions_tensor = torch.stack(model_predictions)

                # 集成策略：取最小值（有任一模型判为 female → female）
                predicted = torch.min(model_predictions_tensor, dim=0)[0]

                # 最大 female 概率（跨模型取 max）
                female_probs_tensor = torch.stack(all_female_probs)
                max_female_probs, _ = torch.max(female_probs_tensor, dim=0)

                # 输出每个样本预测
                for i in range(len(file_paths)):
                    egg_idx = batch_idx * test_loader.batch_size + i + 1
                    print(f"蛋 {egg_idx} (文件: {file_paths[i]}): "
                          f"最大female概率: {max_female_probs[i].item():.4f}, "
                          f"最终预测: {class_names[predicted[i].item()]}")

                # 更新统计
                batch_size = labels.size(0)
                stats['total'] += batch_size
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

                if save_predictions:
                    for i in range(batch_size):
                        pred_label = 'female' if predicted[i].item() == 0 else 'male'
                        true_label = 'female' if labels[i].item() == 0 else 'male'
                        predictions.append({
                            'file': file_paths[i],
                            'true_label': true_label,
                            'predicted_label': pred_label,
                            'probability': max_female_probs[i].item(),
                            'correct': predicted[i].item() == labels[i].item()
                        })

        # 计算指标
        accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        female_recall = 100 * stats['female_correct'] / stats['female_total'] if stats['female_total'] > 0 else 0
        male_recall = 100 * stats['male_correct'] / stats['male_total'] if stats['male_total'] > 0 else 0
        female_precision = 100 * stats['pred_female_correct'] / stats['pred_female_total'] if stats[
                                                                                                  'pred_female_total'] > 0 else 0
        male_precision = 100 * stats['pred_male_correct'] / stats['pred_male_total'] if stats[
                                                                                            'pred_male_total'] > 0 else 0

        results.append({
            'Threshold': threshold,
            'Test Accuracy (%)': f"{accuracy:.2f} ({stats['correct']}/{stats['total']})",
            'Female Recall (%)': f"{female_recall:.2f} ({stats['female_correct']}/{stats['female_total']})" if stats[
                                                                                                                   'female_total'] > 0 else "N/A",
            'Male Recall (%)': f"{male_recall:.2f} ({stats['male_correct']}/{stats['male_total']})",
            'Predicted Female Total': stats['pred_female_total'],
            'Female Precision (%)': f"{female_precision:.2f} ({stats['pred_female_correct']}/{stats['pred_female_total']})",
            'Predicted Male Total': stats['pred_male_total'],
            'Male Precision (%)': f"{male_precision:.2f} ({stats['pred_male_correct']}/{stats['pred_male_total']})"
        })

    # 打印表格
    headers = ['Threshold', 'Test Accuracy (%)', 'Female Recall (%)', 'Male Recall (%)',
               'Predicted Female Total', 'Female Precision (%)', 'Predicted Male Total', 'Male Precision (%)']
    print("\n结果表格:")
    print(tabulate([list(r.values()) for r in results], headers=headers, tablefmt="grid"))

    return results


if __name__ == "__main__":
    # 测试数据目录
    test_data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\huayu_test"

    # 模型目录或路径列表（支持多个模型 + _temperature.txt）
    model_source = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin"

    # 测试阈值列表
    thresholds = [0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.6,0.65]
    # thresholds = [0.4,0.3]

    print(f"正在测试 {test_data_dir}，模型来源: {model_source}")
    test_unseen_data(model_source, test_data_dir, thresholds=thresholds, save_predictions=True)
