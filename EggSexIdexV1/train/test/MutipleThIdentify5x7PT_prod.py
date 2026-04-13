# 多模型预测 + 温度缩放，未划分female和male数据，用于5X7
import os
import time
import cv2
import numpy as np
import torch
import json
from osgeo import gdal
import pandas as pd
from tools.tools import read_data, getOnePicture, trans, gaussian_weights, snv_normalize

# =====================
# 模型加载（带温度缩放）
# =====================
def load_model_with_temperature(pt_path, device='cpu', default_temp=1.0):
    """加载 torchscript 模型和温度参数"""
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()

    # 对应的温度文件
    temp_file = pt_path.replace('.pt', '_temperature.txt')
    if os.path.exists(temp_file):
        try:
            with open(temp_file, "r") as f:
                temperature = float(f.read().strip())
            print(f"[INFO] 模型 {os.path.basename(pt_path)} 使用温度 {temperature:.4f}")
        except Exception as e:
            print(f"[WARN] 温度文件读取失败 {temp_file}: {e}，使用默认 {default_temp}")
            temperature = default_temp
    else:
        print(f"[WARN] 未找到温度文件 {temp_file}，使用默认 {default_temp}")
        temperature = default_temp

    return model, temperature


# =====================
# 5x7 固定中心点
# =====================
IMAGES_CENTER = [
    [(302, 425), (238, 425), (173, 425), (112, 425), (46, 425)],
    [(302, 365), (239, 365), (176, 365), (111, 365), (47, 365)],
    [(302, 300), (239, 300), (173, 300), (112, 300), (47, 300)],
    [(302, 240), (239, 240), (173, 240), (111, 240), (47, 240)],
    [(302, 178), (239, 178), (173, 178), (111, 178), (47, 178)],
    [(302, 116), (239, 116), (173, 116), (111, 116), (47, 116)],
    [(302, 53), (239, 53), (173, 53), (111, 53), (47, 53)]
]
adjusted_IMAGES_CENTER = [[(x, 479 - y) for x, y in row] for row in IMAGES_CENTER]


def JsonSegEgg(path, center_position=None):
    """分割 spe 图片中的 5x7 蛋位置，提取光谱数据"""
    data_index = os.path.basename(path).split('.')[0]
    dataset = read_data(path)
    data = getOnePicture(dataset, 150)
    data = trans(data)
    data = cv2.convertScaleAbs(data)

    centers = [(x, y) for row in adjusted_IMAGES_CENTER for (x, y) in row]
    centers_with_index = [(i + 1, x, y) for i, (x, y) in enumerate(centers)]
    centers_with_index.sort(key=lambda item: (item[2], -item[1]))
    centers = [(x, y) for _, x, y in centers_with_index]

    sorted_area_info = [(max(0, x - 10), max(0, y - 10),
                         min(359, x + 10), min(480, y + 10)) for x, y in centers]

    num_bands_range = range(1, dataset.RasterCount + 1)
    data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray()
                               for d in num_bands_range])

    egg_data_list = []
    for (x1, y1, x2, y2) in sorted_area_info:
        window = data_all_bands[:, x1:x2, y1:y2]
        window_transposed = np.transpose(window, (1, 2, 0))
        sliced = window_transposed[:, :, 100:250]
        egg_data_list.append(sliced.reshape(-1, sliced.shape[-1]).tolist())

    preview_dir = './egg_seg_previews_hlh'
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, f"preview_{data_index}.png")
    cv2.imwrite(preview_path, cv2.cvtColor(data, cv2.COLOR_GRAY2BGR))
    print(f"[INFO] 预览图像已保存至: {preview_path}")

    del data_all_bands, window, window_transposed, sliced
    return egg_data_list


# =====================
# 多模型预测 + 温度缩放
# =====================
def pred_test(models_with_temp, tensor, device, num_f, num_m, index, threshold):
    """对单个蛋进行预测"""
    with torch.no_grad():
        inputs = tensor.to(device)

        all_preds = []
        for model, T in models_with_temp:
            outputs = model(inputs) / T  # 温度缩放
            probs = torch.softmax(outputs, dim=1)
            female_prob = probs[:, 0]
            pred = 1 - (female_prob >= threshold).long()
            all_preds.append(pred)

        # 集成策略：有一个模型判定 female，就认为是 female
        preds_tensor = torch.stack(all_preds)
        predicted = torch.min(preds_tensor, dim=0)[0]

        num_f += (predicted == 0).sum().item()
        num_m += (predicted == 1).sum().item()

        eggindex_001 = (index // 5) + 1
        eggindex_002 = (index % 5) + 1
        eggindex_new = f'{eggindex_001}-{eggindex_002}'
        predict = {eggindex_new: int(predicted.item()) + 1}  # 1:female, 2:male

    return predict, num_f, num_m


def plate_cl(data_list, model_paths, batch_size, thresholds):
    """对整个托盘的蛋进行预测"""
    device = torch.device('cpu')
    models_with_temp = [load_model_with_temperature(p, device) for p in model_paths]

    results = []
    data_list = np.array(data_list)

    for threshold in thresholds:
        predict_dict, num_f, num_m = {}, 0, 0
        print(f"\n[INFO] 测试阈值 = {threshold}")

        for i in range(len(data_list)):
            data = np.array(data_list[i], dtype=float)

            result = []
            snv_data = np.apply_along_axis(snv_normalize, 1, data)
            for col in range(snv_data.shape[1]):
                column_data = snv_data[:, col]
                weights = gaussian_weights(column_data, sigma=1.0)
                weighted_sum = np.sum(column_data * weights) / np.sum(weights) if np.sum(weights) > 0 else 0.0
                result.append(weighted_sum)
            result = torch.tensor(result, dtype=torch.float32).reshape(batch_size, 1, 1, 150)

            predict, num_f, num_m = pred_test(models_with_temp, result, device, num_f, num_m, i, threshold)
            predict_dict.update(predict)

        results.append({
            'threshold': threshold,
            'predict': predict_dict,
            'num_f': num_f,
            'num_m': num_m
        })
        print(f"[INFO] Female: {num_f}, Male: {num_m}")

    return results


def writeJsonToCsharp(mem_key, predict, num_f, num_m):
    return json.dumps({
        "baseInfo": {"palletCode": mem_key},
        "statistics": {"female": num_f, "male": num_m},
        "rawData": predict
    })




def buildSpeVPath(spe_binary_data, hdr_str, mem_key):
    spe_path, hdr_path = "/vsimem/" + str(mem_key) + ".spe", "/vsimem/" + str(mem_key) + ".hdr"
    gdal.FileFromMemBuffer(spe_path, bytes(spe_binary_data))
    gdal.FileFromMemBuffer(hdr_path, bytes(hdr_str))
    return spe_path, hdr_path


def freeVPath(spe_path, hdr_path):
    gdal.Unlink(spe_path)
    gdal.Unlink(hdr_path)


def identifyEggs(spe_bin, hdr_str, model_dir, mem_key=11, thresholds=[0.5]):
    v_spe_path, v_hdr_path = buildSpeVPath(spe_bin, hdr_str, mem_key)

    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    print(f"[INFO] 处理中: {mem_key}")
    t_seg1 = time.perf_counter()
    test_list = JsonSegEgg(v_spe_path)
    t_seg2 = time.perf_counter()

    t_pred1 = time.perf_counter()
    results = plate_cl(test_list, model_paths, 1, thresholds)
    t_pred2 = time.perf_counter()

    print(f"[INFO] Prediction time: {t_pred2 - t_pred1:.4f}s")
    print(f"[INFO] Segmentation time: {t_seg2 - t_seg1:.4f}s")

    json_results = []
    for result in results:
        rj = writeJsonToCsharp(mem_key, result['predict'], result['num_f'], result['num_m'])
        json_results.append(rj)
        print(f"[INFO] 阈值 {result['threshold']} JSON: {rj}")

    freeVPath(v_spe_path, v_hdr_path)
    return json_results[0]


def test():
    # 模型目录，支持多个模型
    model_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin"


    spe_file_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\raw_all_huayu28\121A022025072800000101.spe"
    hdr_file_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\raw_all_huayu28\121A022025072800000101.hdr"
    with open(spe_file_path, 'rb') as f:
        spe_bin_data = f.read()
    with open(hdr_file_path, 'rb') as f:
        hdr_str_data = f.read()

    print("\n=== 测试多模型集成 ===")
    identifyEggs(spe_bin_data, hdr_str_data, model_dir, mem_key="121A022025072800000101",
                 thresholds=[0.35,0.4, 0.5])


if __name__ == '__main__':
    test()
