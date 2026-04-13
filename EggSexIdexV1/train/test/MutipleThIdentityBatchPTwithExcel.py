# 批量 spe 文件预测，支持多个模型 + 温度缩放
import os
import time
import cv2
import numpy as np
import torch
import json
from osgeo import gdal
import pandas as pd
import glob
from tools.tools import read_data, getOnePicture, trans, gaussian_weights, snv_normalize


# -------------------------------
# 工具函数
# -------------------------------

def load_model_with_temperature_pt(pt_path, device='cpu', default_temp=1.0):
    """加载 TorchScript 模型，并附带温度参数"""
    model = torch.jit.load(pt_path, map_location=device)
    model.eval()

    temp_file = pt_path.replace('.pt', '_temperature.txt')
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r') as f:
                T = float(f.read().strip())
            print(f"Loaded temperature {T:.4f} for model {os.path.basename(pt_path)}")
        except Exception as e:
            print(f"Error loading temperature for {pt_path}: {e}, using default {default_temp}")
            T = default_temp
    else:
        print(f"Temperature file not found for {pt_path}, using default {default_temp}")
        T = default_temp

    return model, T


# 手动标定的中心点坐标 (5x7)
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
    data_index = os.path.basename(path).split('.')[0]
    dataset = read_data(path)
    data = getOnePicture(dataset, 150)
    data = trans(data)
    data = cv2.convertScaleAbs(data)

    centers = [(x, y) for row in adjusted_IMAGES_CENTER for (x, y) in row]
    centers_with_index = [(i + 1, x, y) for i, (x, y) in enumerate(centers)]
    centers_with_index.sort(key=lambda item: (item[2], -item[1]))
    centers = [(x, y) for _, x, y in centers_with_index]

    sorted_area_info = [(max(0, x - 10), max(0, y - 10), min(359, x + 10), min(480, y + 10)) for x, y in centers]

    num_bands_range = range(1, dataset.RasterCount + 1)
    data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray() for d in num_bands_range])

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
    print(f"预览图像已保存至: {preview_path}")

    del data_all_bands, window, window_transposed, sliced
    return egg_data_list


def pred_test(models_with_temp, tensor, device, num_f, num_m, index, threshold):
    with torch.no_grad():
        inputs = tensor.to(device)

        predictions = []
        for model, T in models_with_temp:
            outputs = model(inputs)
            outputs = outputs / T  # 温度缩放
            probs = torch.softmax(outputs, dim=1)
            female_probs = probs[:, 0]
            predicted = 1 - (female_probs >= threshold).long()
            predictions.append(predicted)

        predictions_tensor = torch.stack(predictions)
        predicted = torch.min(predictions_tensor, dim=0)[0]  # 有任意一个判female则为female

        num_f += (predicted == 0).sum().item()
        num_m += (predicted == 1).sum().item()

        eggindex_001 = (index // 5) + 1
        eggindex_002 = (index % 5) + 1
        eggindex_new = f'{eggindex_001}-{eggindex_002}'
        predict = {eggindex_new: int(predicted.item()) + 1}

    return predict, num_f, num_m


def plate_cl(data_list, model_paths, batch_size, thresholds):
    device = torch.device('cpu')
    models_with_temp = [load_model_with_temperature_pt(p, device) for p in model_paths]

    results = []
    data_list = np.array(data_list)

    for threshold in thresholds:
        predict_dict, num_f, num_m = {}, 0, 0
        print(f"\nTesting with threshold = {threshold}")

        for i in range(len(data_list)):
            data = np.array(data_list[i], dtype=float)

            result = []
            snv_data = np.apply_along_axis(snv_normalize, 1, data)
            for col in range(snv_data.shape[1]):
                column_data = snv_data[:, col]
                weights = gaussian_weights(column_data, sigma=1.0)
                weighted_sum = np.sum(column_data * weights) / np.sum(weights) if np.sum(weights) != 0 else 0.0
                result.append(weighted_sum)
            result = torch.tensor(result, dtype=torch.float32).reshape(batch_size, 1, 1, 150)

            predict, num_f, num_m = pred_test(models_with_temp, result, device, num_f, num_m, i, threshold)
            predict_dict.update(predict)

        results.append({'threshold': threshold, 'predict': predict_dict, 'num_f': num_f, 'num_m': num_m})
        print(f"Female: {num_f}, Male: {num_m}")
    return results


def writeJsonToCsharp(mem_key, predict, num_f, num_m):
    return json.dumps({
        "baseInfo": {"palletCode": mem_key},
        "statistics": {"female": num_f, "male": num_m},
        "rawData": predict
    })


def load_platecode_to_id_mapping(excel_path, sheet_name="Result 1"):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return dict(zip(df['PlateCode'], df['Id']))
    except Exception as e:
        print(f"加载 Excel 文件失败: {e}")
        return {}


def generate_excel(results, platecode_to_id, output_dir='./egg_seg_previews_hlh'):
    os.makedirs(output_dir, exist_ok=True)
    columns = ['Id', '阈值'] + [f"{i}-{j}" for i in range(1, 8) for j in range(1, 6)]
    data = []
    for mem_key, result_list in results:
        tray_id = platecode_to_id.get(mem_key, 'Unknown')
        for result in result_list:
            row = [tray_id, result['threshold']] + [result['predict'].get(f"{i}-{j}", 0) for i in range(1, 8) for j in range(1, 6)]
            data.append(row)
    data.append(['序号'] + [''] + [str(i) for i in range(1, 36)])
    df = pd.DataFrame(data, columns=columns)
    excel_path = os.path.join(output_dir, 'predict_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Excel 文件已保存至: {excel_path}")


def buildSpeVPath(spe_binary_data, hdr_str, mem_key):
    spe_path, hdr_path = "/vsimem/" + str(mem_key) + ".spe", "/vsimem/" + str(mem_key) + ".hdr"
    gdal.FileFromMemBuffer(spe_path, bytes(spe_binary_data))
    gdal.FileFromMemBuffer(hdr_path, bytes(hdr_str))
    return spe_path, hdr_path


def freeVPath(spe_path, hdr_path):
    gdal.Unlink(spe_path)
    gdal.Unlink(hdr_path)


def identifyEggs(spe_bin, hdr_str, model_paths, mem_key=11, thresholds=[0.5]):
    v_spe_path, v_hdr_path = buildSpeVPath(spe_bin, hdr_str, mem_key)

    print(f"处理中: {mem_key}")
    test_list = JsonSegEgg(v_spe_path)
    results = plate_cl(test_list, model_paths, 1, thresholds)

    json_results = []
    for result in results:
        rj = writeJsonToCsharp(mem_key, result['predict'], result['num_f'], result['num_m'])
        json_results.append(rj)
        print(f"\nThreshold {result['threshold']} JSON:\n{rj}")

    freeVPath(v_spe_path, v_hdr_path)
    print(f"处理完成: {mem_key}")
    return json_results, results


def test():
    model_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\multiple_model_origin"
    model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]

    data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\raw_all_huayu28"
    excel_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\train\excel_preprocess\ForecastRecords_0726-28.xlsx"
    thresholds = [0.1, 0.15, 0.4, 0.5]

    platecode_to_id = load_platecode_to_id_mapping(excel_path)
    if not platecode_to_id:
        print("无法加载 PlateCode 到 Id 的映射，退出程序")
        return

    spe_files = glob.glob(os.path.join(data_dir, "*.spe"))
    all_results = []
    for spe_file in spe_files:
        mem_key = os.path.basename(spe_file).split('.')[0]
        hdr_file = os.path.splitext(spe_file)[0] + '.hdr'
        if not os.path.exists(hdr_file):
            print(f"警告: 未找到对应的 .hdr 文件 {hdr_file}，跳过")
            continue

        with open(spe_file, 'rb') as f:
            spe_bin_data = f.read()
        with open(hdr_file, 'rb') as f:
            hdr_str_data = f.read()

        print(f"\n=== 处理文件 {mem_key} ===")
        json_results, results = identifyEggs(spe_bin_data, hdr_str_data, model_paths, mem_key=mem_key, thresholds=thresholds)
        all_results.append((mem_key, results))

    generate_excel(all_results, platecode_to_id)


if __name__ == '__main__':
    test()
