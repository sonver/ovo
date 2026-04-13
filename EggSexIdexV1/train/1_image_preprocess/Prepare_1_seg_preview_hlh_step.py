import os
import cv2
import numpy as np
import concurrent.futures

from tools.tools import read_data, getOnePicture, trans, read_spe_files1

# 手动标定的中心点坐标，基于左下角为 (0, 0) 的坐标系 6*7 5*7
# IMAGES_CENTER = [
#     [(405, 423), (337, 423), (279, 423), (217, 423), (147, 423)],
#     [(404, 362), (340, 362), (278, 362), (216, 362), (152, 362)],
#     [(403, 302), (340, 302), (275, 302), (214, 302), (147, 302)],
#     [(401, 242), (340, 242), (275, 242), (213, 242), (149, 242)],
#     [(401, 178), (337, 178), (273, 178), (209, 178), (144, 178)],
#     [(401, 116), (337, 116), (273, 116), (212, 116), (148, 116)],
#     [(396, 51), (334, 51), (271, 51), (209, 51), (147, 51)]
# ]

# 全图的扫描的中心点 5*7
# IMAGES_CENTER = [
#     [(302, 425), (238, 425), (173, 425), (112, 432), (46, 431)],
#     [(302, 365), (239, 365), (176, 365), (111, 365), (47, 365)],
#     [(302, 300), (239, 300), (173, 300), (112, 300), (47, 300)],
#     [(302, 240), (239, 240), (173, 240), (111, 240), (47, 240)],
#     [(302, 178), (239, 178), (173, 178), (111, 178), (47, 178)],
#     [(302, 116), (239, 116), (173, 116), (111, 116), (47, 116)],
#     [(302, 53), (239, 53), (173, 53), (111, 53), (47, 53)]
# ]

# 三分之一 的扫描的中心点 5*7，蛋宽20px！
# IMAGES_CENTER = [
#     [(98, 425), (78, 425), (58, 425), (38, 425), (18, 425)],
#     [(98, 365), (78, 365), (58, 365), (38, 365), (18, 365)],
#     [(98, 300), (78, 300), (58, 300), (38, 300), (18, 300)],
#     [(98, 240), (78, 240), (58, 240), (38, 240), (18, 240)],
#     [(98, 178), (78, 178), (58, 178), (38, 178), (18, 178)],
#     [(98, 116), (78, 116), (58, 116), (38, 116), (18, 116)],
#     [(98, 54), (78, 54), (58, 54), (38, 54), (18, 54)]
# ]

# 二分之一 的扫描的中心点 5*7
IMAGES_CENTER = [
    [(150, 423), (119, 423), (88, 423), (57, 423), (26, 423)],
    [(150, 365), (119, 365), (88, 365), (57, 365), (26, 365)],
    [(151, 300), (119, 300), (88, 300), (57, 300), (24, 300)],
    [(150, 240), (119, 240), (86, 240), (57, 240), (24, 240)],
    [(150, 178), (118, 178), (88, 178), (55, 178), (24, 178)],
    [(149, 116), (117, 116), (86, 116), (55, 116), (24, 116)],
    [(148, 56), (117, 56), (86, 56), (55, 56), (24, 56)]
]

# 转换为左上角为 (0, 0) 的坐标系（cv2 和 numpy 默认）
adjusted_IMAGES_CENTER = [
    [(x, 479 - y) for x, y in row] for row in IMAGES_CENTER
]


def SegEgg(path):
    try:
        data_index = os.path.basename(path).split('.')[0]
        dataset = read_data(path)

        # 获取原始图像并应用 trans 处理（左右翻转 + 逆时针旋转 90 度）
        data = getOnePicture(dataset, 150)
        data = trans(data)
        data = cv2.convertScaleAbs(data)

        # 使用手动标定中心点，基于左上角为 (0, 0) 的坐标系
        centers = []
        for row in adjusted_IMAGES_CENTER:
            for center in row:
                x_core, y_core = center
                centers.append((x_core, y_core))

        # 按 y 从小到大（从上到下），x 从大到小（从右到左）排序，确保右上角为 1，左下角为 35
        centers_with_index = [(i + 1, x, y) for i, (x, y) in enumerate(centers)]
        centers_with_index.sort(key=lambda item: (item[2], -item[1]))  # 先按 y 升序，再按 x 降序
        centers = [(x, y) for _, x, y in centers_with_index]

        # 生成区域信息，基于中心点扩展 10 像素
        sorted_area_info = []
        for x_core, y_core in centers:
            x1, y1 = max(0, x_core - 9), max(0, y_core - 10)
            x2, y2 = min(179, x_core + 9), min(480, y_core + 10)
            sorted_area_info.append((x1, y1, x2, y2))

        # 标记中心点并导出预览图像

        os.makedirs(preview_dir, exist_ok=True)

        # 将灰度图像转换为 RGB 格式以绘制彩色标记
        preview_img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        # 绘制中心点（红色圆点）并标记蛋序号，从 1（右上角）到 35（左下角）
        for i, (x_core, y_core) in enumerate(centers, 1):
            x1, y1, x2, y2 = sorted_area_info[i - 1]
            cv2.circle(preview_img, (x_core, y_core), 5, (0, 0, 255), -1)  # 红色圆点
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(preview_img, f"{i}", (x_core + 5, y_core - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # 黄色文字

        # 保存预览图像

        preview_path = os.path.join(preview_dir, f"preview_{data_index}.png")
        cv2.imwrite(preview_path, preview_img)
        print(f"预览图像已保存至: {preview_path}")

        num_bands_range = range(1, dataset.RasterCount + 1)
        data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray() for d in num_bands_range])

        # 显式释放 dataset 资源
        dataset = None

        os.makedirs(write_path, exist_ok=True)

        # 使用手动区域进行裁剪和保存
        for i, (x1, y1, x2, y2) in enumerate(sorted_area_info, 1):
            # 直接使用 sorted_area_info 的裁切范围
            window = data_all_bands[:, x1:x2, y1:y2]

            # 检查 window 是否为空或形状异常
            if window.size == 0 or window.shape[1] == 0 or window.shape[2] == 0:
                print(f"警告: {data_index}-{i} 的裁切区域为空，跳过保存 (x: {x1}-{x2}, y: {y1}-{y2})")
                continue

            # 验证 window 形状
            expected_shape = (300, x2 - x1, y2 - y1)  # (bands, height, width)
            if window.shape != expected_shape:
                print(f"警告: {data_index}-{i} 的 window 形状异常: {window.shape}，预期: {expected_shape}，跳过保存")
                continue

            window_transposed = np.transpose(window, (1, 2, 0))
            sliced = window_transposed[:, :, 100:250]
            res = sliced.reshape(-1, sliced.shape[-1])

            # 检查 res 是否符合预期形状 (400, 150)
            if res.shape != (360, 150):
                print(f"警告: {data_index}-{i} 的数据形状异常: {res.shape}，预期: (360, 150)，跳过保存")
                continue

            egg_file = os.path.join(write_path, f'{data_index}-{i}.txt')
            with open(egg_file, 'w') as file:
                for item in res:
                    file.write(f"{' '.join(map(str, item))}\n")
            print(f'{data_index}-{i}.txt -- ok')

        # 释放大内存对象
        del data_all_bands, window, window_transposed, sliced, res, preview_img

    except Exception as e:
        print(f"处理文件 {path} 时发生错误: {e}")
    finally:
        if 'dataset' in locals():
            dataset = None


if __name__ == '__main__':
    preview_dir = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\gray\egg_seg_previews_hlh_half'
    data_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\half\raw_all_0909_13d6h-12h_8000_half"
    write_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\half\seg_all_0909_13d6h-12h_8000_half"
    spe_files_paths = read_spe_files1(data_path)

    # 根据系统资源调整线程数（建议 4-8 之间）
    MAX_WORKERS = min(4, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(SegEgg, path): path for path in spe_files_paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"文件 {path} 处理失败，错误信息: {e}")
