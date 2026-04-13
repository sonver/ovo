import math
import os
import cv2
import numpy as np
import concurrent.futures

from matplotlib import pyplot as plt

from tools.tools import read_data, getOnePicture, trans, find_best_grid_position, refine_grid_by_watershed, \
    read_spe_files1

def sort_egg_line(areas):
    """
    对蛋区域进行排序，镜像旋转后左下角为 1，右上角为 42。
    - 按中心点 y 坐标从大到小（从下到上）。
    - 分组后按中心点 x 坐标从小到大（从左到右）。
    :param areas: 蛋区域列表，格式为 [(x1, y1, x2, y2), ...]
    :return: 排序后的区域列表和中心点列表
    """
    # 计算每个区域的中心点
    centers = []
    for x1, y1, x2, y2 in areas:
        x_core = (x1 + x2) // 2
        y_core = (y1 + y2) // 2
        centers.append((x_core, y_core, x1, y1, x2, y2))

    # 按 y 坐标从大到小排序（从下到上）
    centers.sort(key=lambda item: item[1], reverse=True)

    # 每 6 个一组（对应 6 列）
    grouped_centers = [centers[i:i + 6] for i in range(0, len(centers), 6)]

    # 组内按 x 坐标从小到大排序（从左到右）
    for group in grouped_centers:
        group.sort(key=lambda item: item[0])

    # 展平分组列表
    sorted_centers = [item for group in grouped_centers for item in group]

    # 分离排序后的区域和中心点
    sorted_areas = [(x1, y1, x2, y2) for _, _, x1, y1, x2, y2 in sorted_centers]
    sorted_center_points = [(x_core, y_core) for x_core, y_core, _, _, _, _ in sorted_centers]

    return sorted_areas, sorted_center_points

def SegEgg(path):
    try:
        data_index = os.path.basename(path).split('.')[0]
        dataset = read_data(path)

        data = getOnePicture(dataset, 150)
        data = trans(data)
        data = cv2.convertScaleAbs(data)
        blurred_image = cv2.bilateralFilter(data, d=9, sigmaColor=75, sigmaSpace=75)
        img_edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

        rows, cols = 7, 6
        cell_width, cell_height = 66, 66

        best_grid, best_position, best_score = find_best_grid_position(img_edges, rows, cols, cell_width, cell_height)
        min_feature_threshold = 100
        refined_grid = refine_grid_by_watershed(best_grid, img_edges, min_feature_threshold)

        new_area = []
        for contour in refined_grid:
            x1, y1 = contour[0]
            x2, y2 = contour[2]
            x1 = max(0, x1)
            new_area.append((x1, y1, x2, y2))

        # 使用 sort_egg_line 排序，确保左下角为 1，右上角为 42
        sorted_area_info, centers = sort_egg_line(new_area)

        # 标记中心点并导出预览图像

        os.makedirs(preview_dir, exist_ok=True)

        # 将灰度图像转换为 RGB 格式以绘制彩色标记
        preview_img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        # 绘制中心点（红色圆点）并标记蛋序号
        for i, (x_core, y_core) in enumerate(centers):
            egg_index = i + 1  # 蛋序号从 1 开始
            cv2.circle(preview_img, (x_core, y_core), 5, (0, 0, 255), -1)  # 红色圆点
            # 添加蛋序号标签
            cv2.putText(preview_img, f"{egg_index}", (x_core + 10, y_core - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # 黄色文字

        # 保存预览图像，文件名与源文件序号一致
        preview_path = os.path.join(preview_dir, f"preview_{data_index}.png")
        cv2.imwrite(preview_path, preview_img)
        print(f"预览图像已保存至: {preview_path}")

        num_bands_range = range(1, dataset.RasterCount + 1)
        data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray() for d in num_bands_range])

        # 显式释放 dataset 资源
        dataset = None


        os.makedirs(write_path, exist_ok=True)

        # 使用排序后的区域进行裁剪和保存
        for i, grid in enumerate(sorted_area_info):
            x1, y1, x2, y2 = grid
            x_core = (x1 + x2) // 2
            y_core = (y1 + y2) // 2
            x_start, x_end = x_core - 10, x_core + 10
            y_start, y_end = y_core - 10, y_core + 10

            window = data_all_bands[:, x_start:x_end, y_start:y_end]
            window_transposed = np.transpose(window, (1, 2, 0))
            sliced = window_transposed[:, :, 100:250]
            res = sliced.reshape(-1, sliced.shape[-1])

            egg_index = i + 1  # 蛋序号从 1 开始，左下角为 1，右上角为 42
            egg_file = os.path.join(write_path, f'egg{data_index}-{egg_index}.txt')
            with open(egg_file, 'w') as file:
                for item in res:
                    file.write(f"{' '.join(map(str, item))}\n")
            print(f'egg{data_index}-{egg_index}.txt -- ok')

        # 释放大内存对象
        del data_all_bands, window, window_transposed, sliced, res, preview_img

    except Exception as e:
        print(f"处理文件 {path} 时发生错误: {e}")
    finally:
        if 'dataset' in locals():
            dataset = None

if __name__ == '__main__':
    preview_dir = './egg_seg_previews_1month'
    data_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_jinghong_12\D13.5"
    write_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_jinghong_12\seg_d13.5'
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