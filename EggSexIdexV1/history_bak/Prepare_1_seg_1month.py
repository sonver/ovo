import math
import os
import cv2
import numpy as np
import concurrent.futures
from tools.tools import read_data, getOnePicture, trans, find_best_grid_position, refine_grid_by_watershed, \
    sort_egg_line, read_spe_files1, min_max_normalize_and_scale_spectra


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
        sorted_area_info = sort_egg_line(new_area)

        num_bands_range = range(1, dataset.RasterCount + 1)
        data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray() for d in num_bands_range])
        data_all_bands = trans(data_all_bands)  # 形状从 (bands, H, W) 变为 (bands, W, H)
        # 显式释放dataset资源
        dataset = None

        write_path = '../data/egg_seg_data_1month'
        os.makedirs(write_path, exist_ok=True)

        for i, grid in enumerate(sorted_area_info):
            x1, y1, x2, y2 = grid
            x_core = (x1 + x2) // 2
            y_core = (y1 + y2) // 2
            x_start, x_end = x_core - 10, x_core + 10
            y_start, y_end = y_core - 10, y_core + 10

            # 提取窗口数据并调整形状为 (20, 20, num_bands)
            window = data_all_bands[:, x_start:x_end, y_start:y_end]
            window_transposed = np.transpose(window, (1, 2, 0))

            # num_pixels = window_transposed.shape[0] * window_transposed.shape[1]  # 20×20=400
            # num_bands = window_transposed.shape[2]
            # window_flat = window_transposed.reshape(num_pixels, num_bands)  # 形状 (400, num_bands)
            # window_normalized_flat = min_max_normalize_and_scale_spectra(window_flat)  # 归一化
            # # 恢复形状为 (20, 20, num_bands)
            # window_normalized = window_normalized_flat.reshape(20, 20, num_bands)

            # 切片所需波段范围并扁平化
            # sliced = window_transposed[:, :, 150:300]  # 直接切片
            sliced = window_transposed[:, :, 100:250]  # 直接切片 a
            res = sliced.reshape(-1, sliced.shape[-1])

            egg_file = os.path.join(write_path, f'egg{data_index}-{i + 1}.txt')
            with open(egg_file, 'w') as file:
                for item in res:
                    file.write(f"{' '.join(map(str, item))}\n")
            print(f'egg{data_index}-{i + 1}.txt -- ok')

        # 释放大内存对象
        del data_all_bands, window, window_transposed, sliced, res

    except Exception as e:
        print(f"处理文件 {path} 时发生错误: {e}")
    finally:
        if 'dataset' in locals():
            dataset = None


if __name__ == '__main__':
    data_path = "../data/egg_raw_data_1month"  # 原始.spe和.hdr文件目录
    spe_files_paths = read_spe_files1(data_path)

    # 根据系统资源调整线程数（建议4-8之间）
    MAX_WORKERS = min(4, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(SegEgg, path): path for path in spe_files_paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"文件 {path} 处理失败，错误信息: {e}")