import math
import os

import cv2
import numpy as np

from tools.tools import read_data, getOnePicture, trans, find_best_grid_position, refine_grid_by_watershed, \
    sort_egg_line, read_spe_files1, min_max_normalize_and_scale_spectra


def SegEgg(path):
    data_index = os.path.basename(path).split('.')[0]
    # 读取图像并进行预处理
    dataset = read_data(path)

    data = getOnePicture(dataset, 150)  # 找最清晰的150个波段，前150个索引对应波长是可见光，因反射率等因素，可见光的清晰度较好
    data = trans(data)
    data = cv2.convertScaleAbs(data)
    blurred_image = cv2.bilateralFilter(data, d=9, sigmaColor=75, sigmaSpace=75)

    # Canny边缘检测
    img_edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # 网格模型参数
    rows, cols = 7, 6
    # cell_width, cell_height = 64,64  # 2024111456
    cell_width, cell_height = 66, 66  # 2024121567

    # 查找最佳网格并细化
    best_grid, best_position, best_score = find_best_grid_position(img_edges, rows, cols, cell_width, cell_height)
    min_feature_threshold = 100  # 调整此值作为检测阈值
    refined_grid = refine_grid_by_watershed(best_grid, img_edges, min_feature_threshold)

    # extracted_data = extract_20x20_from_refined_grid(dataset, refined_grid,dataset.RasterCount)
    # print(extracted_data)

    # 绘制网格
    # output_image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

    # 奇
    new_area = []
    # 存红框，排序
    for contour in refined_grid:
        x1, y1 = contour[0]
        x2, y2 = contour[2]
        x1 = max(0, x1)
        new_area.append((x1, y1, x2, y2))
    sorted_area_info = sort_egg_line(new_area)

    num_bands_range = range(1, dataset.RasterCount + 1)
    data_all_bands = np.array([dataset.GetRasterBand(d).ReadAsArray() for d in num_bands_range])

    # 优化后的算法
    for i, grid in enumerate(sorted_area_info):
        x1, y1, x2, y2 = grid
        x_core = (x1 + x2) // 2
        y_core = (y1 + y2) // 2
        x_start, x_end = x_core - 10, x_core + 10
        y_start, y_end = y_core - 10, y_core + 10

        # 提取窗口数据并调整形状为 (20, 20, num_bands)
        window = data_all_bands[:, x_start:x_end, y_start:y_end]
        window_transposed = np.transpose(window, (1, 2, 0))

        # Min - Max归一化切割
        num_pixels = window_transposed.shape[0] * window_transposed.shape[1]  # 20×20=400
        num_bands = window_transposed.shape[2]
        window_flat = window_transposed.reshape(num_pixels, num_bands)  # 形状 (400, num_bands)
        window_normalized_flat = min_max_normalize_and_scale_spectra(window_flat)  # 归一化
        # 恢复形状为 (20, 20, num_bands)
        window_normalized = window_normalized_flat.reshape(20, 20, num_bands)

        # 切片所需波段范围并扁平化
        # sliced = window_transposed[:, :, 150:300]  # 直接切片
        sliced = window_normalized[:, :, 100:250]  # 直接切片 a
        res = sliced.reshape(-1, sliced.shape[-1])

        # continue
        write_path = '../data/egg_seg_data_14month'
        # write_path = './data/egg_seg_data_a'
        write_path = os.path.join(write_path, f'egg{data_index}-{i + 1}.txt')
        with open(write_path, 'w') as file:
            # 遍历列表中的每个元素
            for item in res:
                # if len(item) != 300:
                #     print(ssplit_spe_file[2] + "bug")
                item = ' '.join(map(str, item))
                # 将元素写入文件，每个元素占一行
                file.write(f"{item}\n")
            # print(f'egg{i+1}.txt -- ok')
            print(f'egg{data_index}-{i + 1}.txt -- ok')

    # # 绘制绿框
    # # 原始网格
    # if best_grid is not None:
    #     for contour in best_grid:
    #         contour = contour.astype(int)
    #         cv2.polylines(output_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
    # # 细化后的网格
    # for contour in refined_grid:
    #     contour = contour.astype(int)
    #     cv2.polylines(output_image, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(data_index)
    # plt.imshow(output_image)
    # plt.show()


if __name__ == '__main__':
    # 1分割
    data_path = "../data/egg_raw_data_14month"  # 原始.spe和.hdr文件目录
    spe_files_paths = read_spe_files1(data_path)  # 单层目录读取文件目录
    # spe_files_paths = map_spe_files(data_path) # 递归便遍历目录
    # 根据.spe文件，进行单枚蛋切割
    for spe_file_path in spe_files_paths:
        SegEgg(spe_file_path)
