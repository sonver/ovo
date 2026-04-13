import cv2
import os
import math
from matplotlib import pyplot as plt
import numpy as np
from open3d.cpu.pybind.core import float32
from osgeo import gdal
from data_MQ import read_spe_files1, read_spe_files2, sort_egg_line, sort_egg_row, getSpectralAnalysis
from data_MQ import getSpectralAnalysisData, drawSpectralAnalysisChart, drawSpectralAnalysisChart2


# ========= 新增：SNV 和 高斯权重函数 =========
def snv_normalize(row: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate:
    (x - mean) / std, 如果 std 为 0，则返回全 0
    """
    row = row.astype(np.float32)
    mean = np.mean(row)
    std = np.std(row)
    if std <= 0:
        return np.zeros_like(row, dtype=np.float32)
    return (row - mean) / std


def gaussian_weights(column_data: np.ndarray, mu=None, sigma: float = 1.0) -> np.ndarray:
    """
    对一列数据生成高斯权重
    - column_data: [N_pixel]
    - mu: 均值，默认用列均值
    - sigma: 标准差
    返回: 权重数组 [N_pixel]
    """
    column_data = column_data.astype(np.float32)
    if mu is None:
        mu = float(np.mean(column_data))
    if sigma <= 0:
        sigma = 1.0
    return np.exp(-((column_data - mu) ** 2) / (2.0 * sigma ** 2))


# 读取数据和处理函数
def read_data(spe_file):
    dataset = gdal.Open(spe_file)
    if dataset is None:
        print("Failed to open the SPE file.")
        return
    else:
        return dataset


# 读取第i个波段的数据
def getOnePicture(dataset, i):
    return dataset.GetRasterBand(i).ReadAsArray()


def trans(data):
    data = np.fliplr(data)  # 矩阵左右翻转
    data = np.rot90(data, k=1)  # 矩阵逆时针旋转
    return data


# 创建固定网格
def create_fixed_grid(rows, cols, cell_width, cell_height, start_x, start_y):
    contours = []
    for row in range(rows):
        offset = cell_width // 2 if row % 2 == 1 else 0  # 每隔一行横向偏移
        for col in range(cols):
            x1 = start_x + col * cell_width + offset
            y1 = start_y + row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            contours.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    return contours  # 返回四个顶点坐标


# 横隔板位置清除
def up_and_down_intervals(img):
    segmented_img1 = img
    mask = np.zeros(segmented_img1.shape, dtype=bool)
    # 行
    mask[0: 63, :] = True
    mask[116: 122, :] = True
    mask[174: 180, :] = True
    mask[235: 237, :] = True
    mask[293: 295, :] = True
    mask[353: 357, :] = True
    mask[415: 421, :] = True
    mask[473:, :] = True
    segmented_img1[mask] = 0
    return segmented_img1


def refine_with_high_precision_algorithm(roi, x1, y1, x2, y2):
    # 判断roi图像的通道数
    if len(roi.shape) == 2:
        gray_roi = roi
    else:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    edges = cv2.Canny(gray_roi, 50, 150)

    # 查找边缘上的角点
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        # 将获取到的角点坐标格式从 [[[x,y]]] 转换为 [[x,y]]，以符合cv2.cornerSubPix的要求
        edge_points = np.squeeze(corners).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        cv2.cornerSubPix(gray_roi, edge_points, win_size, zero_zone, criteria)
    else:
        # 如果没有检测到角点，返回原矩形的轮廓
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # 基于边缘点进行轮廓拟合
    if edge_points.size > 0:
        epsilon = 0.01 * cv2.arcLength(np.array(edge_points), True)
        approx_contour = cv2.approxPolyDP(np.array(edge_points), epsilon, True)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(approx_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 获取矩形的角度
        angle = rect[2]
        if angle < -45:
            angle += 90

        # 计算旋转矩阵
        center = (np.sum(box[:, 0]) / 4, np.sum(box[:, 1]) / 4)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋转矩形顶点坐标
        box = np.array([[x, y, 1] for x, y in box])
        new_box = np.dot(box, M.T)[:, :2]

        # 调整坐标使其对应到原图像中的位置
        refined_contour = []
        for point in new_box:
            px, py = point
            refined_contour.append([px + x1, py + y1])

        # 将 refined_contour 转换为 numpy 数组
        refined_contour = np.array(refined_contour, dtype=np.float32)

        # 获取细化后的最小外接矩形
        refined_rect = cv2.minAreaRect(refined_contour)
        refined_box = cv2.boxPoints(refined_rect)
        refined_box = np.int0(refined_box)

        # 确保细化后的矩形不超出原始矩形
        x_min = np.min(refined_box[:, 0])
        x_max = np.max(refined_box[:, 0])
        y_min = np.min(refined_box[:, 1])
        y_max = np.max(refined_box[:, 1])

        x_min = max(x_min, x1)
        x_max = min(x_max, x2)
        y_min = max(y_min, y1)
        y_max = min(y_max, y2)

        refined_contour = np.array([[x_min, y_min],
                                    [x_max, y_min],
                                    [x_max, y_max],
                                    [x_min, y_max]])
        return refined_contour
    else:
        # 如果没有检测到有效的边缘点，返回原矩形的轮廓
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def refine_grid_by_watershed(original_grid, edge_image, min_feature_threshold):
    refined_grid = []
    for contour in original_grid:
        x1, y1 = contour[0]  # 左上角坐标
        x2, y2 = contour[2]  # 右下角坐标

        roi = edge_image[int(y1):int(y2), int(x1):int(x2)]

        feature_count = np.sum(roi) / 255
        if feature_count < min_feature_threshold:
            refined_grid.append(contour)
            continue

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(roi, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(roi_color, markers)
        roi_color[markers == -1] = [255, 0, 0]

        contours, _ = cv2.findContours(np.uint8(markers == 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            width_diff = abs((x2 - x1) - w)
            height_diff = abs((y2 - y1) - h)
            if width_diff < 5 or height_diff < 5:
                refined_contour = refine_with_high_precision_algorithm(roi, x1, y1, x2, y2)
            else:
                x1_refined = max(x1 + x, x1)
                x2_refined = min(x1 + x + w, x2)
                y1_refined = max(y1 + y, y1)
                y2_refined = min(y1 + y + h, y2)

                refined_contour = np.array([[x1_refined, y1_refined],
                                            [x2_refined, y1_refined],
                                            [x2_refined, y2_refined],
                                            [x1_refined, y2_refined]])
            refined_grid.append(refined_contour)
        else:
            refined_grid.append(contour)

    return refined_grid


# 查找最佳网格匹配位置
def find_best_grid_position(edge_image, rows, cols, cell_width, cell_height):
    best_score = -float('inf')
    best_grid = None
    best_position = (0, 0)

    for start_x in range(0, edge_image.shape[1] - cols * cell_width, 10):
        for start_y in range(0, edge_image.shape[0] - rows * cell_height, 10):
            grid_contours = create_fixed_grid(rows, cols, cell_width, cell_height, start_x, start_y)
            score = match_grid_to_edges(grid_contours, edge_image)
            if score > best_score:
                best_score = score
                best_grid = grid_contours
                best_position = (start_x, start_y)

    return best_grid, best_position, best_score


# 匹配评分函数
def match_grid_to_edges(grid_contours, edge_image):
    score = 0
    for contour in grid_contours:
        mask = np.zeros(edge_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(int)], -1, 255, thickness=cv2.FILLED)
        match_score = np.sum(cv2.bitwise_and(edge_image, mask) / 255)
        score += match_score
    return score


def create_pallet():
    """
    生成一个5行×7列的托盘中心点坐标矩阵。
    每个元素格式为 (id, x, y)
    """
    # x_coords = [298, 235, 176, 112, 51]  # 从右到左
    x_coords = [280, 217, 158, 94, 33]  # 从右到左
    y_coords = [57, 119, 178, 242, 302, 362, 425]  # 从下到上
    pallet = []
    current_id = 1

    for y in y_coords:
        column = []
        for x in x_coords:
            column.append((current_id, x, y))
            current_id += 1
        pallet.append(column)

    return np.array(pallet)


def main(path):
    data_index = os.path.basename(path).split('.')[0]
    dataset = read_data(path)
    data = getOnePicture(dataset, 100)  # 找最清晰的波段
    data = trans(data)

    pallet = np.array(create_pallet()).reshape(-1, 3)  # 展平为35x3数组

    for i, x, y in pallet:
        x1 = x - 10
        x2 = x + 10
        y1 = y - 10
        y2 = y + 10

        x_core = x
        y_core = y

        # 收集当前蛋 20x20 小窗内所有像素的光谱（100:250 波段）
        res = []
        data_all = []  # 存所有波段的原始高光谱数据

        num_bands = dataset.RasterCount
        for d in range(1, num_bands + 1):
            band = dataset.GetRasterBand(d)
            data_all.append(band.ReadAsArray())
        # data_all: list[num_bands] 每个 [H, W]

        for j in range(x_core - 10, x_core + 10):
            for k in range(y_core - 10, y_core + 10):
                x_values, y_values = getSpectralAnalysis(data_all, j, k)  # y_values 是整条光谱
                # 只取原始光谱的 100:250 段（150 维），不再做 min-max 和 *1000
                seg = np.array(y_values[100:250], dtype=np.float32)  # [150]
                res.append(seg)

        # 将 res 转为 [N_pixel, 150]，然后做 SNV + 高斯加权，聚合成 150 维
        res_arr = np.array(res, dtype=np.float32)  # [N_pixel, 150]

        # 1) 行向量 SNV
        snv_data = np.apply_along_axis(snv_normalize, 1, res_arr)  # [N_pixel, 150]

        # 2) 列方向高斯加权
        num_pixels, num_bands_used = snv_data.shape
        agg = []
        sigma = 1.0  # 可以按需调整
        for col in range(num_bands_used):
            column_data = snv_data[:, col]         # [N_pixel]
            weights = gaussian_weights(column_data, sigma=sigma)
            w_sum = np.sum(weights)
            if w_sum == 0:
                weighted_sum = 0.0
            else:
                weighted_sum = float(np.sum(column_data * weights) / w_sum)
            agg.append(weighted_sum)

        egg_spec = np.array(agg, dtype=np.float32)  # [150]

        # 写出：每个egg一个txt，只一行150维
        write_path = r'E:\Dataset\2025-11-大午褐\1122-4000'
        os.makedirs(write_path, exist_ok=True)
        write_path = os.path.join(write_path, f'egg{data_index}-{i}.txt')
        with open(write_path, 'w', encoding='utf-8') as file:
            line = ' '.join(map(str, egg_spec.tolist()))
            file.write(line + '\n')
        print(f'egg{data_index}-{i}.txt -- ok')


if __name__ == '__main__':
    data_path = r"E:\Dataset\2025-11\1122\dawuhe"
    spe_files = read_spe_files1(data_path)
    for spe_file in spe_files:
        main(spe_file)
