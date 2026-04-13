"""
类库工具集
"""
import os

import cv2
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


def gaussian_weights(data_column, mu=None, sigma=1.0):
    """
    计算高斯权重
    :param data_column: 列数据，形状为 (400,)
    :param mu: 高斯分布的均值，默认为列均值
    :param sigma: 标准差，默认 1.
    :return: 权重数组，形状为 (400,)
    """
    if mu is None:
        mu = np.mean(data_column)
    weights = np.exp(-((data_column - mu) ** 2) / (2 * sigma ** 2))
    return weights


# 排序1（按照y坐标，从小到大6个一组，总共7组，对每组内按照x坐标从小到大排序）--针对鸡蛋横着排列
def sort_egg_line(list):
    list.sort(key=lambda item: item[1], reverse=True)  # 按照y坐标从大到小排序，sort默认升序
    grouped_contours = [list[i:i + 6] for i in range(0, len(list), 6)]  # 6个一组
    for group in grouped_contours:  # 组内按照x排序
        group.sort(key=lambda item: item[0])
    sorted_contours_info = [item for group in grouped_contours for item in group]
    return sorted_contours_info


# 排序2（按照x坐标，从小到大6个一组，总共7组，对每组内按照y坐标从小到大排序）--针对鸡蛋竖着排列
def sort_egg_row(list):
    list.sort(key=lambda item: item[0])  # 按照y坐标排序
    grouped_contours = [list[i:i + 6] for i in range(0, len(list), 6)]  # 6个一组
    for group in grouped_contours:  # 组内按照x排序
        group.sort(key=lambda item: item[1])
    sorted_contours_info = [item for group in grouped_contours for item in group]
    return sorted_contours_info


def trans(data):
    """仅对最后两个维度（空间维度）进行变换，保持波段维度不变"""
    if data.ndim == 2:
        # 单波段图像处理：先左右翻转，再逆时针旋转90度
        data = np.fliplr(data)
        data = np.rot90(data, k=1)
    elif data.ndim == 3:
        # 多波段数据：逐波段处理空间维度
        data = np.transpose(data, (1, 2, 0))  # 转换为 (H, W, bands) 方便操作
        data = np.fliplr(data)  # 左右翻转
        data = np.rot90(data, k=1)  # 逆时针旋转90度
        data = np.transpose(data, (2, 0, 1))  # 恢复为 (bands, W, H)
    return data


# 如果需要，spe和png文件顺序一致，对二维数组进行左右翻转并逆时针90°
# def trans(data):
#     data = np.fliplr(data)  # 矩阵左右翻转
#     data = np.rot90(data, k=1)  # 矩阵逆时针旋转
#     return data


# 得到最清晰的波段对应的图
def getOnePicture(dataset, i):
    # 获取 1 到 i 索引的波段，GetRasterBand索引从1开始
    return dataset.GetRasterBand(i).ReadAsArray()


gdal.UseExceptions()


# 读取数据和处理函数
def read_data(spe_file):
    try:
        dataset = gdal.Open(spe_file)
        return dataset
    except RuntimeError as e:
        print(f"GDAL 错误: {e}")


# 看txt一行里有多少数据
def count_num(data_path="D:\\Code\\Dataset\\egg\\GT\\testdatav5.1\\female\\1.txt"):
    # data_path = "D:\\Code\\Dataset\\egg\GT\\testdatav5.1\\female\\1.txt"
    fileHandler = open(data_path, "r")
    listOfLines = fileHandler.readlines()
    words = []
    words = listOfLines[0].split(" ")
    fileHandler.close()
    print(words)
    print(len(words))


def delete_single():
    excel_path = 'D:/Code/Dataset/egg/其余全部/手扒未孵化数据.xlsx'
    other_df = pd.read_excel(excel_path, header=None)
    location = []
    for i in range(len(other_df) - 2):
        a, b = other_df.iloc[i][0].split('-')
        name = f'egg{a}-{b}.txt'
        location.append(name)
    # print(location)
    folder_path1 = 'D:/Code/Dataset/egg/GT/testdata_other/female'
    folder_path2 = 'D:/Code/Dataset/egg/GT/testdata_other/male'
    # spe_files = map_spe_files(folder_path)
    for file_name in location:
        file_path1 = os.path.join(folder_path1, file_name)
        file_path2 = os.path.join(folder_path2, file_name)
        if os.path.exists(file_path1):
            os.remove(file_path1)  # 删除文件
            print(f"{file_path1} 已被删除")
        elif os.path.exists(file_path2):
            os.remove(file_path2)
            print(f"{file_path2} 已被删除")
        else:
            print(f"{file_name} 不存在")

    # pd.set_option('display.max_rows', None)
    # print(other_df)


# 某文件夹下面的所有spe文件
def read_spe_files1(root_folder):
    spe_files = []
    for file in os.listdir(root_folder):
        if file.endswith(".spe"):
            file_path = os.path.join(root_folder, file)
            spe_files.append(file_path)
    return spe_files


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
    return contours


# 匹配评分函数
def match_grid_to_edges(grid_contours, edge_image):
    score = 0
    for contour in grid_contours:
        mask = np.zeros(edge_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(int)], -1, 255, thickness=cv2.FILLED)
        match_score = np.sum(cv2.bitwise_and(edge_image, mask) / 255)
        score += match_score
    return score


# 查找最佳网格匹配位置
def find_best_grid_position(edge_image, rows, cols, cell_width, cell_height):
    best_score = -float('inf')
    best_grid = None
    best_position = (0, 0)

    for start_x in range(0, edge_image.shape[1] - cols * cell_width, 5):  # 以10个像素步长
        for start_y in range(0, edge_image.shape[0] - rows * cell_height, 5):
            grid_contours = create_fixed_grid(rows, cols, cell_width, cell_height, start_x, start_y)
            score = match_grid_to_edges(grid_contours, edge_image)
            if score > best_score:
                best_score = score
                best_grid = grid_contours
                best_position = (start_x, start_y)

    return best_grid, best_position, best_score


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
        # 使用合适的拟合算法，例如基于最小二乘法的多项式拟合或者椭圆拟合等
        # 这里以简单的多边形拟合为例，使用cv2.approxPolyDP
        epsilon = 0.01 * cv2.arcLength(np.array(edge_points), True)
        approx_contour = cv2.approxPolyDP(np.array(edge_points), epsilon, True)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(approx_contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

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
        refined_rect = cv2.minAreaRect(refined_contour)  # 这里的 refined_contour 已经是 numpy 数组
        refined_box = cv2.boxPoints(refined_rect)
        refined_box = np.int64(refined_box)

        # 确保细化后的矩形不超出原始矩形
        x_min = np.min(refined_box[:, 0])
        x_max = np.max(refined_box[:, 0])
        y_min = np.min(refined_box[:, 1])
        y_max = np.max(refined_box[:, 1])

        # 将细化后的矩形限制在原矩形的范围内
        x_min = max(x_min, x1)
        x_max = min(x_max, x2)
        y_min = max(y_min, y1)
        y_max = min(y_max, y2)

        # 重新构造细化后的矩形（确保它在原矩形范围内）
        refined_contour = np.array([[x_min, y_min],
                                    [x_max, y_min],
                                    [x_max, y_max],
                                    [x_min, y_max]])

        return refined_contour
    else:
        # 如果没有检测到有效的边缘点，返回原矩形的轮廓
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


# 文件夹下还有文件夹用这个函数读spe文件
def map_spe_files(root_folder):
    spe_files = []
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            # 如果是文件夹，则继续遍历其下的文件夹, os.walk返回三个值
            # root 所指的是当前正在遍历的这个文件夹的本身的地址
            # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
            for subdir, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".spe"):
                        file_path = os.path.join(subdir, file)
                        spe_files.append(file_path)
    return spe_files


def refine_grid_by_watershed(original_grid, edge_image, min_feature_threshold):
    refined_grid = []
    for contour in original_grid:
        x1, y1 = contour[0]
        x2, y2 = contour[2]

        # 提取当前小网格区域
        roi = edge_image[int(y1):int(y2), int(x1):int(x2)]

        # 如果区域内的特征少于阈值，则跳过细化
        feature_count = np.sum(roi) / 255
        if feature_count < min_feature_threshold:
            refined_grid.append(contour)  # 保持原始网格
            continue

        # 形态学操作进行前景和背景分割
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(roi, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记前景和背景
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 应用分水岭算法
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(roi_color, markers)
        roi_color[markers == -1] = [255, 0, 0]  # 用红色标记分割线

        # 提取分水岭分割出的区域，找到最大轮廓
        contours, _ = cv2.findContours(np.uint8(markers == 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # 判断是否需要重新计算（这里简单以差值小于某个阈值为例，你可以根据实际情况调整）
            width_diff = abs((x2 - x1) - w)
            height_diff = abs((y2 - y1) - h)
            if width_diff < 5 or height_diff < 5:  # 这里的5是可调整的阈值，根据实际精度需求改
                # 调用高精度算法重新计算
                refined_contour = refine_with_high_precision_algorithm(roi, x1, y1, x2, y2)

            else:
                # 确保细化后的网格不超出原网格范围
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
            refined_grid.append(contour)  # 如果没有找到合适的区域，保留原始网格

    return refined_grid


def snv_normalize(x):
    """对单个样本点的所有波段进行SNV标准化"""
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std if std != 0 else x * 0.0


def min_max_normalize_and_scale_spectra(spectra):
    """
    对高光谱数据按样本进行最小-最大归一化，并放大 1000 倍。
    参数：
        spectra: numpy 数组，形状为 (n_samples, n_bands)
    返回：
        scaled_spectra: 归一化并放大后的光谱，形状相同，类型为 np.float32
    """
    spectra = np.array(spectra)
    min_vals = np.min(spectra, axis=1, keepdims=True)  # 形状 (n_samples, 1)
    max_vals = np.max(spectra, axis=1, keepdims=True)  # 形状 (n_samples, 1)
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1e-10
    normalized_spectra = (spectra - min_vals) / denominator  # 归一化到 [0, 1]
    scaled_spectra = (normalized_spectra * 1000).astype(np.float32)  # 放大 1000 倍并转换为 float32
    return scaled_spectra


def augment_curve_smooth(specs: np.ndarray, n_augments: int = 150, amplitude: float = 600.0,
                         n_knots: int = 5, random_state: int = None) -> np.ndarray:
    """
    针对输入的单条光谱曲线（shape=(1, 100)），生成 n_augments 条平滑扰动后的曲线，
    扰动为低频随机函数，波动幅度约为 amplitude（默认500）。

    参数:
        specs (np.ndarray): 原始光谱数据，形状为 (1, 100)。
        n_augments (int): 生成扰动曲线的数量，默认 100 条。
        amplitude (float): 扰动曲线总波动幅度，默认 500（即上下波动范围约为 ±250）。
        n_knots (int): 控制点的数量，决定扰动函数的平滑程度。较少的控制点会得到更低频的扰动。
        random_state (int): 随机数种子，保证结果可重复。

    返回:
        np.ndarray: 生成的扰动曲线数据，形状为 (n_augments, 100)。
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 取原始曲线（假设 specs 的形状为 (1, 100)）
    original = specs[0]  # shape: (100,)
    n_points = original.shape[0]
    x_full = np.arange(n_points)

    augmented_curves = np.zeros((n_augments, n_points))

    # 定义控制点在原始曲线坐标下的位置
    knots_x = np.linspace(0, n_points - 1, n_knots)

    for i in range(n_augments):
        # 为每个控制点生成扰动值，范围为 [-amplitude/2, amplitude/2]
        knots_y = np.random.uniform(-amplitude / 2, amplitude / 2, n_knots)
        # 使用 cubic 插值生成平滑扰动曲线
        interp_func = interp1d(knots_x, knots_y, kind='cubic', fill_value="extrapolate")
        smooth_variation = interp_func(x_full)  # shape: (100,)

        # 将平滑扰动曲线叠加到原始曲线上
        new_curve = original + smooth_variation
        augmented_curves[i, :] = new_curve

    return augmented_curves


def preprocess_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    对光谱数据进行 SNV -> Savitzky-Golay 二阶导数 -> 均值中心化 预处理。

    参数:
        spectra (np.ndarray): 原始光谱数据，形状为 (n_samples, n_wavelengths)
        wavelengths (np.ndarray): 波长数组，形状为 (n_wavelengths,)

    返回:
        np.ndarray: 预处理后的光谱数据 (spectra_mc)，形状与输入相同
    """

    # Step 1: SNV（Standard Normal Variate）
    def snv(data):
        return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    spectra_snv = snv(spectra)

    # Step 2: Savitzky-Golay 滤波器，二阶导数（窗口长度9，2阶多项式）
    spectra_sg_deriv = savgol_filter(
        spectra_snv, window_length=11, polyorder=2, deriv=2, axis=1
    )

    # Step 3: 均值中心化（mean centering）
    scaler = StandardScaler(with_mean=True, with_std=False)
    spectra_mc = scaler.fit_transform(spectra_sg_deriv)

    return spectra_mc


WAVELENGTH_INFO = np.array([389.5, 391.71, 393.92, 396.13, 398.34, 400.55, 402.75, 404.95, 407.15, 409.34,
                            411.54, 413.73, 415.92, 418.11, 420.29, 422.47, 424.65, 426.83, 429.01, 431.18,
                            433.36, 435.52, 437.69, 439.86, 442.02, 444.18, 446.34, 448.5, 450.66, 452.81,
                            454.96, 457.11, 459.26, 461.4, 463.55, 465.69, 467.83, 469.97, 472.1, 474.24,
                            476.37, 478.5, 480.63, 482.75, 484.88, 487., 489.12, 491.24, 493.36, 495.48,
                            497.59, 499.71, 501.82, 503.93, 506.03, 508.14, 510.24, 512.35, 514.45, 516.55,
                            518.65, 520.74, 522.84, 524.93, 527.02, 529.11, 531.2, 533.29, 535.38, 537.46,
                            539.54, 541.62, 543.7, 545.78, 547.86, 549.94, 552.01, 554.08, 556.16, 558.23,
                            560.3, 562.36, 564.43, 566.5, 568.56, 570.62, 572.68, 574.74, 576.8, 578.86,
                            580.92, 582.97, 585.03, 587.08, 589.13, 591.18, 593.23, 595.28, 597.33, 599.38,
                            601.42, 603.47, 605.51, 607.55, 609.6, 611.64, 613.68, 615.72, 617.75, 619.79,
                            621.83, 623.86, 625.9, 627.93, 629.96, 631.99, 634.02, 636.05, 638.08, 640.11,
                            642.14, 644.17, 646.19, 648.22, 650.24, 652.27, 654.29, 656.31, 658.33, 660.35,
                            662.38, 664.4, 666.41, 668.43, 670.45, 672.47, 674.49, 676.5, 678.52, 680.53,
                            682.55, 684.56, 686.58, 688.59, 690.6, 692.62, 694.63, 696.64, 698.65, 700.66,
                            702.67, 704.68, 706.69, 708.7, 710.71, 712.72, 714.73, 716.74, 718.75, 720.75,
                            722.76, 724.77, 726.78, 728.78, 730.79, 732.8, 734.8, 736.81, 738.82, 740.82,
                            742.83, 744.83, 746.84, 748.85, 750.85, 752.86, 754.86, 756.87, 758.87, 760.88,
                            762.88, 764.89, 766.9, 768.9, 770.91, 772.91, 774.92, 776.93, 778.93, 780.94,
                            782.95, 784.95, 786.96, 788.97, 790.97, 792.98, 794.99, 797., 799.01, 801.02,
                            803.03, 805.03, 807.04, 809.05, 811.07, 813.08, 815.09, 817.1, 819.11, 821.12,
                            823.14, 825.15, 827.16, 829.18, 831.19, 833.21, 835.22, 837.24, 839.26, 841.27,
                            843.29, 845.31, 847.33, 849.35, 851.37, 853.39, 855.41, 857.44, 859.46, 861.48,
                            863.51, 865.53, 867.56, 869.58, 871.61, 873.64, 875.67, 877.7, 879.73, 881.76,
                            883.79, 885.83, 887.86, 889.89, 891.93, 893.97, 896., 898.04, 900.08, 902.12,
                            904.16, 906.21, 908.25, 910.29, 912.34, 914.39, 916.43, 918.48, 920.53, 922.58,
                            924.63, 926.69, 928.74, 930.8, 932.85, 934.91, 936.97, 939.03, 941.09, 943.15,
                            945.22, 947.28, 949.35, 951.41, 953.48, 955.55, 957.62, 959.7, 961.77, 963.85,
                            965.92, 968., 970.08, 972.16, 974.24, 976.33, 978.41, 980.5, 982.59, 984.68,
                            986.77, 988.86, 990.96, 993.05, 995.15, 997.25, 999.35, 1001.45, 1003.56, 1005.66])
