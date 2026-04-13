import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from osgeo import gdal

# 排序1（按照y坐标，从小到大6个一组，总共7组，对每组内按照x坐标从小到大排序）--针对鸡蛋横着排列
def sort_egg_line(list):
    list.sort(key=lambda item: item[1], reverse=True)
    grouped_contours = [list[i:i + 6] for i in range(0, len(list), 6)]
    for group in grouped_contours:
        group.sort(key=lambda item: item[0])
    sorted_contours_info = [item for group in grouped_contours for item in group]
    return sorted_contours_info

# 匹配评分函数
def match_grid_to_edges(grid_contours, edge_image):
    score = 0
    for contour in grid_contours:
        mask = np.zeros(edge_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(int)], -1, 255, thickness=cv2.FILLED)
        match_score = np.sum(cv2.bitwise_and(edge_image, mask) / 255)
        score += match_score
    return score

# 创建固定网格
def create_fixed_grid(rows, cols, cell_width, cell_height, start_x, start_y):
    contours = []
    for row in range(rows):
        offset = cell_width // 2 if row % 2 == 1 else 0
        for col in range(cols):
            x1 = start_x + col * cell_width + offset
            y1 = start_y + row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            contours.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    return contours

def refine_with_high_precision_algorithm(roi, x1, y1, x2, y2):
    if len(roi.shape) == 2:
        gray_roi = roi
    else:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_roi, 50, 150)
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        edge_points = np.squeeze(corners).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        cv2.cornerSubPix(gray_roi, edge_points, win_size, zero_zone, criteria)
    else:
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    if edge_points.size > 0:
        epsilon = 0.01 * cv2.arcLength(np.array(edge_points), True)
        approx_contour = cv2.approxPolyDP(np.array(edge_points), epsilon, True)
        rect = cv2.minAreaRect(approx_contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        angle = rect[2]
        if angle < -45:
            angle += 90

        center = (np.sum(box[:, 0]) / 4, np.sum(box[:, 1]) / 4)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        box = np.array([[x, y, 1] for x, y in box])
        new_box = np.dot(box, M.T)[:, :2]

        refined_contour = []
        for point in new_box:
            px, py = point
            refined_contour.append([px + x1, py + y1])

        refined_contour = np.array(refined_contour, dtype=np.float32)
        refined_rect = cv2.minAreaRect(refined_contour)
        refined_box = cv2.boxPoints(refined_rect)
        refined_box = np.int64(refined_box)

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
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

def refine_grid_by_watershed(original_grid, edge_image, min_feature_threshold):
    refined_grid = []
    for contour in original_grid:
        x1, y1 = contour[0]
        x2, y2 = contour[2]
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

# 如果需要，spe和png文件顺序一致，对二维数组进行左右翻转并逆时针90°
def trans(data):
    data = np.fliplr(data)
    data = np.rot90(data, k=1)
    return data

# 查找最佳网格匹配位置
def find_best_grid_position(edge_image, rows, cols, cell_width, cell_height):
    best_score = -float('inf')
    best_grid = None
    best_position = (0, 0)

    for start_x in range(0, edge_image.shape[1] - cols * cell_width, 5):
        for start_y in range(0, edge_image.shape[0] - rows * cell_height, 5):
            grid_contours = create_fixed_grid(rows, cols, cell_width, cell_height, start_x, start_y)
            score = match_grid_to_edges(grid_contours, edge_image)
            if score > best_score:
                best_score = score
                best_grid = grid_contours
                best_position = (start_x, start_y)

    return best_grid, best_position, best_score

# 得到最清晰的波段对应的图
def getOnePicture(dataset, i):
    return dataset.GetRasterBand(i).ReadAsArray()

# 某文件夹下面的所有spe文件
def read_spe_files1(root_folder):
    spe_files = []
    for file in os.listdir(root_folder):
        if file.endswith(".spe"):
            file_path = os.path.join(root_folder, file)
            spe_files.append(file_path)
    return spe_files

# 读取数据和处理函数
def read_data(spe_file):
    try:
        dataset = gdal.Open(spe_file)
        return dataset
    except RuntimeError as e:
        print(f"GDAL 错误: {e}")

def SegEgg(path):
    try:
        data_index = os.path.basename(path).split('.')[0]
        dataset = read_data(path)

        # 获取第 150 个波段的图像，用于预览
        data = getOnePicture(dataset, 150)
        data = trans(data)
        data = cv2.convertScaleAbs(data)

        # 应用双边滤波以平滑图像，同时保留边缘
        data = cv2.bilateralFilter(data, d=9, sigmaColor=75, sigmaSpace=75)

        # 将灰度图转换为RGB空间的灰阶图（R=G=B=灰度值）
        output_image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        # 保存RGB空间的灰阶图

        os.makedirs(preview_dir, exist_ok=True)
        gray_rgb_path = os.path.join(preview_dir, f'gray_rgb_{data_index}.png')
        cv2.imwrite(gray_rgb_path, output_image)
        print(f"RGB空间的灰阶图已保存至: {gray_rgb_path}")

        # 显示RGB空间的灰阶图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title(f"Gray RGB Image for {data_index}")
        plt.imshow(output_image)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.show()

        dataset = None

    except Exception as e:
        print(f"处理文件 {path} 时发生错误: {e}")
    finally:
        if 'dataset' in locals():
            dataset = None

if __name__ == '__main__':
    preview_dir = '../gray/raw_all_12d12h_0908'
    data_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_0909_13d6h-12h_8000"
    spe_files_paths = read_spe_files1(data_path)
    MAX_WORKERS = min(4, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(SegEgg, path): path for path in spe_files_paths}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"文件 {path} 处理失败，错误信息: {e}")