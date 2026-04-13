import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spectral.io import envi
from images_center import IMAGES_CENTER  # 导入 IMAGES_CENTER 字典
from multiprocessing import Pool


def trans(data):
    """
    对高光谱数据进行变换：左右翻转 + 逆时针旋转90度。
    :param data: 高光谱数据，形状为 (Height, Width, Bands)
    :return: 变换后的数据
    """
    data = np.fliplr(data)  # 矩阵左右翻转
    data = np.rot90(data, k=1)  # 矩阵逆时针旋转90度
    return data


def load_hyperspectral_image(hdr_path, spe_path):
    """
    加载高光谱图像（.hdr 和 .spe 文件对），并应用变换。
    :param hdr_path: .hdr 文件路径
    :param spe_path: .spe 文件路径
    :return: 变换后的高光谱数据 (numpy 数组)
    """
    try:
        img = envi.open(hdr_path, spe_path)
        data = np.array(img.load())  # 形状为 (Height, Width, Bands)
        print(f"加载文件 {hdr_path} 成功，原始形状: {data.shape}")

        # 应用变换
        data = trans(data)
        print(f"变换后形状: {data.shape}")
        return data
    except Exception as e:
        print(f"加载文件 {hdr_path} 失败: {e}")
        return None


def save_to_txt(data, output_txt_path):
    """
    将高光谱数据保存为 .txt 文件，400 行，每行 150 个数据。
    :param data: 高光谱数据，形状为 (400, 150)
    :param output_txt_path: 输出 .txt 文件路径
    """
    try:
        np.savetxt(output_txt_path, data, fmt='%.6f', delimiter=' ')
        print(f"数据已保存至: {output_txt_path}")
    except Exception as e:
        print(f"保存数据失败: {e}")


def crop_egg_patch(data, center, patch_size=(20, 20, 150)):
    """
    以中心点裁剪高光谱图像块，并展平为 (400, 150)。
    :param data: 高光谱数据，形状为 (Height, Width, Bands)
    :param center: 中心点坐标 (x, y)
    :param patch_size: 裁剪块的形状 (patch_height, patch_width, bands)
    :return: 展平后的数据，形状为 (400, 150)
    """
    height, width, bands = data.shape
    patch_height, patch_width, patch_bands = patch_size

    if bands < patch_bands:
        print(f"警告: 数据波段数 ({bands}) 小于目标波段数 ({patch_bands})")
        return None

    # 计算裁剪区域
    half_height = patch_height // 2
    half_width = patch_width // 2
    x, y = center

    # 确保裁剪区域不超出图像边界
    x_start = max(0, x - half_width)
    x_end = min(width, x + half_width)
    y_start = max(0, y - half_height)
    y_end = min(height, y + half_height)

    # 调整裁剪区域大小为 patch_size
    if x_end - x_start < patch_width:
        if x_start == 0:
            x_end = x_start + patch_width
        else:
            x_start = x_end - patch_width
    if y_end - y_start < patch_height:
        if y_start == 0:
            y_end = y_start + patch_height
        else:
            y_start = y_end - patch_height

    # 裁剪波段，从第 100 个波段开始
    start_band = 100
    end_band = min(start_band + patch_bands, bands)  # 确保不超出波段总数
    if end_band - start_band < patch_bands:
        print(
            f"警告: 从波段 {start_band} 到 {end_band} 只有 {end_band - start_band} 个波段，小于目标 {patch_bands} 个波段")
        return None

    # 裁剪图像块
    patch = data[y_start:y_end, x_start:x_end, start_band:end_band]

    # 确保形状正确
    if patch.shape != patch_size:
        print(f"裁剪块形状 {patch.shape} 不符合目标形状 {patch_size}")
        return None

    # 展平为 (400, 150)
    patch = patch.reshape(patch_height * patch_width, patch_bands)
    return patch


def preview_egg_centers(data, centers_grid, plate, output_dir, patch_size=(20, 20, 150), rows=7, cols=6,
                        save_only=False):
    """
    使用 matplotlib 显示或保存每盘蛋的中心点和裁剪区域预览。
    :param data: 高光谱数据，形状为 (Height, Width, Bands)
    :param centers_grid: 中心点坐标网格
    :param plate: 盘号
    :param output_dir: 输出目录（用于保存预览图像）
    :param patch_size: 裁剪块的形状 (patch_height, patch_width, bands)
    :param rows: 行数
    :param cols: 列数
    :param save_only: 是否仅保存预览图像（不显示）
    :return: 是否继续处理（仅在 save_only=False 时有效）
    """
    # 提取第 100 个波段作为灰度图像
    band_idx = 100
    if band_idx >= data.shape[2]:
        band_idx = data.shape[2] - 1
    preview_img = data[:, :, band_idx]

    # 归一化到 0-1 用于显示
    preview_img = (preview_img - preview_img.min()) / (preview_img.max() - preview_img.min())

    # 创建图形
    plt.figure(figsize=(10, 8))
    plt.imshow(preview_img, cmap='gray')
    plt.title(f"Plate {plate} - Centers and Crop Regions")

    patch_height, patch_width, _ = patch_size
    half_height = patch_height // 2
    half_width = patch_width // 2

    # 绘制中心点和裁剪区域
    for row in range(rows):
        for col in range(cols):
            center = centers_grid[row][col]
            if center is None:
                continue

            x, y = center
            # 绘制中心点（红色圆点）
            plt.scatter(x, y, color='red', s=50, marker='o', label='Center' if row == 0 and col == 0 else None)

            # 绘制裁剪区域（绿色矩形）
            x_start = max(0, x - half_width)
            y_start = max(0, y - half_height)
            rect = Rectangle((x_start, y_start), patch_width, patch_height, linewidth=1, edgecolor='green',
                             facecolor='none',
                             label='Crop Region' if row == 0 and col == 0 else None)
            plt.gca().add_patch(rect)

            # 标注蛋序号
            egg_index = row * cols + col + 1
            plt.text(x + 10, y - 10, f"{egg_index}", color='yellow', fontsize=8)

    plt.legend()
    plt.axis('on')
    plt.tight_layout()

    if save_only:
        # 保存预览图像
        preview_dir = os.path.join(output_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"preview_plate_{plate}.png")
        plt.savefig(preview_path)
        plt.close()
        print(f"预览图像已保存至: {preview_path}")
        return True  # 多进程模式下继续处理
    else:
        # 在 PyCharm 中内嵌显示
        plt.show()

        # 提示用户是否继续
        user_input = input(f"Plate {plate} 预览完成，输入 'q' 退出，或按 Enter 继续: ").strip().lower()
        if user_input == 'q':
            return False  # 用户选择退出
        return True  # 继续处理


def process_plate(plate, data_dir, output_dir, patch_size=(20, 20, 150), flattened_shape=(400, 150), rows=7, cols=6):
    """
    处理单个盘号的数据（加载、变换、裁剪、保存）。
    :param plate: 盘号
    :param data_dir: 输入数据目录
    :param output_dir: 输出目录
    :param patch_size: 裁剪块形状
    :param flattened_shape: 展平后的形状
    :param rows: 行数
    :param cols: 列数
    :return: 处理后的高光谱数据（用于主进程预览）
    """
    # 加载高光谱图像（包含变换）
    hdr_path = os.path.join(data_dir, f"{plate}.hdr")
    spe_path = os.path.join(data_dir, f"{plate}.spe")
    if not (os.path.exists(hdr_path) and os.path.exists(spe_path)):
        print(f"盘号 {plate} 的文件不存在，跳过")
        return None

    # 加载图像
    data = load_hyperspectral_image(hdr_path, spe_path)
    if data is None:
        return None

    # 获取中心点坐标
    centers_grid = IMAGES_CENTER[plate]

    # 遍历每行每列
    for row in range(rows):
        for col in range(cols):
            center = centers_grid[row][col]
            if center is None:
                print(f"盘号 {plate} 行 {row + 1} 列 {col + 1} 缺少中心点坐标，跳过")
                continue

            # 直接使用中心点坐标
            x, y = center

            # 裁剪图像块并展平
            patch = crop_egg_patch(data, (x, y), patch_size)
            if patch is None:
                print(f"盘号 {plate} 行 {row + 1} 列 {col + 1} 裁剪失败，跳过")
                continue

            # 确保形状为 (400, 150)
            if patch.shape != flattened_shape:
                print(f"展平后形状 {patch.shape} 不符合目标形状 {flattened_shape}")
                continue

            # 计算蛋序号（从 1 到 42）
            egg_index = row * cols + col + 1  # 蛋序号从 1 开始

            # 保存为 .txt 文件
            output_txt_path = os.path.join(output_dir, f"egg{plate}-{egg_index}.txt")
            save_to_txt(patch, output_txt_path)

    return (data, centers_grid, plate)  # 返回数据、中心点和盘号用于预览


def main(use_multiprocessing=False):
    """
    主函数，处理高光谱图像并裁剪鸡蛋区域。
    :param use_multiprocessing: 是否使用多进程处理（True: 多进程并保存预览图像; False: 串行并交互式预览）
    """

    os.makedirs(output_dir, exist_ok=True)

    # 目标裁剪形状
    patch_size = (20, 20, 150)  # (Height, Width, Bands)
    flattened_shape = (20 * 20, 150)  # 展平后为 (400, 150)

    # 每盘规格：7 行 6 列
    rows, cols = 7, 6

    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"数据目录 {data_dir} 不存在")
        return

    # 获取所有盘号
    plates = sorted(IMAGES_CENTER.keys())
    print(f"处理盘号: {plates}")

    if use_multiprocessing:
        # 多进程模式
        num_processes = min(len(plates), os.cpu_count())  # 进程数不超过 CPU 核心数或盘号数量
        print(f"使用 {num_processes} 个进程进行处理")

        with Pool(processes=num_processes) as pool:
            # 处理每个盘号
            results = pool.starmap(
                process_plate,
                [(plate, data_dir, output_dir, patch_size, flattened_shape, rows, cols) for plate in plates]
            )

        # 处理完成后，生成并保存预览图像
        for result in results:
            if result is None:
                continue
            data, centers_grid, plate = result
            # 保存预览图像
            preview_egg_centers(data, centers_grid, plate, output_dir, patch_size, rows, cols, save_only=True)

    else:
        # 串行模式（交互式预览）
        for plate in plates:
            # 处理单个盘号
            result = process_plate(plate, data_dir, output_dir, patch_size, flattened_shape, rows, cols)
            if result is None:
                continue

            data, centers_grid, plate = result
            # 显示预览
            continue_processing = preview_egg_centers(data, centers_grid, plate, output_dir, patch_size, rows, cols,
                                                      save_only=False)
            if not continue_processing:
                print("用户选择退出程序")
                break

    print("所有鸡蛋图像块裁剪并保存为 .txt 文件完成！")


if __name__ == "__main__":
    # 数据目录和输出目录
    data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data"
    output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_seg_data_fixed"
    # 预览路径 "D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_seg_data_fixed\previews"

    # 可根据需要设置 use_multiprocessing 为 True 或 False
    main(use_multiprocessing=False)
