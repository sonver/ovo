# 模拟抽帧

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import logging
from spectral.io import envi
from matplotlib import cm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hyperspectral_image(hdr_path, spe_path):
    """
    加载高光谱图像（.hdr 和 .spe 文件对）。
    :param hdr_path: .hdr 文件路径
    :param spe_path: .spe 文件路径
    :return: 高光谱数据 (numpy 数组)
    """
    try:
        # 使用 spectral 库加载 ENVI 格式文件
        img = envi.open(hdr_path, spe_path)
        data = np.array(img.load())  # 形状为 (Height, Width, Bands) = (lines, samples, bands)
        logger.info(f"加载文件 {hdr_path} 成功，形状: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"加载文件 {hdr_path} 失败: {e}")
        return None


def downsample_width(data, ratio):
    """
    对高光谱数据进行抽帧（减少 Width 维度，即 samples）。
    :param data: 高光谱数据，形状为 (Height, Width, Bands)
    :param ratio: 抽帧比例（0 < ratio <= 1），例如 1/2 表示抽取一半的 Width
    :return: 抽帧后的数据
    """
    try:
        if not 0 < ratio <= 1:
            raise ValueError("抽帧比例必须在 (0, 1] 范围内")

        # 第二维是 Width (Width,Height, Bands)
        width, height, bands = data.shape
        new_width = int(width * ratio)
        step = int(1 / ratio)  # 步长，例如 ratio=1/2 时，step=2（每隔 2 个取 1 个）

        # 抽帧：按步长取 Width 维
        downsampled_data = data[::step, :, :]
        downsampled_data = downsampled_data[:new_width, :, :]  # 确保 Width 精确
        logger.info(f"抽帧完成，原始形状: {data.shape}，抽帧后形状: {downsampled_data.shape}")
        return downsampled_data
    except Exception as e:
        logger.error(f"抽帧失败: {e}")
        return None


def save_hyperspectral_image(data, output_hdr_path):
    """
    保存抽帧后的高光谱数据为 ENVI 格式（.hdr 和 .spe 文件）。
    :param data: 抽帧后的高光谱数据，形状为 (Height, Width, Bands)
    :param output_hdr_path: 输出 .hdr 文件路径
    """
    try:
        # 使用 spectral.io.envi.save_image 保存数据，指定数据文件扩展名为 .spe
        envi.save_image(output_hdr_path, data, dtype=np.float32, ext='.spe', force=True)
        logger.info(f"抽帧后的数据已保存至: {output_hdr_path} (数据文件为 {output_hdr_path.replace('.hdr', '.spe')})")
    except Exception as e:
        logger.error(f"保存抽帧数据失败: {e}")


def preview_grayscale_image(data, band_idx, output_path, title_prefix=""):
    """
    预览某个波段映射到指定）
    :param output_path: 保存灰阶图的路径 RGB 空间后的灰阶图。
    :param data: 高光谱数据，形状为 (Height, Width, Bands)
    :param band_idx: 选择的波段索引（必须
    :param title_prefix: 标题前缀，用于区分原始和抽帧后的图像
    """
    try:
        height, width, bands = data.shape
        if band_idx >= bands:
            raise ValueError(f"波段索引 {band_idx} 超出范围，最大为 {bands - 1}")

        # 提取某个波段，形状为 (Height, Width)
        band_data = data[:, :, band_idx]

        # 归一化数据到 [0, 1]
        band_min, band_max = np.min(band_data), np.max(band_data)
        if band_max == band_min:
            normalized_data = np.zeros_like(band_data)
        else:
            normalized_data = (band_data - band_min) / (band_max - band_min)

        # 使用 colormap（例如 jet）映射到 RGB 空间
        colormap = cm.get_cmap('jet')  # jet 是一种常见的伪彩色映射
        rgb_image = colormap(normalized_data)  # 形状为 (Height, Width, 4)，包含 RGBA

        # 提取 RGB 通道（忽略 Alpha 通道）
        rgb_image = rgb_image[:, :, :3]  # 形状为 (Height, Width, 3)

        # 将 RGB 转换为灰阶图
        # 灰度公式：gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        grayscale = (0.2989 * rgb_image[:, :, 0] +
                     0.5870 * rgb_image[:, :, 1] +
                     0.1140 * rgb_image[:, :, 2])  # 形状为 (Height, Width)

        # 绘制灰阶图
        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale, cmap='gray')
        plt.title(f"{title_prefix}波段 {band_idx} 映射到 RGB 空间后的灰阶图")
        plt.xlabel('宽度 (Width)')
        plt.ylabel('高度 (Height)')
        plt.colorbar(label='灰度强度 (Intensity)')
        plt.tight_layout()

        # 保存灰阶图
        plt.savefig(output_path)
        logger.info(f"灰阶图已保存至: {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"预览灰阶图失败: {e}")


def main():
    # 数据目录和输出目录
    data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\egg_raw_data"
    output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\egg_raw_data_half"
    preview_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\results\half_previews"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    # 抽帧比例（例如 1/2 表示抽取一半的 Width）
    ratio = 0.5

    # 选择的波段索引（例如 150）
    band_idx = 150

    # 检查数据目录
    if not os.path.exists(data_dir):
        logger.error(f"数据目录 {data_dir} 不存在")
        return

    # 查找所有 .hdr 文件
    hdr_files = glob.glob(os.path.join(data_dir, "*.hdr"))
    if not hdr_files:
        logger.error(f"未在 {data_dir} 中找到 .hdr 文件")
        return

    # 批处理所有高光谱图片
    for hdr_path in hdr_files:
        # 获取对应的 .spe 文件
        spe_path = hdr_path.replace('.hdr', '.spe')
        if not os.path.exists(spe_path):
            logger.error(f"未找到对应的 .spe 文件: {spe_path}")
            continue

        # 加载高光谱数据
        data = load_hyperspectral_image(hdr_path, spe_path)
        if data is None:
            continue

        # 抽帧前的预览
        base_name = os.path.basename(hdr_path)
        original_preview_path = os.path.join(preview_dir, f"original_grayscale_{base_name}.png")
        preview_grayscale_image(data, band_idx=band_idx, output_path=original_preview_path, title_prefix="原始图像 - ")

        # 抽帧
        downsampled_data = downsample_width(data, ratio)
        if downsampled_data is None:
            continue

        # 保存抽帧后的数据，文件名与源文件保持一致
        output_hdr_path = os.path.join(output_dir, base_name)
        save_hyperspectral_image(downsampled_data, output_hdr_path)

        # 抽帧后的预览
        preview_path = os.path.join(preview_dir, f"grayscale_{base_name}.png")
        preview_grayscale_image(downsampled_data, band_idx=band_idx, output_path=preview_path,
                                title_prefix="抽帧后图像 - ")

        # 打印抽帧后的形状
        logger.info(f"处理完成 {base_name}，抽帧后形状: {downsampled_data.shape}")


if __name__ == "__main__":
    main()
