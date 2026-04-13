import os
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image, envi
from scipy.ndimage import median_filter
import logging

# 设置日志（减少磁盘写入）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 减少日志频率，仅记录关键信息
logger.setLevel(logging.WARNING)

# 文件目录配置
DARK_REF_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2025-04-20采样数据-黑白采样\暗信号30分钟后'
SAMPLE_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data'
OUTPUT_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data_d'


def load_hyperspectral_image(base_path, ext='.spe'):
    """加载ENVI格式的高光谱图像，指定扩展名以减少检查"""
    base_path = base_path.replace('.hdr', '')
    hdr_file = base_path + '.hdr'
    data_file = base_path + ext
    if os.path.exists(data_file):
        img = envi.open(hdr_file, data_file)
        return np.array(img.load())
    raise FileNotFoundError(f"找不到数据文件：{data_file}")


def save_hyperspectral_image(data, output_path, metadata=None):
    """保存高光谱图像为ENVI格式"""
    data = np.array(data, dtype=np.float32)
    if metadata is None:
        metadata = {
            'lines': data.shape[0],
            'samples': data.shape[1],
            'bands': data.shape[2],
            'data type': 4,
            'interleave': 'bsq',
            'file type': 'ENVI Standard',
        }
    data_file = output_path + '.spe'
    envi.save_image(
        output_path + '.hdr',
        data,
        metadata=metadata,
        force=True,
        ext='.spe'
    )


def correct_bad_columns(data, bad_columns=[19, 328, 435]):
    """向量化校准坏列，仅对坏列附近应用中值滤波"""
    corrected_data = data.copy()

    # 向量化替换坏列
    for y in bad_columns:
        if y == 0:
            corrected_data[:, y, :] = corrected_data[:, y + 1, :]
        elif y == corrected_data.shape[1] - 1:
            corrected_data[:, y, :] = corrected_data[:, y - 1, :]
        else:
            corrected_data[:, y, :] = (corrected_data[:, y - 1, :] + corrected_data[:, y + 1, :]) / 2

    # 仅对坏列及其邻近列应用中值滤波
    for y in bad_columns:
        y_start = max(0, y - 1)
        y_end = min(corrected_data.shape[1], y + 2)
        corrected_data[:, y_start:y_end, :] = median_filter(
            corrected_data[:, y_start:y_end, :], size=(3, 3, 1)
        )
    return corrected_data


def load_and_average_dark_references(dark_ref_dir, num_images=10):
    """加载并平均暗信号图像"""
    dark_refs = []
    for i in range(num_images):
        file_base = os.path.join(dark_ref_dir, f'{i}')
        if os.path.exists(file_base + '.hdr'):
            img = load_hyperspectral_image(file_base)
            dark_refs.append(img)
            del img  # 释放内存
    D_avg = np.mean(dark_refs, axis=0)
    D_avg = correct_bad_columns(D_avg)
    return D_avg


def correct_sample_image(sample_image, D_avg, ref_width):
    """执行暗信号校准，校准公式：corrected = I - D"""
    sample_image = correct_bad_columns(sample_image)

    sample_width = sample_image.shape[0]

    # 处理样本宽度与参考宽度不一致的情况
    if sample_width > ref_width:
        sample_image = sample_image[:ref_width, :, :]
        sample_width = ref_width
    elif sample_width < ref_width:
        D_avg = D_avg[:sample_width, :, :]

    D_avg_cropped = D_avg

    # 仅执行暗信号校准
    corrected = sample_image - D_avg_cropped
    corrected = np.clip(corrected, 0, None)
    return corrected


def process_sample(base, D_avg, ref_width, output_dir):
    """处理单个样本图像"""
    try:
        img_path = os.path.join(SAMPLE_DIR, base)

        # 加载图像
        sample = load_hyperspectral_image(img_path)

        # 执行暗信号校准
        corrected = correct_sample_image(sample, D_avg, ref_width)

        # 保存结果
        output_path = os.path.join(output_dir, f"{base}")
        save_hyperspectral_image(corrected, output_path)

        result = (base, sample, corrected)

        del sample
        del corrected
        return result
    except Exception as e:
        with open(os.path.join(output_dir, 'error_log.txt'), 'a') as f:
            f.write(f"{base}: {str(e)}\n")
        return base, None, None


def process_samples(D_avg):
    """逐个处理样本图像，避免并行读写磁盘"""
    sample_bases = []
    for f in os.listdir(SAMPLE_DIR):
        if f.endswith('.hdr'):
            base = f[:-4]
            data_found = any(
                os.path.exists(os.path.join(SAMPLE_DIR, base + ext))
                for ext in ['.spe', '.dat', '.img', '.bin']
            )
            if data_found:
                sample_bases.append(base)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取参考宽度
    ref_width = D_avg.shape[0]

    # 逐个处理样本
    first_sample = True
    for base in sample_bases:
        result = process_sample(base, D_avg, ref_width, OUTPUT_DIR)
        base, sample, corrected = result

        # 可视化第一个成功处理的样本
        if first_sample and sample is not None and corrected is not None:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(sample[:, :, 150], cmap='gray')
            plt.title('Raw Sample (Band 150)')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(corrected[:, :, 150], cmap='gray')
            plt.title('Dark Signal Corrected (Band 150)')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'visualization.png'))
            plt.close()
            first_sample = False


def main():
    # 仅加载暗信号
    D_avg = load_and_average_dark_references(DARK_REF_DIR)

    # 处理样品图像
    process_samples(D_avg)


if __name__ == "__main__":
    main()