import os
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image, envi
from scipy.ndimage import median_filter
from multiprocessing import Pool
from functools import partial
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 文件目录配置
WHITE_REF_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2025-04-20采样数据-黑白采样\白参考30分钟后'
DARK_REF_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2025-04-20采样数据-黑白采样\暗信号30分钟后'
SAMPLE_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data'
OUTPUT_DIR = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data_dw'


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
    logger.info(f"已保存: {output_path}.hdr 和 {data_file}")


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


def load_and_average_white_references(white_ref_dir, num_images=10):
    """加载并平均入射光强（透射光）图像"""
    incident_light = []
    for i in range(num_images):
        file_base = os.path.join(white_ref_dir, f'{i}')
        if os.path.exists(file_base + '.hdr'):
            img = load_hyperspectral_image(file_base)
            incident_light.append(img)
    I_0_avg = np.mean(incident_light, axis=0)
    I_0_avg = correct_bad_columns(I_0_avg)
    return I_0_avg


def load_and_average_dark_references(dark_ref_dir, num_images=10):
    """加载并平均暗信号图像"""
    dark_refs = []
    for i in range(num_images):
        file_base = os.path.join(dark_ref_dir, f'{i}')
        if os.path.exists(file_base + '.hdr'):
            img = load_hyperspectral_image(file_base)
            dark_refs.append(img)
    D_avg = np.mean(dark_refs, axis=0)
    D_avg = correct_bad_columns(D_avg)
    return D_avg


def correct_sample_image(sample_image, I_0_avg, D_avg, ref_width):
    """执行透射率校正，校准公式：T = (I - D) / (I_0 - D)"""
    sample_image = correct_bad_columns(sample_image)

    sample_width = sample_image.shape[0]

    # 处理样本宽度与参考宽度不一致的情况
    if sample_width > ref_width:
        logger.info(f"样本宽度 {sample_width} 大于参考图宽度 {ref_width}，裁剪样本至 {ref_width}")
        sample_image = sample_image[:ref_width, :, :]
        sample_width = ref_width
    elif sample_width < ref_width:
        logger.info(f"样本宽度 {sample_width} 小于参考图宽度 {ref_width}，裁剪参考图至 {sample_width}")
        I_0_avg = I_0_avg[:sample_width, :, :]
        D_avg = D_avg[:sample_width, :, :]

    I_0_avg_cropped = I_0_avg
    D_avg_cropped = D_avg

    # 计算分母并缓存
    denominator = I_0_avg_cropped - D_avg_cropped
    zero_count = np.sum(denominator == 0)
    total_count = denominator.size
    if zero_count > 0:
        median_nonzero = np.median(denominator[denominator != 0])
        denominator[denominator == 0] = median_nonzero * 1e-3

    T_raw = (sample_image - D_avg_cropped) / denominator
    T = np.clip(T_raw, 0, 1)
    return T


def process_sample(base, I_0_avg, D_avg, ref_width):
    """处理单个样本图像"""
    try:
        img_path = os.path.join(SAMPLE_DIR, base)
        logger.info(f"正在处理：{base}")

        # 加载图像
        sample = load_hyperspectral_image(img_path)

        # 执行透射率校正
        corrected = correct_sample_image(sample, I_0_avg, D_avg, ref_width)

        # 保存结果
        output_path = os.path.join(OUTPUT_DIR, f"{base}")
        save_hyperspectral_image(corrected, output_path)

        del sample
        del corrected

        return base
    except Exception as e:
        logger.error(f"处理 {base} 时发生错误: {str(e)}")
        with open(os.path.join(OUTPUT_DIR, 'error_log.txt'), 'a') as f:
            f.write(f"{base}: {str(e)}\n")
        return base, None, None

def process_samples(I_0_avg, D_avg):
    """并行处理所有样品图像"""
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
            else:
                logger.warning(f"跳过 {f}，未找到对应的数据文件")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取参考宽度
    ref_width = I_0_avg.shape[0]
    process_func = partial(process_sample, I_0_avg=I_0_avg, D_avg=D_avg, ref_width=ref_width)

    # 并行处理
    with Pool(processes=2) as pool:
        results = pool.map(process_func, sample_bases)




def main():
    # 加载入射光强和暗信号
    I_0_avg = load_and_average_white_references(WHITE_REF_DIR)
    D_avg = load_and_average_dark_references(DARK_REF_DIR)

    # 不再提前裁剪 I_0_avg 和 D_avg，交给 correct_sample_image 处理
    logger.info(f"入射光强 (I_0_avg) 最大值: {I_0_avg.max()}")
    logger.info(f"暗信号 (D_avg) 最大值: {D_avg.max()}")

    # 处理样品图像
    process_samples(I_0_avg, D_avg)


if __name__ == "__main__":
    main()