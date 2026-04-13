import os
import numpy as np
import matplotlib.pyplot as plt
import re
from osgeo import gdal
from scipy.signal import savgol_filter

def extract_wavelengths(hdr_path):
    """从 .hdr 文件中提取 wavelength 数组"""
    with open(hdr_path, 'r') as f:
        content = f.read()
    match = re.search(r'wavelength =\{\n?([\s\S]*?)\n?\}', content)
    if match:
        wavelengths_str = match.group(1)
        wavelengths = [float(w.strip()) for w in wavelengths_str.split(',') if w.strip()]
        return np.array(wavelengths)
    else:
        raise ValueError(f"Wavelength information not found in {hdr_path}")

def extract_center_spectrum(spe_path, hdr_path, dtype=np.float32, interleave='bil'):
    """从 .spe 和 .hdr 文件中提取中心点的光谱曲线"""
    dataset = gdal.Open(spe_path)
    if dataset is None:
        raise ValueError(f"Failed to open {spe_path}")
    samples, lines, bands = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount

    if samples != 480 or lines != 519 or bands != 300:
        print(f"警告: {spe_path} 尺寸为 samples={samples}, lines={lines}, bands={bands}，预期 480x519x300")

    x_center, y_center = samples // 2, lines // 2
    spectrum = [dataset.GetRasterBand(b + 1).ReadAsArray(x_center, y_center, 1, 1)[0, 0] for b in range(bands)]
    wavelengths = extract_wavelengths(hdr_path)
    if len(wavelengths) != bands:
        print(f"警告: {hdr_path} 中的波长数量 {len(wavelengths)} 不匹配波段数 {bands}")
        wavelengths = np.arange(bands)
    return wavelengths, np.array(spectrum)

def snv_normalize(spectrum):
    """应用 SNV 归一化：SNV(x) = (x - mean(x)) / std(x)"""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std == 0:
        print("警告: 标准差为 0，无法进行 SNV 归一化，返回原始数据")
        return spectrum
    return (spectrum - mean) / std

def gaussian_weights(data, sigma=1.0):
    """计算高斯权重，中心对齐"""
    length = len(data)
    x = np.arange(-(length-1)//2, (length+1)//2)
    weights = np.exp(-x ** 2 / (2 * sigma ** 2))
    weights = weights / np.sum(weights)  # 归一化
    return weights[len(weights)//2 - length//2:len(weights)//2 + length//2 + 1]

def apply_gaussian_weighting(spectrum, sigma=1.0):
    """对数据应用高斯加权平滑"""
    weights = gaussian_weights(spectrum, sigma)
    if len(weights) != len(spectrum):
        raise ValueError(f"Weight length {len(weights)} does not match spectrum length {len(spectrum)}")
    smoothed = np.convolve(spectrum, weights, mode='same')
    return smoothed

def compute_derivative(wavelengths, spectrum, use_sg=True):
    """计算光谱曲线的一阶导数，使用 SG 滤波或简单差分"""
    if use_sg:
        derivative = savgol_filter(spectrum, window_length=19, polyorder=2, deriv=1)
        deriv_wavelengths = wavelengths  # SG 保持原始长度
    else:
        derivative = np.diff(spectrum) / np.diff(wavelengths)
        deriv_wavelengths = (wavelengths[:-1] + wavelengths[1:]) / 2
    return deriv_wavelengths, derivative

def plot_spectra(file1_spe, file1_hdr, file2_spe, file2_hdr):
    """绘制原始光谱曲线、SNV 归一化后曲线、SNV+高斯加权后曲线及导数曲线"""
    wavelengths1, spectrum1 = extract_center_spectrum(file1_spe, file1_hdr)
    wavelengths2, spectrum2 = extract_center_spectrum(file2_spe, file2_hdr)

    if not np.allclose(wavelengths1, wavelengths2):
        print("警告: 两个文件的波长数组不完全一致，可能影响比较")

    # 应用 SNV 归一化
    spectrum1_snv = snv_normalize(spectrum1)
    spectrum2_snv = snv_normalize(spectrum2)

    # 应用高斯加权平滑到 SNV 后的光谱
    spectrum1_gauss = apply_gaussian_weighting(spectrum1_snv, sigma=1.0)
    spectrum2_gauss = apply_gaussian_weighting(spectrum2_snv, sigma=1.0)

    # 调试信息
    index_700 = np.searchsorted(wavelengths1, 700)
    print(f"Index near 700 nm: {index_700}")
    print(f"File 1 SNV at ~700 nm: {spectrum1_snv[index_700]:.4f}")
    print(f"File 1 Gauss at ~700 nm: {spectrum1_gauss[index_700]:.4f}")

    # 绘制原始光谱曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths1, spectrum1, label='File 1 Center Spectrum', alpha=0.7)
    ax.plot(wavelengths2, spectrum2, label='File 2 Center Spectrum', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Center Point Spectral Curves (Original)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    output_dir = os.path.dirname(file1_spe)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'spectral_curves_original.png'), bbox_inches='tight')
    plt.show()
    print(f"原始光谱曲线已保存至: {os.path.join(output_dir, 'spectral_curves_original.png')}")

    # 绘制 SNV 归一化后光谱曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths1, spectrum1_snv, label='File 1 Center Spectrum (SNV)', alpha=0.7)
    ax.plot(wavelengths2, spectrum2_snv, label='File 2 Center Spectrum (SNV)', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Center Point Spectral Curves (SNV Normalized)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_curves_snv.png'), bbox_inches='tight')
    plt.show()
    print(f"SNV 归一化光谱曲线已保存至: {os.path.join(output_dir, 'spectral_curves_snv.png')}")

    # 绘制 SNV+高斯加权后光谱曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths1, spectrum1_gauss, label='File 1 Center Spectrum (SNV + Gaussian)', alpha=0.7)
    ax.plot(wavelengths2, spectrum2_gauss, label='File 2 Center Spectrum (SNV + Gaussian)', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Smoothed Normalized Intensity')
    ax.set_title('Center Point Spectral Curves (SNV + Gaussian Weighted, sigma=1.0)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_curves_gauss.png'), bbox_inches='tight')
    plt.show()
    print(f"SNV+高斯加权光谱曲线已保存至: {os.path.join(output_dir, 'spectral_curves_gauss.png')}")

    # 计算原始导数并绘制
    deriv_wl1, deriv1 = compute_derivative(wavelengths1, spectrum1, use_sg=True)
    deriv_wl2, deriv2 = compute_derivative(wavelengths2, spectrum2, use_sg=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(deriv_wl1, deriv1, label='File 1 Derivative', alpha=0.7)
    ax.plot(deriv_wl2, deriv2, label='File 2 Derivative', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Derivative of Intensity')
    ax.set_title('Derivative of Center Point Spectral Curves (Original)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'derivative_curves_original.png'), bbox_inches='tight')
    plt.show()
    print(f"原始导数曲线已保存至: {os.path.join(output_dir, 'derivative_curves_original.png')}")

    # 计算 SNV 后导数并绘制
    deriv_wl1_snv, deriv1_snv = compute_derivative(wavelengths1, spectrum1_snv, use_sg=True)
    deriv_wl2_snv, deriv2_snv = compute_derivative(wavelengths2, spectrum2_snv, use_sg=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(deriv_wl1_snv, deriv1_snv, label='File 1 Derivative (SNV)', alpha=0.7)
    ax.plot(deriv_wl2_snv, deriv2_snv, label='File 2 Derivative (SNV)', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Derivative of Normalized Intensity')
    ax.set_title('Derivative of Center Point Spectral Curves (SNV Normalized)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'derivative_curves_snv.png'), bbox_inches='tight')
    plt.show()
    print(f"SNV 导数曲线已保存至: {os.path.join(output_dir, 'derivative_curves_snv.png')}")

    # 计算 SNV+高斯加权后导数并绘制
    deriv_wl1_gauss, deriv1_gauss = compute_derivative(wavelengths1, spectrum1_gauss, use_sg=True)
    deriv_wl2_gauss, deriv2_gauss = compute_derivative(wavelengths2, spectrum2_gauss, use_sg=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(deriv_wl1_gauss, deriv1_gauss, label='File 1 Derivative (SNV + Gaussian)', alpha=0.7)
    ax.plot(deriv_wl2_gauss, deriv2_gauss, label='File 2 Derivative (SNV + Gaussian)', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Derivative of Smoothed Normalized Intensity')
    ax.set_title('Derivative of Center Point Spectral Curves (SNV + Gaussian Weighted)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'derivative_curves_gauss.png'), bbox_inches='tight')
    plt.show()
    print(f"SNV+高斯加权导数曲线已保存至: {os.path.join(output_dir, 'derivative_curves_gauss.png')}")

    print("Plots saved as 'spectral_curves_original.png', 'spectral_curves_snv.png', 'spectral_curves_gauss.png', 'derivative_curves_original.png', 'derivative_curves_snv.png' and 'derivative_curves_gauss.png'")

    # ========= 新增：完整预处理流程 (SNV -> SG一阶 -> median pooling -> 均值中心化) =========
    # 读取整个cube
    dataset1 = gdal.Open(file1_spe)
    dataset2 = gdal.Open(file2_spe)
    cube1 = np.stack([dataset1.GetRasterBand(b+1).ReadAsArray() for b in range(dataset1.RasterCount)], axis=-1)
    cube2 = np.stack([dataset2.GetRasterBand(b+1).ReadAsArray() for b in range(dataset2.RasterCount)], axis=-1)

    # --- File1 ---
    snv_data1 = np.apply_along_axis(snv_normalize, 1, cube1.reshape(-1, cube1.shape[-1]))
    sg_data1 = savgol_filter(snv_data1, window_length=9, polyorder=2, deriv=1, axis=1)
    median_spectrum1 = np.median(sg_data1, axis=0)
    proc_spectrum1 = median_spectrum1 - np.mean(median_spectrum1)

    # --- File2 ---
    snv_data2 = np.apply_along_axis(snv_normalize, 1, cube2.reshape(-1, cube2.shape[-1]))
    sg_data2 = savgol_filter(snv_data2, window_length=9, polyorder=2, deriv=1, axis=1)
    median_spectrum2 = np.median(sg_data2, axis=0)
    proc_spectrum2 = median_spectrum2 - np.mean(median_spectrum2)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths1, proc_spectrum1, label='File 1 Preprocessed (SNV→SG1→Median→Center)', alpha=0.7)
    ax.plot(wavelengths2, proc_spectrum2, label='File 2 Preprocessed (SNV→SG1→Median→Center)', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Processed Intensity')
    ax.set_title('Custom Preprocessing (SNV→SG1→Median→Centerized)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_curves_custom_pipeline.png'), bbox_inches='tight')
    plt.show()
    print(f"完整预处理光谱曲线已保存至: {os.path.join(output_dir, 'spectral_curves_custom_pipeline.png')}")


if __name__ == "__main__":
    file1_spe = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_13d12h\6022025090800000101.spe'
    file1_hdr = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_13d12h\6022025090800000101.hdr'
    file2_spe = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_12d18h\6022025090700001601.spe'
    file2_hdr = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_12d18h\6022025090700001601.hdr'

    plot_spectra(file1_spe, file1_hdr, file2_spe, file2_hdr)