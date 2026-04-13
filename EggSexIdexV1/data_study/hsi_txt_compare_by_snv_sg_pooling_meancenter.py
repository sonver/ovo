import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def snv_normalize(spectrum):
    """应用 SNV 归一化：SNV(x) = (x - mean(x)) / std(x)"""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std == 0:
        print("警告: 标准差为 0，返回原始数据")
        return spectrum
    return (spectrum - mean) / std

def load_txt_spectrum(txt_path):
    """读取 txt 光谱文件，假设每行是一个样本，每行有150个波段"""
    data = np.loadtxt(txt_path)
    # 如果是单行（1x150），保持 shape
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data  # shape: [num_samples, num_bands]

def preprocess_pipeline(data, use_median=True):
    """
    完整预处理流程：
    1. SNV
    2. SG 一阶导
    3. mean/median pooling
    4. 中心化
    """
    # 1) SNV
    snv_data = np.apply_along_axis(snv_normalize, 1, data)

    # 2) SG (一阶导)
    sg_data = savgol_filter(snv_data, window_length=9, polyorder=2, deriv=1, axis=1)

    # 3) pooling
    if use_median:
        spectrum = np.median(sg_data, axis=0)
    else:
        spectrum = np.mean(sg_data, axis=0)

    # 4) 中心化
    spectrum = spectrum - spectrum.mean()
    return spectrum

def plot_txt_spectra(file1_txt, file2_txt, wavelengths=None, use_median=True):
    """绘制两个 txt 光谱文件的原始和预处理后的曲线"""
    data1 = load_txt_spectrum(file1_txt)
    data2 = load_txt_spectrum(file2_txt)

    # 默认波长轴
    if wavelengths is None:
        wavelengths = np.arange(data1.shape[1])

    # 原始平均光谱
    mean_spec1 = np.mean(data1, axis=0)
    mean_spec2 = np.mean(data2, axis=0)

    # 预处理
    proc_spec1 = preprocess_pipeline(data1, use_median=use_median)
    proc_spec2 = preprocess_pipeline(data2, use_median=use_median)

    output_dir = os.path.dirname(file1_txt)

    # 原始对比
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths, mean_spec1, label='File 1 Raw Mean Spectrum', alpha=0.7)
    ax.plot(wavelengths, mean_spec2, label='File 2 Raw Mean Spectrum', alpha=0.7)
    ax.set_xlabel('Band Index / Wavelength')
    ax.set_ylabel('Intensity')
    ax.set_title('Raw Mean Spectra')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_curves_raw.png'), bbox_inches='tight')
    plt.show()

    # 预处理后对比
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavelengths, proc_spec1, label='File 1 Preprocessed', alpha=0.7)
    ax.plot(wavelengths, proc_spec2, label='File 2 Preprocessed', alpha=0.7)
    ax.set_xlabel('Band Index / Wavelength')
    ax.set_ylabel('Processed Intensity')
    ax.set_title('Preprocessed Spectra (SNV→SG1→Pooling→Center)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_curves_preprocessed.png'), bbox_inches='tight')
    plt.show()

    print(f"图像已保存至: {output_dir}")

if __name__ == "__main__":
    file1_txt = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\seg_all_12d18h\egg6022025090700001603-1.txt"
    file2_txt = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\seg_all_13d12h\egg6022025090800000103-1.txt"
    # 如果有 wavelength.txt，可以替换 np.arange(150)
    plot_txt_spectra(file1_txt, file2_txt, wavelengths=np.arange(150), use_median=True)
