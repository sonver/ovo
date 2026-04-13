import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import ttest_ind
from sklearn.feature_selection import SelectKBest, f_classif
from tools import WAVELENGTH_INFO

def getFandMHsiData(num_f, f_start, num_m, m_start):
    f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\6month\female'
    m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\6month\male'
    f_files = []
    m_files = []

    for file in os.listdir(f_path):
        if f_start < num_f:
            if file.endswith(".txt"):
                file_path = os.path.join(f_path, file)
                f_files.append(file_path)
                f_start += 1
    for file in os.listdir(m_path):
        if m_start < num_m:
            if file.endswith(".txt"):
                file_path = os.path.join(m_path, file)
                m_files.append(file_path)
                m_start += 1
    return f_files, m_files

def apply_fft(spectra, wavelengths):
    fft_result = np.fft.fft(spectra, axis=1)
    freqs = np.fft.fftfreq(spectra.shape[1], d=(wavelengths[1] - wavelengths[0]))
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_amplitude = np.abs(fft_result)[:, pos_mask]
    return freqs, fft_amplitude

def analyze_periodicity(fft_amplitude, freqs, min_freq=0.01, max_freq=0.25, prominence=0.01):
    mean_amplitude = np.mean(fft_amplitude, axis=0)
    mean_amplitude = savgol_filter(mean_amplitude, window_length=3, polyorder=1)
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_filtered = freqs[freq_mask]
    mean_amplitude_filtered = mean_amplitude[freq_mask]
    peaks, properties = find_peaks(mean_amplitude_filtered, height=0.005, prominence=prominence)
    peak_freqs = freqs_filtered[peaks]
    peak_heights = properties['peak_heights']
    periods = 1 / peak_freqs if len(peak_freqs) > 0 else np.array([])
    return mean_amplitude_filtered, peaks, peak_freqs, peak_heights, periods, freqs_filtered

def detrend_spectrum(spectra):
    for i in range(spectra.shape[0]):
        x = np.arange(spectra.shape[1])
        coeffs = np.polyfit(x, spectra[i], 2)
        baseline = np.polyval(coeffs, x)
        spectra[i] -= baseline
    return spectra

if __name__ == '__main__':

    num_f, f_start = 200, 0
    num_m, m_start = 200, 0

    female_files, male_files = getFandMHsiData(num_f, f_start, num_m, m_start)

    f_data_list = []
    m_data_list = []

    for file in female_files:
        with open(file, 'r') as f:
            f_data_list.append(np.array([list(map(float, line.strip().split())) for line in f]))
    f_data_list = np.array(f_data_list)

    for file in male_files:
        with open(file, 'r') as f:
            m_data_list.append(np.array([list(map(float, line.strip().split())) for line in f]))
    m_data_list = np.array(m_data_list)

    start_idx, end_idx = 100, 250
    wavelengths = WAVELENGTH_INFO[start_idx:end_idx]
    num_bands = len(wavelengths)

    female_mean = np.mean(f_data_list.reshape(f_data_list.shape[0], -1, num_bands), axis=1)
    male_mean = np.mean(m_data_list.reshape(m_data_list.shape[0], -1, num_bands), axis=1)

    # 预处理步骤
    # 1. 均值中心化
    female_mean_centered = female_mean - np.mean(female_mean, axis=1, keepdims=True)
    male_mean_centered = male_mean - np.mean(male_mean, axis=1, keepdims=True)

    # 2. 去除低频趋势（基线）
    female_detrended = detrend_spectrum(female_mean_centered.copy())
    male_detrended = detrend_spectrum(male_mean_centered.copy())

    # 3. 减小平滑窗口
    window_length = 5
    polyorder = 1
    female_smooth = savgol_filter(female_detrended, window_length, polyorder, axis=1)
    male_smooth = savgol_filter(male_detrended, window_length, polyorder, axis=1)

    # 4. 一阶导数（减小平滑窗口）
    female_deriv = savgol_filter(female_smooth, window_length=5, polyorder=1, deriv=1, axis=1)
    male_deriv = savgol_filter(male_smooth, window_length=5, polyorder=1, deriv=1, axis=1)
    female_processed = female_deriv
    male_processed = male_deriv

    # 可视化预处理后的光谱
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, np.mean(female_smooth, axis=0), 'r-', label='Female Smooth')
    plt.plot(wavelengths, np.mean(male_smooth, axis=0), 'g-', label='Male Smooth')
    plt.title('Mean Spectra (After Smoothing)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, np.mean(female_deriv, axis=0), 'r-', label='Female Derivative')
    plt.plot(wavelengths, np.mean(male_deriv, axis=0), 'g-', label='Male Derivative')
    plt.title('Mean Spectra (After 1st Derivative)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 应用傅里叶变换
    freqs_female, fft_female = apply_fft(female_processed, wavelengths)
    freqs_male, fft_male = apply_fft(male_processed, wavelengths)

    # 分析周期性
    min_freq = 0.01
    max_freq = 0.1  # 恢复高频峰值
    prominence = 0.01
    female_mean_amplitude, female_peaks, female_peak_freqs, female_peak_heights, female_periods, freqs_female_filtered = analyze_periodicity(fft_female, freqs_female, min_freq, max_freq, prominence)
    male_mean_amplitude, male_peaks, male_peak_freqs, male_peak_heights, male_periods, freqs_male_filtered = analyze_periodicity(fft_male, freqs_male, min_freq, max_freq, prominence)

    # 打印周期性分析结果
    print("Female Spectra Periodicity Analysis (After Preprocessing):")
    if len(female_peak_freqs) > 0:
        for i, (freq, height, period) in enumerate(zip(female_peak_freqs, female_peak_heights, female_periods)):
            print(f"Peak {i+1}: Frequency = {freq:.4f} (1/nm), Amplitude = {height:.4f}, Period = {period:.4f} (nm)")
    else:
        print("No significant peaks found. The female spectra may not be periodic.")

    print("\nMale Spectra Periodicity Analysis (After Preprocessing):")
    if len(male_peak_freqs) > 0:
        for i, (freq, height, period) in enumerate(zip(male_peak_freqs, male_peak_heights, male_periods)):
            print(f"Peak {i+1}: Frequency = {freq:.4f} (1/nm), Amplitude = {height:.4f}, Period = {period:.4f} (nm)")
    else:
        print("No significant peaks found. The male spectra may not be periodic.")

    # 绘制频域均值图
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_female_filtered, female_mean_amplitude, 'r-', label='Female Mean', linewidth=2)
    plt.plot(freqs_male_filtered, male_mean_amplitude, 'g-', label='Male Mean', linewidth=2)

    for peak, height in zip(female_peaks, female_peak_heights):
        plt.plot(freqs_female_filtered[peak], height, 'r*', markersize=10)
        plt.text(freqs_female_filtered[peak], height + 0.05 * max(female_mean_amplitude), f'{freqs_female_filtered[peak]:.4f}', color='red', fontsize=8, ha='center', va='bottom')

    for peak, height in zip(male_peaks, male_peak_heights):
        plt.plot(freqs_male_filtered[peak], height, 'g*', markersize=10)
        plt.text(freqs_male_filtered[peak], height + 0.05 * max(male_mean_amplitude), f'{freqs_male_filtered[peak]:.4f}', color='green', fontsize=8, ha='center', va='bottom')

    plt.title('Mean FFT Amplitude (Frequency Domain) with Peaks - Preprocessed Spectra')
    plt.xlabel('Frequency (1/nm)')
    plt.ylabel('Mean Amplitude')
    plt.legend()
    plt.xlim(min_freq, max_freq)
    plt.ylim(0, max(max(female_mean_amplitude), max(male_mean_amplitude)) * 1.2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.show()

    # 特征提取
    if len(female_peak_freqs) > 0 and len(male_peak_freqs) > 0:
        # 提取所有峰值频率的特征
        female_features_temp = np.zeros((female_processed.shape[0], len(female_peak_freqs)))
        male_features_temp = np.zeros((male_processed.shape[0], len(male_peak_freqs)))

        for i in range(female_processed.shape[0]):
            freqs, fft = apply_fft(female_processed[i:i+1], wavelengths)
            fft = fft[:, (freqs >= min_freq) & (freqs <= max_freq)]
            freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
            for j, peak_freq in enumerate(female_peak_freqs):
                idx = np.argmin(np.abs(freqs - peak_freq))
                female_features_temp[i, j] = fft[0, idx]

        for i in range(male_processed.shape[0]):
            freqs, fft = apply_fft(male_processed[i:i+1], wavelengths)
            fft = fft[:, (freqs >= min_freq) & (freqs <= max_freq)]
            freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
            for j, peak_freq in enumerate(male_peak_freqs):
                idx = np.argmin(np.abs(freqs - peak_freq))
                male_features_temp[i, j] = fft[0, idx]

        # 选择差异较大的峰值频率（基于 t 检验和相对差异）
        selected_freqs = []
        for j, freq in enumerate(female_peak_freqs):
            if freq in male_peak_freqs:
                female_height = np.mean(female_features_temp[:, j])
                male_height = np.mean(male_features_temp[:, j])
                rel_diff = abs(female_height - male_height) / min(female_height, male_height)
                t_stat, p_value = ttest_ind(female_features_temp[:, j], male_features_temp[:, j])
                if p_value < 0.01 and rel_diff > 0.2:  # 放宽 t 检验，收紧相对差异
                    selected_freqs.append(freq)

        if len(selected_freqs) > 0:
            female_features = np.zeros((female_processed.shape[0], len(selected_freqs)))
            male_features = np.zeros((male_processed.shape[0], len(selected_freqs)))

            for i in range(female_processed.shape[0]):
                freqs, fft = apply_fft(female_processed[i:i+1], wavelengths)
                fft = fft[:, (freqs >= min_freq) & (freqs <= max_freq)]
                freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
                for j, peak_freq in enumerate(selected_freqs):
                    idx = np.argmin(np.abs(freqs - peak_freq))
                    female_features[i, j] = fft[0, idx]

            for i in range(male_processed.shape[0]):
                freqs, fft = apply_fft(male_processed[i:i+1], wavelengths)
                fft = fft[:, (freqs >= min_freq) & (freqs <= max_freq)]
                freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
                for j, peak_freq in enumerate(selected_freqs):
                    idx = np.argmin(np.abs(freqs - peak_freq))
                    male_features[i, j] = fft[0, idx]

            print("\nSelected Frequencies for Features (based on t-test and relative difference):", selected_freqs)
            print("Female Features Shape:", female_features.shape)
            print("Male Features Shape:", male_features.shape)

            # 可视化特征分布
            for j, freq in enumerate(selected_freqs):
                plt.figure(figsize=(6, 4))
                plt.hist(female_features[:, j], bins=30, alpha=0.5, label='Female', color='red')
                plt.hist(male_features[:, j], bins=30, alpha=0.5, label='Male', color='green')
                plt.title(f'Feature Distribution at Frequency {freq:.4f} (1/nm)')
                plt.xlabel('Amplitude')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                plt.show()
        else:
            print("\nNo peaks with significant differences, using SelectKBest for feature selection.")
            # 使用 SelectKBest 选择特征
            features = np.vstack((female_features_temp, male_features_temp))
            labels = np.hstack((np.zeros(female_features_temp.shape[0]), np.ones(male_features_temp.shape[0])))
            selector = SelectKBest(score_func=f_classif, k=3)
            features_selected = selector.fit_transform(features, labels)
            selected_indices = selector.get_support()
            selected_freqs = female_peak_freqs[selected_indices]
            female_features = features_selected[:female_features_temp.shape[0], :]
            male_features = features_selected[female_features_temp.shape[0]:, :]
            print("\nSelected Frequencies for Features (based on SelectKBest):", selected_freqs)
            print("Female Features Shape:", female_features.shape)
            print("Male Features Shape:", male_features.shape)

    else:
        print("\nNo peaks detected, using full FFT amplitude with feature selection.")
        freqs_female, fft_female = apply_fft(female_processed, wavelengths)
        freqs_male, fft_male = apply_fft(male_processed, wavelengths)
        female_features = fft_female[:, (freqs_female >= min_freq) & (freqs_female <= max_freq)]
        male_features = fft_male[:, (freqs_male >= min_freq) & (freqs_male <= max_freq)]
        freqs_filtered = freqs_female[(freqs_female >= min_freq) & (freqs_female <= max_freq)]

        # 使用 SelectKBest 选择特征
        features = np.vstack((female_features, male_features))
        labels = np.hstack((np.zeros(female_features.shape[0]), np.ones(male_features.shape[0])))
        selector = SelectKBest(score_func=f_classif, k=5)
        features_selected = selector.fit_transform(features, labels)
        selected_indices = selector.get_support()
        selected_freqs = freqs_filtered[selected_indices]
        female_features = features_selected[:female_features.shape[0], :]
        male_features = features_selected[female_features.shape[0]:, :]
        print("\nSelected Frequencies for Features (based on SelectKBest):", selected_freqs)
        print("Female Features Shape:", female_features.shape)
        print("Male Features Shape:", male_features.shape)