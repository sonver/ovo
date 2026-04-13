import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tabulate import tabulate

# 波长数组（389.5 nm 到 1005.66 nm，300个波段）
WAVELENGTH_INFO = np.array([
    389.5, 391.71, 393.92, 396.13, 398.34, 400.55, 402.75, 404.95, 407.15, 409.34,
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
    986.77, 988.86, 990.96, 993.05, 995.15, 997.25, 999.35, 1001.45, 1003.56, 1005.66
])

def load_spectral_data(file_path):
    """加载 .txt 文件的高光谱数据，形状为 (400, 150)"""
    try:
        data = np.loadtxt(file_path)
        if data.shape != (400, 150):
            print(f"警告: {file_path} 数据形状为 {data.shape}，预期 (400, 150)")
        return data
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None

def merge_consecutive_segments(segments):
    """合并相邻或重叠的连续波段"""
    if not segments:
        return []
    merged = []
    current = segments[0].copy()
    for next_segment in segments[1:]:
        if float(next_segment['start_wavelength']) <= float(current['end_wavelength']) + 2.21:  # 波长间隔 ~2.21 nm
            current['end_wavelength'] = next_segment['end_wavelength']
            current['mean'] = f"{(float(current['mean']) + float(next_segment['mean'])) / 2:.2f}"
            current['variance'] = f"{min(float(current['variance']), float(next_segment['variance'])):.2f}"
        else:
            merged.append(current)
            current = next_segment.copy()
    merged.append(current)
    return merged

def plot_all_spectral_curves(data_dir, wavelengths=WAVELENGTH_INFO[100:250], max_files=None,
                             min_consecutive_bands=10, value_threshold=4000, variance_threshold=100,
                             focus_gender_range=False):
    """绘制平均光谱曲线、按标准差排序的曲线和独立按均值排序的均值曲线，并筛选连续波段值 > 4000 且方差低的曲线"""
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if max_files is not None:
        txt_files = txt_files[:max_files]
    if not txt_files:
        print(f"目录 {data_dir} 中未找到 .txt 文件")
        return

    # 验证波长数组
    if len(wavelengths) != 150:
        print(f"警告: 波长数量 {len(wavelengths)} 不匹配数据波段数 150，使用默认索引")
        wavelengths = np.arange(150)
    else:
        print(f"使用波长范围: {wavelengths[0]:.2f} nm - {wavelengths[-1]:.2f} nm")

    # 性别波段范围 (749–861 nm)
    gender_range = (749, 861) if focus_gender_range else None
    if gender_range:
        gender_indices = np.where((wavelengths >= gender_range[0]) & (wavelengths <= gender_range[1]))[0]
        print(f"聚焦性别波段: {gender_range[0]}–{gender_range[1]} nm (索引 {gender_indices[0]}–{gender_indices[-1]})")
    else:
        gender_indices = range(len(wavelengths) - min_consecutive_bands + 1)

    # 初始化数据存储
    avg_spectra = []
    file_names = []
    max_stds = []
    data_means = []
    high_value_flat_segments = []  # 存储连续波段值 > 4000 且方差低的记录

    # 加载数据并计算均值、标准差和连续波段
    for file_path in txt_files:
        data = load_spectral_data(file_path)
        if data is None:
            continue

        # 计算平均光谱、均值和标准差
        avg_spectrum = np.mean(data, axis=0)
        data_mean = data.mean()
        pixel_std = np.std(data, axis=0)
        max_std = pixel_std.max()

        avg_spectra.append(avg_spectrum)
        file_names.append(os.path.basename(file_path))
        max_stds.append(max_std)
        data_means.append(data_mean)

        # 日志输出
        print(f"{file_names[-1]}: Data Mean={data_mean:.4f}, Max Std={max_std:.6f}")
        if max_std < 1e-6:
            print(f"警告: {file_names[-1]} 数据变异性过低，最大像素标准差: {max_std:.6f}")

        # 筛选连续波段值 > 4000 且方差低
        file_segments = []
        for start_idx in gender_indices:
            window = avg_spectrum[start_idx:start_idx + min_consecutive_bands]
            if len(window) == min_consecutive_bands and np.all(window > value_threshold):
                window_variance = np.var(window)
                if window_variance < variance_threshold:
                    start_wavelength = wavelengths[start_idx]
                    end_wavelength = wavelengths[start_idx + min_consecutive_bands - 1]
                    window_mean = np.mean(window)
                    file_segments.append({
                        'file_name': file_names[-1],
                        'start_wavelength': f"{start_wavelength:.2f}",
                        'end_wavelength': f"{end_wavelength:.2f}",
                        'mean': f"{window_mean:.2f}",
                        'variance': f"{window_variance:.2f}"
                    })
        # 合并相邻波段
        if file_segments:
            merged_segments = merge_consecutive_segments(file_segments)
            high_value_flat_segments.extend(merged_segments)

    # 检查文件间变异性
    if len(avg_spectra) > 1:
        avg_spectra = np.array(avg_spectra)
        spectra_std = np.std(avg_spectra, axis=0)
        print(f"文件间平均光谱最大标准差: {spectra_std.max():.6f}")
        if spectra_std.max() < 1e-6:
            print(f"警告: 文件间变异性过低，最大平均光谱标准差: {spectra_std.max():.6f}")

    # 输出均值和标准差范围
    if data_means and max_stds:
        print(f"Data Mean Range: min={min(data_means):.4f}, max={max(data_means):.4f}")
        print(f"Max Std Range: min={min(max_stds):.6f}, max={max(max_stds):.6f}")

    # 计算均值图的动态阈值
    if data_means:
        median = np.median(data_means)
        low_threshold_mean = 500  # 中位数 - 50%
        high_threshold_mean = 1500  # 中位数 + 50%
        print(f"均值图区间: Low Mean (0 - {low_threshold_mean:.4f}), Medium Mean ({low_threshold_mean:.4f} - {high_threshold_mean:.4f}), High Mean (>{high_threshold_mean:.4f})")
    else:
        low_threshold_mean = 0.8
        high_threshold_mean = 1.2
        print(f"警告: 无均值数据，使用默认阈值 Low Mean=0.8, High Mean=1.2")

    # 保存高值平坦段的 Markdown 表格
    output_dir = os.path.join(data_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    if high_value_flat_segments:
        headers = ['File Name', 'Start Wavelength (nm)', 'End Wavelength (nm)', 'Mean Value', 'Variance']
        table_data = [[entry['file_name'], entry['start_wavelength'], entry['end_wavelength'],
                       entry['mean'], entry['variance']] for entry in high_value_flat_segments]
        table_md = f"# High Value Flat Segments\n\n" + tabulate(table_data, headers=headers, tablefmt='github')
        output_path_table = os.path.join(output_dir, "high_value_flat_segments.md")
        with open(output_path_table, 'w', encoding='utf-8') as f:
            f.write(table_md)
        print(f"高值平坦段表格已保存至: {output_path_table}")
    else:
        print("未找到满足条件的连续波段（值 > 4000 且方差 < 100）")

    # 创建原始图（平均光谱和标准差）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # 子图1：平均光谱曲线
    ax1.set_title("Average Spectral Curves for All Files")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.grid(True)

    for avg_spectrum, file_name in zip(avg_spectra, file_names):
        ax1.plot(wavelengths, avg_spectrum, alpha=0.7, label=file_name)

    ax1.axvspan(749, 861, color='yellow', alpha=0.2, label="Gender Identification Range (749-861 nm)")
    if len(file_names) <= 20:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        print("文件过多，图例已省略以避免拥挤")
        ax1.legend(["Sample File", "Gender Identification Range"], fontsize=8)

    # 子图2：按标准差排序的曲线
    sorted_indices_std = np.argsort(max_stds)[::-1]
    sorted_max_stds = np.array(max_stds)[sorted_indices_std]
    sorted_file_names_std = np.array(file_names)[sorted_indices_std]

    ax2.plot(range(1, len(sorted_max_stds) + 1), sorted_max_stds, marker='o', color='blue', label='Max Std')
    ax2.set_title("Maximum Standard Deviation per File (Sorted)")
    ax2.set_xlabel("File Index")
    ax2.set_ylabel("Maximum Standard Deviation")
    ax2.grid(True)

    # 固定标准差区间
    low_threshold = 200
    high_threshold = 400
    ax2.axhspan(0, low_threshold, facecolor='green', alpha=0.2, label=f'Low Std (0 - {low_threshold})')
    ax2.axhspan(low_threshold, high_threshold, facecolor='yellow', alpha=0.2, label=f'Medium Std ({low_threshold} - {high_threshold})')
    ax2.axhspan(high_threshold, max(sorted_max_stds) * 1.1 if sorted_max_stds.size > 0 else high_threshold * 1.1,
                facecolor='red', alpha=0.2, label=f'High Std (>{high_threshold})')
    ax2.legend(fontsize=8)

    if sorted_max_stds.size > 0:
        ax2.set_ylim(0, max(sorted_max_stds) * 1.1)

    plt.tight_layout()

    # 保存标准差图
    output_path_std = os.path.join(output_dir, "spectral_std_plot.png")
    plt.savefig(output_path_std, bbox_inches='tight')
    print(f"标准差图像已保存至: {output_path_std}")

    plt.close(fig)  # 关闭标准差图

    # 创建独立均值图
    fig_mean, ax_mean = plt.subplots(figsize=(12, 5))

    # 按均值排序的曲线
    sorted_indices_mean = np.argsort(data_means)[::-1]
    sorted_data_means = np.array(data_means)[sorted_indices_mean]
    sorted_file_names_mean = np.array(file_names)[sorted_indices_mean]

    ax_mean.plot(range(1, len(sorted_data_means) + 1), sorted_data_means, marker='o', color='blue', label='Mean')
    ax_mean.set_title("Average Spectral Intensity per File (Sorted)")
    ax_mean.set_xlabel("File Index")
    ax_mean.set_ylabel("Average Spectral Intensity")
    ax_mean.grid(True)

    # 动态均值区间
    ax_mean.axhspan(0, low_threshold_mean, facecolor='green', alpha=0.2, label=f'Low Mean (0 - {low_threshold_mean:.4f})')
    ax_mean.axhspan(low_threshold_mean, high_threshold_mean, facecolor='yellow', alpha=0.2, label=f'Medium Mean ({low_threshold_mean:.4f} - {high_threshold_mean:.4f})')
    ax_mean.axhspan(high_threshold_mean, max(sorted_data_means) * 1.1 if sorted_data_means.size > 0 else high_threshold_mean * 1.1,
                    facecolor='red', alpha=0.2, label=f'High Mean (>{high_threshold_mean:.4f})')
    ax_mean.legend(fontsize=8)

    if sorted_data_means.size > 0:
        ax_mean.set_ylim(0, max(sorted_data_means) * 1.1)

    plt.tight_layout()

    # 保存均值图
    output_path_mean = os.path.join(output_dir, "spectral_mean_plot.png")
    plt.savefig(output_path_mean, bbox_inches='tight')
    print(f"均值图像已保存至: {output_path_mean}")

    plt.show()

if __name__ == "__main__":
    data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\seg_all_xm26"
    plot_all_spectral_curves(data_dir, max_files=2500, min_consecutive_bands=1, value_threshold=4000, variance_threshold=200, focus_gender_range=False)