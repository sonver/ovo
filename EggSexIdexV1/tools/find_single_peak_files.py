
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ========== 在这里填写路径 ==========
# 输入：待检测的 txt 光谱文件夹
INPUT_DIR = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000\female"

# 输出：单峰文件复制到这个文件夹（如果不想复制，设为 None）
OUTPUT_SINGLE_DIR = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000\single_peak_female"

# 绘图输出文件夹（每个文件生成一张光谱曲线图，带红点标记峰；不想保存设为 None）
PLOT_DIR = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000\plots_female"

# 结果保存为 CSV 文件（包含文件名、峰个数、峰位置等；不想保存设为 None）
CSV_OUT = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000\female_single_results.csv"

# 峰检测与判断参数（可调）
PROMINENCE_REL = 0.05   # 峰显著性阈值（相对振幅范围），用于 find_peaks
DISTANCE = 5            # 峰之间最小间距（波段数）
MONO_FRAC = 0.85        # 右侧差分负值所占比例阈值（>= 该值视为“绝大部分单调递减”）
POOLING = "median"      # 如果 txt 中是多行数据，可选聚合："median"/"mean"/"first"
# ============================================

def load_spectrum_from_txt(path, pooling='median'):
    """读取 txt 文件并得到 1D 光谱向量（不做平滑）"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        spec = data.astype(float)
    elif data.ndim == 2:
        if pooling == 'median':
            spec = np.median(data, axis=0).astype(float)
        elif pooling == 'mean':
            spec = np.mean(data, axis=0).astype(float)
        else:
            spec = data[0].astype(float)
    else:
        raise ValueError(f"Unsupported data shape {data.shape} in file {path}")
    return spec

def detect_peaks_raw(spec, prominence_rel=0.05, distance=5):
    """在原始光谱上检测峰（局部最大值）"""
    if spec.size == 0:
        return np.array([], dtype=int), {}
    amp_range = spec.max() - spec.min()
    prom = 0.0 if amp_range == 0 else prominence_rel * amp_range
    peaks, props = find_peaks(spec, prominence=prom, distance=distance)
    return peaks, props

def first_peak_then_right_decreasing_no_second(spec, peaks, props, mono_frac=0.85):
    """
    规则：
    - 需要有至少一个峰 -> 取第一个峰 peaks[0]
    - 峰右侧（包含峰后面的部分）差分中，至少 mono_frac 比例为负值（下降）
    - 峰右侧没有第二个显著峰（即在 peaks 中没有 p > first_peak）
    返回 (flag, first_peak_idx, frac_decreasing, n_peaks_right)
    """
    if len(peaks) == 0:
        return False, None, 0.0, 0

    first_peak = int(peaks[0])

    # 右侧序列（从第一个峰的下一个元素开始判断趋势）
    right = spec[first_peak+1:]  # 不包含峰点本身，专注于“后面有没有上升形成第二峰”
    if right.size == 0:
        # 峰在末尾，视为“之后没有第二峰”，把它标记为 True（右侧没有上升）
        return True, first_peak, 1.0, 0

    diffs = np.diff(np.concatenate(([spec[first_peak]], right)))  # 从峰到最后的差分（包含首差）
    # 判定下降（我们认为负差表示下降，允许非常小的数值波动用 tol）
    tol = 1e-8
    dec_count = np.sum(diffs < -tol)
    frac_dec = dec_count / diffs.size

    # 检查 peaks 列表中是否存在在 first_peak 右侧的峰
    peaks_right = [int(p) for p in peaks if int(p) > first_peak]
    n_peaks_right = len(peaks_right)

    flag = (frac_dec >= mono_frac) and (n_peaks_right == 0)
    return bool(flag), first_peak, float(frac_dec), n_peaks_right

def plot_and_save(spec, first_peak_idx, out_png_path, title=None):
    x = np.arange(spec.size)
    plt.figure(figsize=(10, 4))
    plt.plot(x, spec, label='原始光谱')
    if first_peak_idx is not None:
        plt.plot(first_peak_idx, spec[first_peak_idx], 'ro', label='first peak')
    plt.xlabel('波段索引')
    plt.ylabel('强度')
    plt.title(title or os.path.basename(out_png_path))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

def process_folder():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    if OUTPUT_SINGLE_DIR:
        os.makedirs(OUTPUT_SINGLE_DIR, exist_ok=True)
    if PLOT_DIR:
        os.makedirs(PLOT_DIR, exist_ok=True)

    records = []
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.txt')])

    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            spec = load_spectrum_from_txt(fpath, pooling=POOLING)
        except Exception as e:
            print(f"[跳过] 无法读取 {fname}: {e}")
            continue

        peaks, props = detect_peaks_raw(spec, prominence_rel=PROMINENCE_REL, distance=DISTANCE)
        flag, first_peak_idx, frac_dec, n_peaks_right = first_peak_then_right_decreasing_no_second(
            spec, peaks, props, mono_frac=MONO_FRAC)

        rec = {
            'file': fname,
            'n_peaks': len(peaks),
            'first_peak_idx': int(first_peak_idx) if first_peak_idx is not None else '',
            'frac_right_decreasing': round(frac_dec, 4),
            'n_peaks_right': int(n_peaks_right),
            'match_rule': int(flag)
        }
        records.append(rec)

        # 复制满足条件的
        if flag and OUTPUT_SINGLE_DIR:
            shutil.copy(fpath, os.path.join(OUTPUT_SINGLE_DIR, fname))

        # 绘图（标记第一个峰）
        if PLOT_DIR:
            png_name = os.path.splitext(fname)[0] + '.png'
            png_path = os.path.join(PLOT_DIR, png_name)
            plot_and_save(spec, first_peak_idx, png_path, title=fname)

    # 保存 CSV
    if CSV_OUT:
        import csv
        with open(CSV_OUT, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=['file', 'n_peaks', 'first_peak_idx',
                                                   'frac_right_decreasing', 'n_peaks_right', 'match_rule'])
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    total = len(records)
    matched = sum(r['match_rule'] for r in records)
    print(f"总文件数: {total}, 满足规则文件数: {matched}")
    return records

if __name__ == "__main__":
    recs = process_folder()
