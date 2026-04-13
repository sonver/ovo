import numpy as np
import pandas as pd
import os
import glob
import re
from scipy.stats import zscore
import matplotlib.pyplot as plt
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_spectrum(file_path):
    """
    加载 .txt 文件中的高光谱数据。
    :param file_path: .txt 文件路径
    :return: 1D 光谱数组
    """
    try:
        data = np.loadtxt(file_path, delimiter=' ')
        data = data.flatten() if data.ndim > 1 else data
        if data.size < 3:  # 确保足以计算二阶差分
            logger.warning(f"文件 {file_path} 数据点不足: {data.size}")
            return None
        if not np.all(np.isfinite(data)):
            logger.warning(f"文件 {file_path} 包含 NaN 或无穷值")
            return None
        return data
    except Exception as e:
        logger.error(f"加载 {file_path} 失败: {e}")
        return None

def calculate_smoothness_index(data, band_range=None):
    """
    计算平滑指数（二阶差分）。
    :param data: 1D 光谱数据 (n_bands,)
    :param band_range: 波段范围，例如 (start, end)
    :return: 平滑指数
    """
    try:
        if band_range:
            start, end = band_range
            data = data[start:end]
        if len(data) < 3:
            logger.warning(f"波段范围 {band_range} 数据点不足: {len(data)}")
            return np.nan
        diff2 = np.diff(data, n=2) ** 2
        return np.sum(diff2)
    except Exception as e:
        logger.error(f"计算平滑指数失败: {e}")
        return np.nan

def calculate_snr(spectra, band_range=None):
    """
    计算一盘样本的信噪比（SNR）。
    :param spectra: 样本光谱列表 (n_samples, n_bands)
    :param band_range: 波段范围，例如 (start, end)
    :return: SNR 值
    """
    try:
        if not spectra:
            return np.nan
        spectra = np.array(spectra)
        if band_range:
            start, end = band_range
            spectra = spectra[:, start:end]
        # 计算每样本的 SNR，然后取平均
        snrs = []
        for spectrum in spectra:
            signal = np.mean(spectrum)  # 信号：光谱均值
            noise = np.std(np.diff(spectrum, n=2))  # 噪声：二阶差分的标准差
            noise = max(noise, 1e-10)  # 避免除以零
            snr = signal / noise
            if np.isfinite(snr):
                snrs.append(snr)
        return np.mean(snrs) if snrs else np.nan
    except Exception as e:
        logger.error(f"计算 SNR 失败: {e}")
        return np.nan

def detect_noisy_samples(data_dir, band_range=None, z_threshold=3, output_dir='noise_analysis'):
    """
    检测噪声样本，分组（噪声很大/一般/小），分析 High/Medium 样本的盘号分布，计算每盘 SNR。
    在筛选出 SNR < 10 的盘号后，删除对应文件。
    :param data_dir: 数据目录 (包含 female 和 male 子目录)
    :param band_range: 波段范围
    :param z_threshold: Z 分数阈值（用于标记异常）
    :param output_dir: 输出目录
    :return: 结果 DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集样本信息和光谱数据（用于 SNR 计算）
    results = []
    invalid_files = []
    pallet_spectra = {}  # 按盘号存储光谱数据
    for gender in ['female', 'male']:
        dir_path = os.path.join(data_dir, gender)
        for file_path in glob.glob(os.path.join(dir_path, 'egg*.txt')):
            filename = os.path.basename(file_path)
            # 解析文件名：egg{PalletCode}-{Index}.txt
            match = re.match(r'egg([0-9A-Za-z]+)-(\d+)\.txt', filename)
            if not match:
                logger.warning(f"跳过无效文件名: {filename}")
                invalid_files.append(filename)
                continue
            pallet, index = map(str, match.groups())

            # 加载光谱数据
            spectrum = load_spectrum(file_path)
            if spectrum is None:
                invalid_files.append(filename)
                continue

            # 计算平滑指数
            smoothness = calculate_smoothness_index(spectrum, band_range)
            if np.isnan(smoothness):
                logger.warning(f"样本 {filename} 平滑指数无效，跳过")
                invalid_files.append(filename)
                continue

            # 存储光谱数据（用于 SNR 计算）
            if pallet not in pallet_spectra:
                pallet_spectra[pallet] = []
            pallet_spectra[pallet].append(spectrum)

            results.append({
                'Filename': filename,
                'PalletCode': pallet,
                'Index': index,
                'SmoothnessIndex': smoothness,
                'Gender': gender.capitalize()
            })

    if invalid_files:
        logger.info(
            f"共跳过 {len(invalid_files)} 个无效文件: {invalid_files[:5]}{'...' if len(invalid_files) > 5 else ''}")

    # 转换为 DataFrame
    df_results = pd.DataFrame(results)
    logger.info(f"加载 {len(df_results)} 个有效样本")
    if df_results.empty:
        logger.error("无有效样本数据")
        return None

    # 清理平滑指数中的无效值
    smoothness_indices = df_results['SmoothnessIndex'].values
    if np.any(np.isnan(smoothness_indices)) or np.any(np.isinf(smoothness_indices)):
        logger.warning("平滑指数包含 NaN 或无穷值，清理数据")
        df_results = df_results[np.isfinite(df_results['SmoothnessIndex'])]
        smoothness_indices = df_results['SmoothnessIndex'].values
        logger.info(f"清理后剩余 {len(df_results)} 个有效样本")
        if len(df_results) < 2:
            logger.error("有效样本少于 2 个，无法计算 Z 分数")
            df_results['ZScore'] = np.nan
            df_results['IsNoisy'] = False
            df_results['NoiseLevel'] = 'Unknown'
            return df_results

    # 异常检测
    try:
        z_scores = zscore(smoothness_indices, nan_policy='omit')
        df_results['ZScore'] = z_scores
        df_results['IsNoisy'] = np.abs(z_scores) > z_threshold
        logger.info("Z 分数和 IsNoisy 列生成成功")
    except Exception as e:
        logger.error(f"计算 Z 分数或生成 IsNoisy 失败: {e}")
        traceback.print_exc()
        df_results['ZScore'] = np.nan
        df_results['IsNoisy'] = False
        logger.warning("设置 IsNoisy 为默认值 False")

    # 噪声水平分组
    df_results['NoiseLevel'] = np.where(np.isnan(z_scores), 'Unknown',
                                        np.where(z_scores > 2, 'High',
                                                 np.where(z_scores > 0.5, 'Medium', 'Low')))
    logger.info(f"噪声水平分组完成: {df_results['NoiseLevel'].value_counts().to_dict()}")

    # 按噪声水平分组并排序
    grouped_dfs = []
    for level in ['High', 'Medium', 'Low', 'Unknown']:
        group_df = df_results[df_results['NoiseLevel'] == level][[
            'Filename', 'PalletCode', 'Index', 'SmoothnessIndex', 'ZScore', 'Gender', 'NoiseLevel'
        ]].sort_values(by='SmoothnessIndex', ascending=False)
        if not group_df.empty:
            grouped_dfs.append(group_df)
            print(f"\n{level} 噪声水平样本表格:")
            print(group_df.to_string(index=False, float_format='%.4f'))

    # 合并排序后的 DataFrame
    df_results = pd.concat(grouped_dfs, ignore_index=True) if grouped_dfs else df_results

    # 保存主表格
    output_file = os.path.join(output_dir, 'noisy_samples_table.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"主表格已保存至: {output_file}")

    # 计算每盘 SNR
    snr_results = []
    for pallet, spectra in pallet_spectra.items():
        snr = calculate_snr(spectra, band_range)
        if not np.isnan(snr):
            snr_results.append({'PalletCode': pallet, 'SNR': snr})
    df_snr = pd.DataFrame(snr_results)
    if df_snr.empty:
        logger.error("无法计算任何盘的 SNR")
    else:
        # 筛选 SNR 小于 10 的盘号
        df_low_snr = df_snr[df_snr['SNR'] < 12]
        if not df_low_snr.empty:
            print("\n信噪比小于 12 的盘号:")
            print(df_low_snr.to_string(index=False, float_format='%.4f'))
            low_snr_file = os.path.join(output_dir, 'low_snr_pallets.csv')
            df_low_snr.to_csv(low_snr_file, index=False, encoding='utf-8-sig')
            logger.info(f"SNR 小于 12 的盘号已保存至: {low_snr_file}")

            # 删除 SNR 小于 10 的盘号对应的文件
            if deleteSNR:
                deleted_files_count = 0
                for pallet in df_low_snr['PalletCode']:
                    for gender in ['female', 'male']:
                        dir_path = os.path.join(data_dir, gender)
                        # 匹配文件名模式：egg{pallet}-*.txt
                        pattern = os.path.join(dir_path, f'egg{pallet}-*.txt')
                        files_to_delete = glob.glob(pattern)
                        for file_path in files_to_delete:
                            try:
                                os.remove(file_path)
                                logger.info(f"已删除文件: {file_path}")
                                deleted_files_count += 1
                            except Exception as e:
                                logger.warning(f"删除文件 {file_path} 失败: {e}")
                print(f"\n共删除 {deleted_files_count} 个文件（SNR 小于 12 的盘号）")
                logger.info(f"共删除 {deleted_files_count} 个文件（SNR 小于 12 的盘号）")
        else:
            print("\n未找到信噪比小于 12 的盘号")
            logger.info("未找到 SNR 小于 12 的盘号")

        # 按 SNR 升序排序（用于第一个图）
        df_snr_by_snr = df_snr.sort_values(by='SNR', ascending=True)
        logger.info(f"计算 {len(df_snr)} 个盘的 SNR，范围: {df_snr['SNR'].min():.4f} 到 {df_snr['SNR'].max():.4f}")

        # 保存 SNR 表格
        snr_file = os.path.join(output_dir, 'pallet_snr.csv')
        df_snr_by_snr.to_csv(snr_file, index=False, encoding='utf-8-sig')
        logger.info(f"SNR 表格已保存至: {snr_file}")

        # 第一个图：按 SNR 升序的折线图
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(df_snr_by_snr)), df_snr_by_snr['SNR'], marker='o', color='blue', label='SNR')
        for i, row in df_snr_by_snr.iterrows():
            if i % 5 == 0:  # 每隔 5 个点标注
                plt.text(i, row['SNR'], str(row['PalletCode']), fontsize=8, ha='center', va='bottom')
        # 调整 X 轴刻度，每隔 10 个盘号显示
        ticks = np.arange(0, len(df_snr_by_snr), 10)
        tick_labels = [str(df_snr_by_snr['PalletCode'].iloc[i]) for i in ticks]
        plt.xticks(ticks, tick_labels, rotation=45)
        plt.xlabel('盘号 (按 SNR 升序排列)')
        plt.ylabel('信噪比 (SNR)')
        plt.title('各盘信噪比 (SNR) 分布')
        plt.legend()
        plt.grid(True, which='major', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pallet_snr_plot.png'))
        plt.close()

        # 第二个图：按 PalletCode 升序的折线图
        df_snr_by_pallet = df_snr.sort_values(by='PalletCode', ascending=True)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(df_snr_by_pallet)), df_snr_by_pallet['SNR'], marker='o', color='blue', label='SNR')
        for i, row in df_snr_by_pallet.iterrows():
            if i % 5 == 0:  # 每隔 5 个点标注
                plt.text(i, row['SNR'], str(row['PalletCode']), fontsize=8, ha='center', va='bottom')
        # 调整 X 轴刻度，每隔 10 个盘号显示
        ticks = np.arange(0, len(df_snr_by_pallet), 10)
        tick_labels = [str(df_snr_by_pallet['PalletCode'].iloc[i]) for i in ticks]
        plt.xticks(ticks, tick_labels, rotation=45)
        plt.xlabel('盘号 (按盘号升序排列)')
        plt.ylabel('信噪比 (SNR)')
        plt.title('各盘信噪比 (SNR) 分布 (按盘号升序)')
        plt.legend()
        plt.grid(True, which='major', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pallet_snr_sorted_by_pallet.png'))
        plt.close()

    # 分析 High 和 Medium 样本的盘号分布
    high_medium_df = df_results[df_results['NoiseLevel'].isin(['High', 'Medium'])]
    if not high_medium_df.empty:
        # 按盘号统计 High 和 Medium 样本数
        pallet_dist = high_medium_df.groupby(['PalletCode', 'NoiseLevel']).size().unstack(fill_value=0)
        pallet_dist = pallet_dist.reindex(columns=['High', 'Medium'], fill_value=0)
        pallet_dist['TotalCount'] = pallet_dist['High'] + pallet_dist['Medium']
        pallet_dist = pallet_dist.sort_values(by='TotalCount', ascending=False).reset_index()

        # 打印盘号分布表格
        print("\nHigh 和 Medium 噪声样本的盘号分布:")
        print(pallet_dist.to_string(index=False, float_format='%.0f'))

        # 保存盘号分布表格
        dist_file = os.path.join(output_dir, 'pallet_noise_distribution.csv')
        pallet_dist.to_csv(dist_file, index=False, encoding='utf-8-sig')
        logger.info(f"盘号分布表格已保存至: {dist_file}")

        # 可视化盘号分布（Top 10）
        top_n = 10
        top_pallets = pallet_dist.head(top_n)
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        index = np.arange(len(top_pallets))
        plt.bar(index - bar_width, top_pallets['High'], bar_width, label='High 样本', color='red')
        plt.bar(index, top_pallets['Medium'], bar_width, label='Medium 样本', color='orange')
        plt.bar(index + bar_width, top_pallets['TotalCount'], bar_width, label='High + Medium 样本', color='blue')
        plt.xlabel('盘号')
        plt.ylabel('样本数')
        plt.title(f'High、Medium 和 High+Medium 噪声样本的盘号分布 (Top {top_n})')
        plt.xticks(index, top_pallets['PalletCode'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pallet_noise_distribution.png'))
        plt.close()
    else:
        logger.warning("\n无 High 或 Medium 噪声样本，无法生成盘号分布")

    # 打印噪声异常样本（Z 分数 > z_threshold）
    if 'IsNoisy' in df_results.columns:
        noisy_df = df_results[df_results['IsNoisy']]
        if not noisy_df.empty:
            print("\n噪声异常样本表格 (Z 分数 > {}):".format(z_threshold))
            print(noisy_df[['Filename', 'PalletCode', 'Index', 'SmoothnessIndex', 'ZScore', 'Gender',
                            'NoiseLevel']].to_string(index=False, float_format='%.4f'))
        else:
            print("未检测到噪声异常样本 (Z 分数 > {})".format(z_threshold))
    else:
        logger.error("无法生成噪声异常样本表格：'IsNoisy' 列缺失")

    # 可视化噪声样本（仅 High 噪声水平，最多 5 个）
    high_noise_df = df_results[df_results['NoiseLevel'] == 'High'].head(5)
    for _, row in high_noise_df.iterrows():
        file_path = os.path.join(data_dir, row['Gender'].lower(), row['Filename'])
        spectrum = load_spectrum(file_path)
        if spectrum is None:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(spectrum, label=f'样本 {row["PalletCode"]}-{row["Index"]} (平滑指数: {row["SmoothnessIndex"]:.4f})')
        if band_range:
            plt.axvspan(band_range[0], band_range[1], color='gray', alpha=0.2, label='分析波段范围')
        plt.xlabel('波段')
        plt.ylabel('强度')
        plt.title(f'噪声样本光谱曲线 (盘号: {row["PalletCode"]}, 序号: {row["Index"]}, 噪声水平: {row["NoiseLevel"]})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'noisy_sample_{row["PalletCode"]}_{row["Index"]}.png'))
        plt.close()

    # 可视化平滑指数分布
    plt.figure(figsize=(8, 5))
    plt.hist(smoothness_indices, bins=50, alpha=0.7, label='平滑指数分布')
    plt.axvline(np.mean(smoothness_indices) + z_threshold * np.std(smoothness_indices), color='r', linestyle='--',
                label=f'异常阈值 (Z={z_threshold})')
    plt.axvline(np.mean(smoothness_indices) + 2 * np.std(smoothness_indices), color='orange', linestyle='--',
                label='High 阈值 (Z=2)')
    plt.axvline(np.mean(smoothness_indices) + 0.5 * np.std(smoothness_indices), color='green', linestyle='--',
                label='Medium 阈值 (Z=0.5)')
    plt.xlabel('平滑指数')
    plt.ylabel('样本数')
    plt.title('平滑指数分布与噪声水平阈值')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'smoothness_distribution.png'))
    plt.close()

    return df_results

def main():

    # data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187"
    # output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\results\no34no187"

    data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\xiaoming_new_0804"
    output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\DDD"

    band_range = (0, 150)  # 示例：(50, 100) 或 None（全部波段）

    if not os.path.exists(data_dir):
        logger.error(f"数据目录 {data_dir} 不存在")
        return

    try:
        df_results = detect_noisy_samples(data_dir, band_range=band_range, z_threshold=3, output_dir=output_dir)
        if df_results is not None:
            logger.info(f"噪声样本检测完成，表格保存至 {output_dir}")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    deleteSNR = False
    main()