import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from spectral import envi


matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 配置数据路径
data_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data\egg_raw_data_d"

# 获取并排序.spe文件（假设文件名为1.spe到10.spe）
spe_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.spe')],
                   key=lambda x: int(os.path.splitext(x)[0]))

all_spectra = []
wavelengths = None

for spe_file in spe_files[:50]:  # 处理前10个文件
    base_name = os.path.splitext(spe_file)[0]
    hdr_path = os.path.join(data_dir, base_name + '.hdr')
    spe_path = os.path.join(data_dir, spe_file)

    try:
        # 加载ENVI格式数据
        dataset = envi.open(hdr_path, image=spe_path)
        data = dataset.load()

        # 获取波长信息（仅第一个文件）
        if not wavelengths:
            wavelengths = dataset.metadata.get('wavelength', [])
            if wavelengths:
                try:
                    wavelengths = list(map(float, wavelengths))
                except ValueError:
                    wavelengths = None

        # 计算平均光谱
        if data.ndim == 3:
            avg_spectrum = data.mean(axis=(0, 1))
            all_spectra.append(avg_spectrum)
            print(f"成功处理：{spe_file}，形状：{data.shape}")
            print(f"D_avg 最大值为{np.max(data)}")
        else:
            print(f"跳过 {spe_file}：维度错误（需3维，实际{data.ndim}维）")

    except Exception as e:
        print(f"处理 {spe_file} 时出错：{str(e)}")
        continue

# 绘制频谱图
plt.figure(figsize=(14, 7), dpi=100)

for idx, spectrum in enumerate(all_spectra):
    if wavelengths and len(wavelengths) == len(spectrum):
        x = wavelengths
        xlabel = '波长 (nm)'
    else:
        x = np.arange(len(spectrum))
        xlabel = '波段号'

    plt.plot(x, spectrum, linewidth=1.5, alpha=0.8,
             label=f'图像 {idx + 1}')

plt.title('高光谱图像平均频谱对比', fontsize=14, pad=20)
plt.xlabel(xlabel, fontsize=12)
plt.ylabel('辐射强度', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()
plt.show()
