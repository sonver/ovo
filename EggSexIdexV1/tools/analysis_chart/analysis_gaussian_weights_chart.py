import os
import numpy as np
from matplotlib import pyplot as plt
from tools.tools import WAVELENGTH_INFO, gaussian_weights


def getFandMHsiData(num_f, f_start, num_m, m_start):
    # f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\1month\female'
    # m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\1month\male'
    #
    # f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187\female'
    # m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\no34no187\male'

    # f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh\Dataset\4month\female'
    # m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh\Dataset\4month\male'

    # f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\6month\female'
    # m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\6month\male'

    # f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\6month_step\female'
    # m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\Dataset\6month_step\male'

    f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\huayu\female'
    m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\huayu\male'

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


if __name__ == '__main__':

    num_f, f_start = 500, 0
    num_m, m_start = 500, 0
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # 绘制公、母的光谱曲线图
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
    num_bands = len(wavelengths)  # 150

    # 计算每组的平均光谱
    female_mean = np.mean(f_data_list.reshape(f_data_list.shape[0], -1, num_bands), axis=1)  # (111, 150)
    male_mean = np.mean(m_data_list.reshape(m_data_list.shape[0], -1, num_bands), axis=1)  # (210, 150)

    window_size = 7  # 平滑窗口大小
    polyorder = 2  # 多项式阶数
    # 对每组光谱进行预处理
    female_processed = []
    male_processed = []

    # 第一种
    # sigma = 1.0  # 调整标准差
    sigma = 1.0
    for img in f_data_list:
        result = []

        for col in range(150):
            column_data = img[:,col] # 提取每一列数据 (400,);
            # sigma = np.std(column_data) * 0.5 + 0.5  # 调整sigma逻辑（根据SNV后的标准差）
            weights = gaussian_weights(column_data, sigma=sigma)  # 计算高斯权重
            if np.sum(weights) == 0:
                weighted_sum = 0
            else:
                weighted_sum = np.sum(column_data * weights) / np.sum(weights)  # 加权平均
            result.append(weighted_sum)  # 将结果添加到列表
        female_processed.append(np.array(result).reshape(1, -1))

    for img in m_data_list:
        result = []

        for col in range(150):
            column_data = img[:,col] # 提取每一列数据 (400,);
            weights = gaussian_weights(column_data, sigma=sigma)  # 计算高斯权重
            if np.sum(weights) == 0:
                weighted_sum = 0
            else:
                weighted_sum = np.sum(column_data * weights) / np.sum(weights)  # 加权平均
            result.append(weighted_sum)  # 将结果添加到列表
        male_processed.append(np.array(result).reshape(1, -1))


    female_processed = np.array(female_processed)
    male_processed = np.array(male_processed)

    # 绘制所有曲线
    plt.figure(figsize=(10, 6))
    for spec in female_processed:
        spec = spec.reshape(150,-1)
        plt.plot(wavelengths, spec, 'r-', alpha=1, linewidth=0.8,label='Female' if spec is female_processed[0] else "")
    for spec in male_processed:
        spec = spec.reshape(150, -1)
        plt.plot(wavelengths, spec, 'g-', alpha=0.3, linewidth=0.8, label='Male' if spec is male_processed[0] else "")


    plt.title('Gaussian Weights')
    plt.xlabel('Wavelength (nm)' + str(wavelengths[0]) + '-' + str(wavelengths[-1]))
    plt.ylabel('Preprocessed Transmittance')
    plt.legend()
    plt.xlim(wavelengths[0], wavelengths[-1])

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
