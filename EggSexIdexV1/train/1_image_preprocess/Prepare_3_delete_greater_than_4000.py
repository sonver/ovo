import os
import numpy as np

def getFandMHsiData(num_f, f_start, num_m, m_start):
    f_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\huayu_test\female'
    m_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\huayu_test\male'

    f_files = []
    m_files = []
    files_with_high_values = []

    # 处理雌性文件
    for file in os.listdir(f_path):
        if f_start < num_f:
            if file.endswith(".txt"):
                file_path = os.path.join(f_path, file)
                try:
                    # 读取文件检查最大值
                    with open(file_path, 'r') as f:
                        data = np.array([list(map(float, line.strip().split())) for line in f])
                    if np.sum(data > 4080) >= 10:
                        files_with_high_values.append(file_path)
                        # 删除文件
                        os.remove(file_path)
                        print(f"删除文件: {os.path.basename(file_path)} (最大值: {np.max(data):.2f})")
                        continue  # 跳过最大值超过4000的文件
                    f_files.append(file_path)
                    f_start += 1
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")

    # 处理雄性文件
    for file in os.listdir(m_path):
        if m_start < num_m:
            if file.endswith(".txt"):
                file_path = os.path.join(m_path, file)
                try:
                    # 读取文件检查最大值
                    with open(file_path, 'r') as f:
                        data = np.array([list(map(float, line.strip().split())) for line in f])
                    if np.sum(data > 4080) >= 10:
                        files_with_high_values.append(file_path)
                        # 删除文件
                        os.remove(file_path)
                        print(f"删除文件: {os.path.basename(file_path)} (最大值: {np.max(data):.2f})")
                        continue  # 跳过最大值超过4000的文件
                    m_files.append(file_path)
                    m_start += 1
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")

    return f_files, m_files, files_with_high_values

if __name__ == '__main__':
    num_f, f_start = 8000, 0
    num_m, m_start = 8000, 0

    # 获取公、母文件列表和最大值超过4000的文件
    female_files, male_files, files_with_high_values = getFandMHsiData(num_f, f_start, num_m, m_start)

    # 打印被删除的文件和有效文件数
    if files_with_high_values:
        print("\n光谱数据最大值超过4000的文件（已从磁盘删除）：")
        for fname in files_with_high_values:
            print(os.path.basename(fname))
    else:
        print("\n没有光谱数据最大值超过4000的文件。")
    print(f"有效雌性文件数：{len(female_files)}")
    print(f"有效雄性文件数：{len(male_files)}")

    # 检查是否有有效文件
    if not female_files and not male_files:
        print("错误：所有文件均被排除（最大值超过4000），无法继续处理。")
        exit(1)
