import os
import shutil


def copy_files(A_dir, B_dir, C_dir):
    # 检查文件夹 C 是否存在，不存在则创建
    if not os.path.exists(C_dir):
        os.makedirs(C_dir)

    # 获取 A 文件夹中 female 和 male 子文件夹路径
    a_female_dir = os.path.join(A_dir, "female")
    a_male_dir = os.path.join(A_dir, "male")

    # 确保 C 文件夹下有 female 和 male 子文件夹
    a_female_dest = os.path.join(C_dir, "female")
    a_male_dest = os.path.join(C_dir, "male")

    if not os.path.exists(a_female_dest):
        os.makedirs(a_female_dest)
    if not os.path.exists(a_male_dest):
        os.makedirs(a_male_dest)

    # 获取 A 中 female 和 male 文件夹下的所有文件名（去除扩展名以便后续匹配）
    female_files = set(f.replace('.txt', '') for f in os.listdir(a_female_dir) if f.endswith('.txt'))
    male_files = set(f.replace('.txt', '') for f in os.listdir(a_male_dir) if f.endswith('.txt'))

    # 获取 B 文件夹中的所有 txt 文件
    b_files = set(f.replace('.txt', '') for f in os.listdir(B_dir) if f.endswith('.txt'))

    # 复制 B 中与 A 同名的文件到 C 文件夹中的相应子文件夹
    for filename in b_files:
        if filename in female_files:
            src_path = os.path.join(B_dir, filename + '.txt')
            dest_path = os.path.join(a_female_dest, filename + '.txt')
            shutil.copy(src_path, dest_path)
            print(f"复制文件 {filename}.txt 到 {a_female_dest}")
        elif filename in male_files:
            src_path = os.path.join(B_dir, filename + '.txt')
            dest_path = os.path.join(a_male_dest, filename + '.txt')
            shutil.copy(src_path, dest_path)
            print(f"复制文件 {filename}.txt 到 {a_male_dest}")

    print("文件复制完成。")


# 示例使用
# A 分female和male
A_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000"
# B 刚裁切的，不分female和male的
B_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\half\seg_all_0907_13d6h-12h_4000_half"
# Dataset 同名蛋分公母
C_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\half\huayu_13d12h_4000_half"

# 调用函数
copy_files(A_dir, B_dir, C_dir)
