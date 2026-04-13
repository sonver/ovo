# 临时处理12d12h和12d18h的数据，需要用经过Prepare_2_gt_hlh_multiple_date分类好的13d12h的txt，并结合
# 文瑶给的PlateCode对应关系表，来进行female和male分割

import os
import shutil
import pandas as pd
import re
from tqdm import tqdm

def normalize_plate(s):
    """
    将 plate-like 值正规化为仅数字的字符串，去掉空格/小数点/科学计数符等。
    例如：
       "6.022025090700005e+18" -> "6022025090700005..." (注意：如果是 float 表示且已丢精度，无法恢复)
       "6022025090800007604" -> "6022025090800007604"
    这个函数不会恢复被 float 丢掉的精度；最好的做法是确保 Excel 的列为文本。
    """
    if s is None:
        return ""
    s = str(s).strip()
    # 直接提取所有数字字符（保守且常用）
    digits = re.sub(r'\D', '', s)
    return digits

def build_b_index(folder_b):
    """
    遍历 B 文件夹，建立 (plate_norm, idx_str) -> full_path 的字典索引。
    假设 B 中的文件名格式为 {PlateCode}-{Index}.txt（其他格式会被跳过）
    """
    index = {}
    for fname in os.listdir(folder_b):
        if not fname.lower().endswith('.txt'):
            continue
        base = os.path.splitext(fname)[0]
        if '-' not in base:
            continue
        plate_part, idx_part = base.split('-', 1)
        plate_norm = normalize_plate(plate_part)
        idx_str = idx_part.strip()
        key = (plate_norm, idx_str)
        index[key] = os.path.join(folder_b, fname)
    return index

def main(
    folder_a,         # A 目录（含 female/ male 子目录）
    folder_b,         # B 目录（未分类）
    mapping_file,     # PlateCode 对应表（Excel/CSV），包含 oldPlateCode -> newPlateCode
    old_col='oldPlateCode',
    new_col='newPlateCode',
    out_female=None,
    out_male=None
):
    # 输出目录
    if out_female is None:
        out_female = os.path.join(folder_b, 'out_female')
    if out_male is None:
        out_male = os.path.join(folder_b, 'out_male')
    os.makedirs(out_female, exist_ok=True)
    os.makedirs(out_male, exist_ok=True)

    # 1) 读取 mapping 表，尽量以字符串读入（防止 float -> 科学计数法）
    #    如果 mapping_file 是 CSV 可改用 pd.read_csv
    try:
        mapping_df = pd.read_excel(mapping_file, dtype=str,sheet_name="4000")
    except Exception as e:
        print(f"[WARN] 读取 Excel 失败（尝试用 CSV），错误: {e}")
        mapping_df = pd.read_csv(mapping_file, dtype=str)

    # 检查列是否存在
    if old_col not in mapping_df.columns or new_col not in mapping_df.columns:
        raise ValueError(f"映射表必须包含列: {old_col} 和 {new_col}. 当前列: {mapping_df.columns.tolist()}")

    # 规范化 mapping 字典
    mapping_dict = {}
    for old, new in zip(mapping_df[old_col].astype(str), mapping_df[new_col].astype(str)):
        old_n = normalize_plate(old)
        new_n = normalize_plate(new)
        if old_n:
            mapping_dict[old_n] = new_n

    print(f"[INFO] 映射表加载完毕，条目: {len(mapping_dict)} (规范化后)")

    # 2) 预索引 B 文件夹
    b_index = build_b_index(folder_b)
    print(f"[INFO] B 文件夹索引建立完毕，条目: {len(b_index)}")

    # 3) 遍历 A 下 female/male 子目录，查找对应 B 文件并复制
    stats = {'copied_female':0, 'copied_male':0, 'missing_mapping':0, 'missing_bfile':0}
    missing_mapping_files = []
    missing_b_files = []

    for gender in ('female','male'):
        dir_a = os.path.join(folder_a, gender)
        if not os.path.isdir(dir_a):
            print(f"[WARN] A 子目录不存在: {dir_a}, 跳过")
            continue

        for fname in tqdm(os.listdir(dir_a), desc=f"Processing {gender}"):
            if not fname.lower().endswith('.txt'):
                continue
            base = os.path.splitext(fname)[0]
            if '-' not in base:
                # 如果文件名不符合 {Plate}-{idx}.txt 格式，跳过或记录
                print(f"[WARN] 非预期文件名格式 (跳过): {fname}")
                continue

            plate_part, idx_part = base.split('-',1)
            plate_norm = normalize_plate(plate_part)
            idx_str = idx_part.strip()

            if plate_norm == "":
                stats['missing_mapping'] += 1
                missing_mapping_files.append((fname, "plate empty after normalize"))
                continue

            # 映射到 B 的 plate
            new_plate = mapping_dict.get(plate_norm)
            if new_plate is None:
                # mapping 表中找不到对应的 oldPlateCode，记录并继续
                stats['missing_mapping'] += 1
                missing_mapping_files.append((fname, plate_norm))
                continue

            key = (new_plate, idx_str)
            target_path = b_index.get(key)
            if target_path and os.path.exists(target_path):
                # 复制到输出目录
                dst_dir = out_female if gender=='female' else out_male
                shutil.copy(target_path, os.path.join(dst_dir, os.path.basename(target_path)))
                if gender=='female':
                    stats['copied_female'] += 1
                else:
                    stats['copied_male'] += 1
            else:
                stats['missing_bfile'] += 1
                missing_b_files.append((fname, new_plate, idx_str, key))

    print("==== 完成 ====")
    print(stats)
    if missing_mapping_files:
        print("\n[Missing mapping examples] (A_filename, normalized_plate):")
        for ex in missing_mapping_files[:20]:
            print(" ", ex)
    if missing_b_files:
        print("\n[Missing B-file examples] (A_filename, mapped_new_plate, idx, key):")
        for ex in missing_b_files[:20]:
            print(" ", ex)

    return stats, missing_mapping_files, missing_b_files

if __name__ == "__main__":
    # ==== 路径配置 ====
    # A 文件夹（已分类）
    folder_a = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_4000"
    # B 文件夹（未分类）
    folder_b = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\seg_all_0907_13d0h_4000"

    # PlateCode 对应关系表 (Excel/CSV)，假设有两列 oldPlateCode, newPlateCode
    mapping_file = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\2025-09采样对应关系.xlsx"

    # 示例路径（替换为你的实际路径）

    out_female = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d0h_4000\female"
    out_male   = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d0h_4000\male"

    stats, miss_map, miss_b = main(folder_a, folder_b, mapping_file, old_col='oldPlateCode', new_col='newPlateCode',
                                   out_female=out_female, out_male=out_male)
