

import os
import shutil
import pandas as pd


# 多日期的分类程序，需要对照两个表，特殊情况的16号与17号字段不一致，需要调整
# A 表 (华裕海兰褐测试结果)
a_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\华裕海兰褐测试结果20250916-17.xlsx"
# B 表 (ForecastSequenceRecord)
b_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\ForecastSequenceRecord0909-早上结束后的-8000枚-公母全局编号.xlsx"

# txt 文件所在目录
txt_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\seg_all_0909_13d6h-12h_8000"
# 输出目录
female_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_8000\female"
male_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\Dataset\huayu_13d12h_8000\male"

os.makedirs(female_dir, exist_ok=True)
os.makedirs(male_dir, exist_ok=True)

# 读取 A 表的 16号 sheet
a_df = pd.read_excel(a_path, sheet_name="17号")
# 读取 B 表
b_df = pd.read_excel(b_path)

# EggCode -> index 的映射，例如 "3-5" -> 15
def eggcode_to_index(eggcode: str) -> int:
    row, col = eggcode.split("-")
    row, col = int(row), int(col)
    return (row - 1) * 5 + col  # 1-based

copied_female, copied_male, missing_files = 0, 0, 0

# 遍历 A 表每一行
for _, row in a_df.iterrows():
    seq = row["Number"]  # Sequence 对应
    for col in a_df.columns[2:]:
        value = row[col]
        if value not in [1, 2]:  # 只关心 female/male
            continue

        # Mxx / Fxx → Category, Rounds
        if col.startswith("M"):
            category = 2
            rounds = int(col[1:])
        elif col.startswith("F"):
            category = 1
            rounds = int(col[1:])
        else:
            continue

        # 在 B 表里找匹配行 (不限制 EventDate)
        matched = b_df[(b_df["Category"] == category) &
                       (b_df["GlobalCategoryRoundsSequence"] == rounds) &
                       # (b_df["Rounds"] == rounds) &
                       (b_df["Sequence"] == seq)]

        for _, b_row in matched.iterrows():
            plate = str(b_row["PlateCode"])
            eggcode = str(b_row["EggCode"])
            idx = eggcode_to_index(eggcode)
            filename = f"{plate}-{idx}.txt"
            filepath = os.path.join(txt_dir, filename)

            if os.path.exists(filepath):
                if value == 1:  # female
                    shutil.copy(filepath, female_dir)
                    copied_female += 1
                elif value == 2:  # male
                    shutil.copy(filepath, male_dir)
                    copied_male += 1
            else:
                print(f"[缺失] 文件不存在: {filepath}")
                missing_files += 1

print(f"\n✅ 复制完成: female={copied_female}, male={copied_male}, 缺失文件={missing_files}")

