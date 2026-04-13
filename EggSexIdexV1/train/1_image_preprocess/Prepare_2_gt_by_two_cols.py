
# 分割公母：采集数据后，分公母落盘的情况，对应处理rename_files_by_id.py梳理后的文件 {蛋序号}-{盘号}.txt.
# 根据D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\出雏记录20250806.xlsx
# 这种2两列的标签表，进行{蛋序号}-{盘号} 和 {蛋序号}-{盘号}.txt进行匹配，并划分到对应的female和male中

import os
import pandas as pd
import glob
import re
import shutil
from tabulate import tabulate


def load_egg_gender_mapping(excel_path, sheet_name=0):
    """从 Excel 文件加载 EggID 到性别的映射"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if df.shape[1] < 2:
            print("错误: 表格必须包含至少两列（鸡蛋编号 和 复检结果）")
            return {}
        df.columns = ['EggID', 'Gender'] if len(df.columns) == 2 else df.columns
        mapping = {}
        for _, row in df.iterrows():
            egg_id = str(row['EggID'])
            gender = row['Gender']
            # 解析 EggID，格式为 {蛋序号}-{盘号}，如 02-255
            match = re.match(r'^(\d+)-(\d+)$', egg_id)
            if not match:
                print(f"警告: EggID {egg_id} 格式不匹配，跳过")
                continue
            egg_num, tray_num = match.groups()
            egg_num = str(int(egg_num))  # 去掉前导零，如 '02' -> '2'
            id_egg = f"{tray_num}-{egg_num}"  # 如 255-2
            # 严格检查性别，仅 1 或 2 有效
            if pd.isna(gender) or gender not in [1, 2]:
                mapping[id_egg] = (None, egg_id, gender)
                print(f"警告: EggID {egg_id} 的复检结果 {gender} 无效，跳过")
            else:
                mapping[id_egg] = ('female' if gender == 1 else 'male', egg_id, gender)
        return mapping
    except Exception as e:
        print(f"加载 Excel 文件失败: {e}")
        return {}


def get_unique_target_path(target_path):
    """生成唯一的文件路径，避免覆盖"""
    if not os.path.exists(target_path):
        return target_path
    base, ext = os.path.splitext(target_path)
    suffix = 1
    while True:
        new_path = f"{base}_{suffix}{ext}"
        if not os.path.exists(new_path):
            return new_path
        suffix += 1


def move_egg_files_by_gender(input_dir, excel_path, output_dir, sheet_name=0):
    """根据性别表格将重命名后的文件复制到 female 或 male 文件夹，保留原始文件"""
    egg_gender_mapping = load_egg_gender_mapping(excel_path, sheet_name)
    if not egg_gender_mapping:
        print("无法加载 EggID 到性别的映射，退出程序")
        return

    female_dir = os.path.join(output_dir, "female")
    male_dir = os.path.join(output_dir, "male")
    os.makedirs(female_dir, exist_ok=True)
    os.makedirs(male_dir, exist_ok=True)

    log_entries = []
    success_count = 0
    failure_count = 0

    pattern = re.compile(r'^(\d+-\d+)\.txt$')

    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if not match:
            log_entries.append({
                'Original File': file_name,
                'Target Path': '-',
                'Status': 'Failed',
                'Reason': '文件名格式不匹配'
            })
            failure_count += 1
            continue

        id_egg = match.group(1)  # 如 255-2
        if id_egg not in egg_gender_mapping:
            log_entries.append({
                'Original File': file_name,
                'Target Path': '-',
                'Status': 'Failed',
                'Reason': f'Id-蛋序号 {id_egg} 在表格中未找到'
            })
            failure_count += 1
            continue

        gender, orig_egg_id, orig_gender = egg_gender_mapping[id_egg]
        if gender is None:
            log_entries.append({
                'Original File': file_name,
                'Target Path': '-',
                'Status': 'Failed',
                'Reason': f'Id-蛋序号 {id_egg} (EggID: {orig_egg_id}) 性别无效 (Gender: {orig_gender})'
            })
            failure_count += 1
            continue

        target_dir = female_dir if gender == 'female' else male_dir
        target_path = os.path.join(target_dir, file_name)
        target_path = get_unique_target_path(target_path)

        try:
            shutil.copy(file_path, target_path)
            log_entries.append({
                'Original File': file_name,
                'Target Path': target_path,
                'Status': 'Success',
                'Reason': f'复制到 {gender} 文件夹 (EggID: {orig_egg_id}, Gender: {orig_gender})'
            })
            success_count += 1
            print(f"复制成功: {file_name} -> {target_path} (EggID: {orig_egg_id}, Gender: {orig_gender})")
        except Exception as e:
            log_entries.append({
                'Original File': file_name,
                'Target Path': target_path,
                'Status': 'Failed',
                'Reason': f'复制失败: {str(e)} (EggID: {orig_egg_id}, Gender: {orig_gender})'
            })
            failure_count += 1

    log_path = os.path.join(output_dir, "move_log.md")
    headers = ['Original File', 'Target Path', 'Status', 'Reason']
    table_data = [[entry['Original File'], entry['Target Path'], entry['Status'], entry['Reason']]
                  for entry in log_entries]
    table_md = f"# File Copy Log\n\n" + tabulate(table_data, headers=headers, tablefmt='github')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(table_md)
    print(f"日志已保存至: {log_path}")
    print(f"处理完成: 成功 {success_count} 个，失败 {failure_count} 个")


if __name__ == "__main__":
    # 分割8月4号的蛋图像文件，对应分公母出雏的情况的表
    input_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\seg_all_huayu28"
    excel_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\2025-07-27采样种蛋出雏复检结果.xlsx"
    output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\Dataset\xiaoming_new_27"
    move_egg_files_by_gender(input_dir, excel_path, output_dir)
