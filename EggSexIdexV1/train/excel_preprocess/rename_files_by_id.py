"""
与rename_files作用相同，但是处理的数据格式不同，这个用于处理两列数据的，即第一列为F03-153，第二列为性别1 | 2的表，进行列重命名，变成153-11这种盘号-序号的
"""
import os
import pandas as pd
import glob
import re
from tabulate import tabulate


def load_platecode_to_id_mapping(excel_path, sheet_name="Result 1"):
    """从 Excel 文件加载 PlateCode 到 Id 的映射"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        # 提取 PlateCode 和 Id 列，构建映射字典
        mapping = dict(zip(df['PlateCode'], df['Id']))
        return mapping
    except Exception as e:
        print(f"加载 Excel 文件失败: {e}")
        return {}


def rename_egg_files(input_dir, excel_path, output_dir=None):
    """重命名文件夹中的 egg 文件，使用 Excel 中的 PlateCode 到 Id 映射"""
    # 加载 PlateCode 到 Id 的映射
    platecode_to_id = load_platecode_to_id_mapping(excel_path)
    if not platecode_to_id:
        print("无法加载 PlateCode 到 Id 的映射，退出程序")
        return

    # 创建输出目录（若未提供，则与输入目录相同）
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    # 日志记录
    log_entries = []
    success_count = 0
    failure_count = 0

    # 正则表达式匹配文件名：egg + 22位盘号 + - + 蛋序号 + .txt
    pattern = re.compile(r'^egg([A-Za-z0-9]{22})-(\d+)\.txt$')

    # 扫描输入目录中的 .txt 文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if not match:
            log_entries.append({
                'Original File': file_name,
                'New File': '-',
                'Status': 'Failed',
                'Reason': '文件名格式不匹配'
            })
            failure_count += 1
            continue

        platecode = match.group(1)
        egg_number = match.group(2)

        # 查找对应的 Id
        if platecode not in platecode_to_id:
            log_entries.append({
                'Original File': file_name,
                'New File': '-',
                'Status': 'Failed',
                'Reason': f'PlateCode {platecode} 在 Excel 中未找到'
            })
            failure_count += 1
            continue

        id_value = platecode_to_id[platecode]
        new_file_name = f"{id_value}-{egg_number}.txt"
        new_file_path = os.path.join(output_dir, new_file_name)

        # 检查目标文件是否已存在
        if os.path.exists(new_file_path):
            log_entries.append({
                'Original File': file_name,
                'New File': new_file_name,
                'Status': 'Failed',
                'Reason': '目标文件已存在'
            })
            failure_count += 1
            continue

        # 执行重命名
        try:
            os.rename(file_path, new_file_path)
            log_entries.append({
                'Original File': file_name,
                'New File': new_file_name,
                'Status': 'Success',
                'Reason': '-'
            })
            success_count += 1
            print(f"重命名成功: {file_name} -> {new_file_name}")
        except Exception as e:
            log_entries.append({
                'Original File': file_name,
                'New File': new_file_name,
                'Status': 'Failed',
                'Reason': f'重命名失败: {str(e)}'
            })
            failure_count += 1

    # 保存日志到 Markdown 文件
    log_path = os.path.join(output_dir, "rename_log.md")
    headers = ['Original File', 'New File', 'Status', 'Reason']
    table_data = [[entry['Original File'], entry['New File'], entry['Status'], entry['Reason']]
                  for entry in log_entries]
    table_md = f"# File Rename Log\n\n" + tabulate(table_data, headers=headers, tablefmt='github')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(table_md)
    print(f"日志已保存至: {log_path}")
    print(f"处理完成: 成功 {success_count} 个，失败 {failure_count} 个")


if __name__ == "__main__":
    input_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\seg_all_huayu28"
    excel_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\excel_preprocess\ForecastRecords_0726-28.xlsx"
    output_dir = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train\seg_all_huayu28_renamed"
    rename_egg_files(input_dir, excel_path, output_dir)
