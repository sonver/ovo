"""
将表中的404-1-5这种数据，变更为23-315这种，保持一致
"""
import pandas as pd
import re
import os


def convert_egg_id(egg_id):
    """转换鸡蛋编号格式"""
    if not isinstance(egg_id, str):
        return egg_id, False, f"非法格式: 非字符串 ({egg_id})"

    # 匹配 {蛋序号}-{盘号} 格式，如 05-372
    match_simple = re.match(r'^(\d+)-(\d+)$', egg_id)
    if match_simple:
        return egg_id, True, "无需转换"

    # 匹配 {盘号}-{行号}-{蛋序号} 格式，如 403-6-2
    match_complex = re.match(r'^(\d+)-(\d+)-(\d+)$', egg_id)
    if match_complex:
        tray_num, row_num, egg_num = map(int, match_complex.groups())
        if 1 <= row_num <= 7 and 1 <= egg_num <= 5:
            new_egg_num = (row_num - 1) * 5 + egg_num
            new_egg_id = f"{new_egg_num}-{tray_num}"
            return new_egg_id, True, f"转换: {egg_id} -> {new_egg_id}"
        else:
            return egg_id, False, f"非法行号或蛋序号: 行号={row_num}, 蛋序号={egg_num}"

    return egg_id, False, f"非法格式: {egg_id}"


def convert_egg_ids_in_excel(input_path, output_path, log_path):
    """转换落盘数据表的鸡蛋编号格式，并保存所有工作表"""
    # 读取所有工作表
    xls = pd.ExcelFile(input_path)
    sheet_names = xls.sheet_names

    # 处理 落盘数据 表
    df_luopan = pd.read_excel(input_path, sheet_name="落盘数据")
    if df_luopan.columns[0] != '鸡蛋编号' or df_luopan.columns[1] != '出雏结果':
        print("错误: 落盘数据表列名不符合预期（预期: 鸡蛋编号, 出雏结果）")
        return

    log_entries = []
    df_luopan['鸡蛋编号_原始'] = df_luopan['鸡蛋编号']  # 保留原始编号用于日志
    for idx, row in df_luopan.iterrows():
        egg_id = row['鸡蛋编号']
        new_egg_id, success, reason = convert_egg_id(egg_id)
        df_luopan.at[idx, '鸡蛋编号'] = new_egg_id
        log_entries.append({
            'Original EggID': egg_id,
            'New EggID': new_egg_id,
            'Status': 'Success' if success else 'Failed',
            'Reason': reason
        })

    # 保存日志
    df_log = pd.DataFrame(log_entries)
    df_log.to_excel(log_path, index=False)
    print(f"日志已保存至: {log_path}")

    # 保存所有工作表到新 Excel 文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 保存修改后的 落盘数据 表
        df_luopan.drop(columns=['鸡蛋编号_原始']).to_excel(writer, sheet_name="落盘数据", index=False)

        # 复制其他工作表
        for sheet_name in sheet_names:
            if sheet_name != "落盘数据":
                df_other = pd.read_excel(input_path, sheet_name=sheet_name)
                df_other.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"转换后的 Excel 文件已保存至: {output_path}")


if __name__ == "__main__":
    input_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\train\excel_preprocess\method_diff_fm\2025-07-27采样种蛋出雏复检结果.xlsx"
    output_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\train\excel_preprocess\method_diff_fm\xiaoming_hlh_727_modified.xlsx"
    log_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\train\excel_preprocess\method_diff_fm\convert_log.xlsx"

    convert_egg_ids_in_excel(input_path, output_path, log_path)