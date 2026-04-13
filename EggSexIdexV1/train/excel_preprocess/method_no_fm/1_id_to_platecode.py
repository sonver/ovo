"""
将标签表中的ID，与映射表中的ID关联到，并替换标签表的ID为PlateCode，用于image_preprocess第二步，进行按标签分割公母文件
"""
import pandas as pd
import uuid

# 读取 Excel 文件
try:
    test_data = pd.read_excel("6月华裕海兰褐出雏结果再次严格筛选.xlsx", sheet_name="Sheet3",
                              header=None)  # 不假设 header，直接读取
    forecast_data = pd.read_excel("ForecastRecords.xlsx", sheet_name="Result 1")
except FileNotFoundError as e:
    print(f"错误：文件未找到 - {e}")
    exit(1)
except Exception as e:
    print(f"读取 Excel 文件时出错：{e}")
    exit(1)

# 调试输出：打印 test_data 的前几行和列名
print("test_data 的原始数据（前5行）：\n", test_data.iloc[:5])
print("test_data 的列数：", test_data.shape[1])

# 假设 PalletCode 行是第一行（索引 0）
pallet_row_index = 0
pallet_codes = test_data.iloc[pallet_row_index, 1:].values  # 从第二列开始获取数字（跳过第一列的“PalletCode”）
print("PalletCode 行的原始值：", pallet_codes)

# 创建 ID 到 PlateCode 的映射字典
id_to_platecode = dict(zip(forecast_data['Id'], forecast_data['PlateCode']))
print("ID 到 PlateCode 的映射（前5项）：", dict(list(id_to_platecode.items())[:5]))

# 将 PalletCode 数字替换为对应的 PlateCode 值
new_pallet_codes = []
for code in pallet_codes:
    if pd.notna(code):  # 忽略空值
        try:
            new_pallet_codes.append(id_to_platecode.get(int(code), code))  # 转换为 int 并查找 PlateCode
        except ValueError:
            new_pallet_codes.append(code)  # 如果无法转换为 int，保留原值
    else:
        new_pallet_codes.append(code)  # 保留空值

print("替换后的 PlateCode 值：", new_pallet_codes)

# 更新 DataFrame 中的 PalletCode 行
test_data.iloc[pallet_row_index, 1:1 + len(new_pallet_codes)] = new_pallet_codes

# 保存更新后的 DataFrame 到新的 Excel 文件
try:
    output_filename = f"updated_test_data_{uuid.uuid4().hex[:8]}.xlsx"
    test_data.to_excel(output_filename, index=False, header=False)  # 不写入索引和 header
    print(f"更新后的文件已保存为 {output_filename}")
except Exception as e:
    print(f"保存文件时出错：{e}")
