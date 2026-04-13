import os
import shutil
import pandas as pd
import re


def getsex202412156():
    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder1, exist_ok=True)
    os.makedirs(destination_folder2, exist_ok=True)

    # 读取 Excel 文件
    df1 = pd.read_excel(root_path)

    # 删除不需要的行和列，重置行索引
    df1 = df1.drop(df1.columns[0:2], axis=1)
    df1 = df1.drop(index=[0, 1, 2])
    df1 = df1.reset_index(drop=True)

    # 填充 NaN 值为 0
    pd.set_option('display.max_rows', None)
    df1 = df1.fillna(0)
    print("df1 structure:")
    print(df1)

    # 打印列索引和类型
    print("df1 columns:", df1.columns.tolist())
    print("df1 columns type:", type(df1.columns[0]))

    # 获取源文件夹中的所有 txt 文件
    txt_files = os.listdir(source_folder)
    print(f"Found {len(txt_files)} files in source folder")

    for txt_file in txt_files:
        source_file_path = os.path.join(source_folder, txt_file)

        # 使用正则表达式提取文件名中的 X 和 Y
        match = re.match(r'egg(\d+)-(\d+)\.txt', txt_file)
        if not match:
            print(f"文件 '{txt_file}' 名称格式不正确，跳过")
            continue

        X = int(match.group(1))
        Y = int(match.group(2))
        print(f"Processing file '{txt_file}': Extracted X: {X}, Y: {Y}")

        # 将 X 转换为字符串，与 df1.columns 匹配
        X_str = X

        # 检查 X 是否在 DataFrame 的列中
        if X_str in df1.columns:
            # 检查 Y 是否在有效范围内
            if 0 <= Y - 1 < len(df1):
                result = df1.loc[Y - 1, X_str]
                # 将 result 转换为整数
                result = int(result)
                print(f"Value at (X={X_str}, Y={Y}): {result}")

                # 只处理值为 1（母蛋）或 2（公蛋）的情况
                if result == 2:  # 公蛋
                    shutil.copy(source_file_path, destination_folder1)
                    print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder1}' 完成")
                elif result == 1:  # 母蛋
                    shutil.copy(source_file_path, destination_folder2)
                    print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder2}' 完成")
                # elif result == 3: # 未受精蛋
                #     shutil.copy(source_file_path, destination_folder3)
                #     print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder3}' 完成")
                # elif result == 4: # 死胎蛋
                #     shutil.copy(source_file_path, destination_folder4)
                #     print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder4}' 完成")
                else:
                    print(f"文件 '{txt_file}' 的值 '{result}' 不是 1 或 2，忽略")
            else:
                print(f"文件 '{txt_file}' 的 Y={Y} 超出范围 (1-{len(df1) + 1})，忽略")
        else:
            print(f"文件 '{txt_file}' 的 X={X_str} 不在 df1 列索引中 ({df1.columns[0]}-{df1.columns[-1]})，忽略")


if __name__ == '__main__':
    # Excel 表路径
    root_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2525年4月7日-褐蛋\2024-12-褐壳出雏结果1-59.xlsx'
    # 源文件夹和目标文件夹
    source_folder = r'./data/egg_seg_data_1month'
    destination_folder1 = './data/Dataset/1month/male'  # 公蛋存放目录
    destination_folder2 = './data/Dataset/1month/female'  # 母蛋存放目录
    destination_folder3 = './data/Dataset/1month/unfertilized'  # 白蛋存放目录
    destination_folder4 = './data/Dataset/1month/dead'  # 死蛋存放目录

    # 执行函数
    getsex202412156()