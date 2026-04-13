import os
import shutil

import pandas as pd
import re


def getsex202412156(root_path, source_folder, destination_folder1, destination_folder2):
    df1 = pd.read_excel(root_path)
    # print(len(df1.columns))
    # 删行，删列，重置行索引
    df1 = df1.drop(df1.columns[0:2], axis=1)
    df1 = df1.drop(index=[0, 1, 2])  # 去掉索引行
    df1 = df1.reset_index(drop=True)
    # print(len(df1.columns))
    # print(df1)

    # # 重置列名
    # # new_column_names = list(range(1, 96))
    # new_column_names = list(range(97, 288))
    # df1.columns = new_column_names

    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    df1 = df1.fillna(0)
    print(df1)
    # return

    txt_files = os.listdir(source_folder)
    for txt_file in txt_files:
        source_file_path = os.path.join(source_folder, txt_file)
        match = re.match(r'egg(\d+)-(\d+)\.txt', txt_file)
        X = int(match.group(1))
        Y = int(match.group(2))
        print(f"Extracted X: {X}, Y: {Y}")
        # continue
        # 查找 DataFrame 中的对应值
        # result = df1[(df1['A'] == X) & (df1['B'] == Y)]['C']
        # label = X in df1
        # print(label)
        if X in df1:  # 如果蛋盘在表里有
            result = df1.loc[Y - 1, X]
            if result == 2:  # 如果是公
                shutil.copy(source_file_path, destination_folder1)
                print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder1}' 完成")
            if result == 1:  # 如果是母
                shutil.copy(source_file_path, destination_folder2)
                print(f"复制文件 '{txt_file}' 到文件夹 '{destination_folder2}' 完成")

        else:
            print(f"文件 '{txt_file}' 属于有问题盘")
            continue


if __name__ == '__main__':
    # 2构造数据集GT

    # 采集数据现场鉴定的结果的excel表路径D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2525年4月7日-褐蛋\3月测试数据20250407.xlsx
    root_path = r'D:\workspace\gdv-egg-model\Code_EggGenderDet\data\2525年4月7日-褐蛋\2024-12-褐壳出雏结果1-59.xlsx'
    # GT存放位置
    # source_folder = './data/egg_seg_data_a'  # 经过第一步分割后的单枚蛋数据目录
    source_folder = '../data/egg_seg_data_1month'  # 经过第一步分割后的单枚蛋数据目录
    # destination_folder1 = './data/Dataset/testdatav12.15_a/male'  # 公蛋存放目录
    destination_folder1 = '../data/Dataset/1month/male'  # 公蛋存放目录
    # destination_folder2 = './data/Dataset/testdatav12.15_a/female'  # 母蛋存放目录
    destination_folder2 = '../data/Dataset/1month/female'  # 母蛋存放目录


    # 2024 12 15 京褐
    getsex202412156(root_path, source_folder, destination_folder1, destination_folder2)

    # delete_single()
    # find_GT_label()
    # new_data_all()
