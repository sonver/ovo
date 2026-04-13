"""
重命名文件格式，将文件命中的_1和_2替换为空，用于image_preprocess
"""
import os
import glob


def rename_files(input_dir, pattern="egg*.txt", old_str="_2", new_str=""):
    """
    批量重命名文件，将文件名中的 old_str 替换为 new_str。

    参数:
        input_dir (str): 输入文件夹路径
        pattern (str): 文件名模式（默认匹配 egg 开头的 .txt 文件）
        old_str (str): 要替换的字符串（默认 "_2"）
        new_str (str): 替换后的字符串（默认空字符串）
    """
    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return

    # 获取匹配模式的文件列表
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        print(f"警告：在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return

    # 计数器
    renamed_count = 0
    skipped_count = 0

    # 遍历文件并重命名
    for old_path in files:
        # 获取文件名（不含路径）
        old_name = os.path.basename(old_path)

        # 检查是否包含 old_str
        if old_str in old_name:
            # 替换字符串
            new_name = old_name.replace(old_str, new_str)
            new_path = os.path.join(input_dir, new_name)

            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"跳过 {old_name}：目标文件名 {new_name} 已存在")
                skipped_count += 1
                continue

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"已重命名：{old_name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名 {old_name} 失败：{str(e)}")
                skipped_count += 1
        else:
            print(f"跳过 {old_name}：未找到 {old_str}")
            skipped_count += 1

    # 总结
    print(f"\n处理完成：成功重命名 {renamed_count} 个文件，跳过 {skipped_count} 个文件")


if __name__ == "__main__":
    # 示例用法
    input_directory = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\half\seg_all_0909_13d6h-12h_8000_half"  # 替换为你的文件夹路径
    rename_files(input_directory, pattern="*.txt", old_str="_1", new_str="")