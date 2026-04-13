"""
将目标文件夹下的带有指定字符，如"_1",移动到指定文件夹内，用于将不同采集方式的文件区分开
"""
import os
import shutil
import argparse
import logging
from datetime import datetime

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'file_move_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def move_files_with_pattern(source_dir, target_dir, pattern):
    """
    将源文件夹中文件名包含指定字符的文件移动到目标文件夹。

    Args:
        source_dir (str): 源文件夹路径
        target_dir (str): 目标文件夹路径
        pattern (str): 要匹配的文件名字符
    """
    try:
        # 检查源文件夹是否存在
        if not os.path.isdir(source_dir):
            logger.error(f"源文件夹 '{source_dir}' 不存在！")
            return

        # 检查并创建目标文件夹
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info(f"创建目标文件夹 '{target_dir}'")

        # 计数器
        moved_files = 0
        skipped_files = 0

        # 遍历源文件夹中的文件
        for filename in os.listdir(source_dir):
            if pattern.lower() in filename.lower():  # 忽略大小写匹配
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)

                # 确保是文件而不是文件夹
                if os.path.isfile(source_path):
                    try:
                        # 检查目标路径是否已存在同名文件
                        if os.path.exists(target_path):
                            logger.warning(f"目标文件夹中已存在 '{filename}'，跳过移动")
                            skipped_files += 1
                            continue

                        # 移动文件
                        shutil.move(source_path, target_path)
                        logger.info(f"已移动文件 '{filename}' 到 '{target_dir}'")
                        moved_files += 1
                    except Exception as e:
                        logger.error(f"移动文件 '{filename}' 失败：{e}")
                        skipped_files += 1
                else:
                    logger.warning(f"'{filename}' 不是文件，跳过")
                    skipped_files += 1

        # 总结
        logger.info(f"操作完成：成功移动 {moved_files} 个文件，跳过 {skipped_files} 个文件")

    except Exception as e:
        logger.error(f"发生错误：{e}")


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将文件名包含指定字符的文件移动到目标文件夹")
    parser.add_argument("source_dir", help="源文件夹路径")
    parser.add_argument("target_dir", help="目标文件夹路径")
    parser.add_argument("pattern", help="文件名中要匹配的字符")

    args = parser.parse_args()

    # 运行文件移动函数
    move_files_with_pattern(args.source_dir, args.target_dir, args.pattern)


if __name__ == "__main__":
    # python ./train/excel_preprocess/method_no_fm/0_mv_file_to_folder.py "D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\raw_all_0909_13d6h-12h_8000" "D:\workspace\gdv-egg-model\Code_EggGenderDet\data_hlh_train_20250907_hy\half\raw_all_0909_13d6h-12h_8000_half" "_1"
    main()
