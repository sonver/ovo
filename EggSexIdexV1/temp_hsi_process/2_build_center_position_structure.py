import pandas as pd
import os

def generate_images_center(excel_path, output_path=None):
    """
    从 Excel 文件生成 IMAGES_CENTER 字典，盘号从表格中动态读取，并检查盘号-行-列的唯一性。
    :param excel_path: Excel 文件路径
    :param output_path: 可选，输出 Python 文件路径
    :return: IMAGES_CENTER 字典
    """
    # 读取 Excel 文件
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        return None

    # 每盘规格：7 行 6 列
    rows, cols = 7, 6

    # 从表格中获取所有唯一的盘号
    plates = set(df['盘号'].astype(int))
    print(f"读取到的盘号: {sorted(plates)}")

    # 检查盘号-行-列的唯一性
    seen_positions = set()  # 记录已见过的 (盘号, 行号, 列号) 组合
    for _, row in df.iterrows():
        plate = int(row['盘号'])
        row_num = int(row['行号'])  # 行号从 1 开始
        col_num = int(row['列号'])  # 列号从 1 开始
        position = (plate, row_num, col_num)

        if position in seen_positions:
            raise ValueError(f"发现重复的盘号-行-列组合: 盘号 {plate}, 行 {row_num}, 列 {col_num}")
        seen_positions.add(position)

    # 初始化 IMAGES_CENTER 字典
    IMAGES_CENTER = {}
    for plate in plates:
        # 初始化 7 行 6 列的嵌套列表，填充 None
        IMAGES_CENTER[plate] = [[None for _ in range(cols)] for _ in range(rows)]

    # 遍历 Excel 数据，填充坐标
    for _, row in df.iterrows():
        plate = int(row['盘号'])
        row_num = int(row['行号']) - 1  # 行号从 1 开始，转换为 0-based 索引
        col_num = int(row['列号']) - 1  # 列号从 1 开始，转换为 0-based 索引
        x = int(row['x坐标'])
        y = int(row['y坐标'])

        # 填充坐标
        if plate in IMAGES_CENTER and 0 <= row_num < rows and 0 <= col_num < cols:
            IMAGES_CENTER[plate][row_num][col_num] = (x, y)
        else:
            print(f"数据异常: 盘号 {plate}, 行号 {row_num + 1}, 列号 {col_num + 1}")

    # 检查是否有缺失数据
    for plate, grid in IMAGES_CENTER.items():
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is None:
                    print(f"盘号 {plate} 行 {r + 1} 列 {c + 1} 缺少坐标数据")

    # 如果指定了输出路径，保存为 Python 文件
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("IMAGES_CENTER = {\n")
            for plate in sorted(IMAGES_CENTER.keys()):  # 按盘号排序
                f.write(f"    {plate}: [\n")
                for row in IMAGES_CENTER[plate]:
                    f.write(f"        {row},\n")
                f.write("    ],\n")
            f.write("}\n")
        print(f"IMAGES_CENTER 已保存至: {output_path}")

    return IMAGES_CENTER

def main():
    # Excel 文件路径
    excel_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\temp_hsi_process\center_xy.xlsx"

    # 可选：输出 Python 文件路径
    output_path = r"D:\workspace\gdv-egg-model\Code_EggGenderDet\temp_hsi_process\images_center.py"

    # 生成 IMAGES_CENTER 字典
    try:
        IMAGES_CENTER = generate_images_center(excel_path, output_path)
    except ValueError as e:
        print(f"错误: {e}")
        return

    if IMAGES_CENTER:
        print("IMAGES_CENTER 字典生成成功！")
        # 可选：打印部分数据以验证
        print("示例数据(盘号 11):")
        print(IMAGES_CENTER[11])

if __name__ == "__main__":
    main()