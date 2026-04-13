"""
标定图片中心点可视化程序工具，坐标系为左下角为0,0
"""
import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


def annotate_image(image_path):
    """
    交互式标注图片的像素坐标，用户手动记录，坐标文本避免遮挡。
    :param image_path: 图片路径
    :return: 是否继续（True 表示继续，False 表示退出）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return True

    # 获取文件名用于窗口标题
    file_name = os.path.basename(image_path)
    # 创建窗口，标题包含文件名
    window_name = f"Annotating: {file_name} (Click to mark, 'n' for next, 'q' to quit)"
    cv2.namedWindow(window_name)

    # 深拷贝图片以避免修改原图
    img_display = img.copy()
    height, width = img.shape[:2]

    # 存储当前图片的点击坐标和文本区域
    points = []  # 存储点击点 (x, y_adjusted)
    text_boxes = []  # 存储文本区域 (x1, y1, x2, y2)

    def check_overlap(new_box, existing_boxes):
        """检查新文本框是否与已有文本框重叠"""
        x1, y1, x2, y2 = new_box
        for box in existing_boxes:
            bx1, by1, bx2, by2 = box
            # 检查是否重叠
            if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                return True
        return False

    def find_text_position(pt, text, existing_boxes):
        """
        找到合适的文本显示位置，避免遮挡。
        :param pt: 点击点 (x, y_adjusted)
        :param text: 坐标文本
        :param existing_boxes: 已有文本区域
        :return: 文本位置 (text_x, text_y) 和文本区域 (x1, y1, x2, y2)
        """
        x, y = pt[0], height - pt[1]  # 转换回 OpenCV 坐标系
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_w, text_h = text_size[0], text_size[1]

        # 可能的文本位置（相对点击点）
        positions = [
            (x + 10, y - 10),  # 右上方
            (x + 10, y + 20),  # 右下方
            (x - text_w - 10, y - 10),  # 左上方
            (x - text_w - 10, y + 20),  # 左下方
        ]

        # 对应的文本区域
        for text_x, text_y in positions:
            text_box = (text_x, text_y - text_h, text_x + text_w, text_y)
            if not check_overlap(text_box, existing_boxes):
                return text_x, text_y, text_box

        # 如果所有位置都重叠，默认使用右上方并稍微偏移
        offset = len(existing_boxes) * 20
        text_x, text_y = x + 10 + offset, y - 10
        text_box = (text_x, text_y - text_h, text_x + text_w, text_y)
        return text_x, text_y, text_box

    def mouse_callback(event, x, y, flags, param):
        """鼠标回调函数，处理点击事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 转换为以左下角为原点的坐标
            y_adjusted = height - y
            points.append((x, y_adjusted))

            # 重新绘制图片
            img_display[:] = img.copy()
            text_boxes.clear()  # 清空旧的文本区域

            for pt in points:
                # 绘制点击点
                cv2.circle(img_display, (pt[0], height - pt[1]), 5, (0, 0, 255), -1)  # 红色圆点

                # 确定文本位置
                text = f"({pt[0]}, {pt[1]})"
                text_x, text_y, text_box = find_text_position(pt, text, text_boxes)
                text_boxes.append(text_box)

                # 显示坐标，颜色为蓝色 (BGR: 0, 0, 255)
                cv2.putText(img_display, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(window_name, img_display)

    # 设置鼠标回调
    cv2.setMouseCallback(window_name, mouse_callback)

    # 显示图片
    cv2.imshow(window_name, img_display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # 按 'n' 切换到下一张图片
            break
        elif key == ord('q'):  # 按 'q' 退出
            cv2.destroyAllWindows()
            return False

    cv2.destroyWindow(window_name)
    return True


def main():
    # 使用 tkinter 选择目录
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 提示用户选择目录
    messagebox.showinfo("提示", "请在弹出的窗口中选择图片所在目录")
    input_dir = filedialog.askdirectory(title="选择图片目录")

    # 检查输入目录是否有效
    if not input_dir:
        messagebox.showerror("错误", "未选择目录，程序将退出！")
        return
    if not os.path.exists(input_dir):
        messagebox.showerror("错误", f"输入目录 {input_dir} 不存在，请检查路径！")
        return

    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        messagebox.showerror("错误", f"在 {input_dir} 中未找到图片")
        return

    # 显示操作说明
    messagebox.showinfo("操作说明",
                        f"找到 {len(image_paths)} 张图片。\n\n"
                        "- 点击图片以标注坐标（左下角为 (0,0)），坐标以蓝色字体显示。\n"
                        "- 请手动记录坐标，程序不会保存任何数据。\n"
                        "- 按 'n' 切换到下一张图片。\n"
                        "- 按 'q' 退出程序。")

    # 逐张处理图片
    for image_path in image_paths:
        print(f"\n正在处理: {image_path}")
        continue_processing = annotate_image(image_path)
        if not continue_processing:
            break

    print("\n程序已退出。")


if __name__ == "__main__":
    main()
