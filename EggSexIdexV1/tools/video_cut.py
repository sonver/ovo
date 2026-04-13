import cv2
import os


def split_video_fixed_box(input_path, output_dir,
                          box_x, box_y, box_w, box_h,
                          rows=6, cols=7):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 视频信息: {fps:.2f} FPS, 总帧数 {total_frames}")

    # 每个小框尺寸
    cell_w = box_w // cols
    cell_h = box_h // rows

    # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1  # 从1开始编号
            output_path = os.path.join(video_dir, f"{idx}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (cell_w, cell_h))
            writers.append(writer)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 截取大框
        big_frame = frame[box_y:box_y + box_h, box_x:box_x + box_w]

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                x1 = c * cell_w
                y1 = r * cell_h
                sub_frame = big_frame[y1:y1 + cell_h, x1:x1 + cell_w]
                writers[idx].write(sub_frame)

                # 每秒保存一帧图像
                if frame_idx % int(fps) == 0:
                    second = frame_idx // int(fps) + 1
                    frame_num = f"{second:02d}"
                    img_name = f"{idx + 1}-{frame_num}.jpg"
                    img_path = os.path.join(image_dir, img_name)
                    cv2.imwrite(img_path, sub_frame)

        frame_idx += 1
        if frame_idx % int(fps) == 0:
            print(f"已处理 {frame_idx}/{total_frames} 帧 ({frame_idx // fps}s)...")

    cap.release()
    for w in writers:
        w.release()

    print(f"\n✅ 分割完成！输出目录: {output_dir}")
    print(f"📁 视频保存在: {video_dir}")
    print(f"🖼 图片保存在: {image_dir}")


# ============================
# ⚙️ 使用示例
# ============================
if __name__ == "__main__":
    input_video = "input.mp4"
    output_folder = "output_result"

    # 固定大框坐标（按实际调整）
    box_x, box_y = 100, 200  # 大框左上角坐标
    box_w, box_h = 1400, 1200  # 大框宽高

    split_video_fixed_box(input_video, output_folder, box_x, box_y, box_w, box_h, rows=6, cols=7)
