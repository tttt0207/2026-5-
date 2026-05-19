import cv2
import numpy as np
import time


# =========================
# 香橙派参数区
# =========================

# 如果你的摄像头不是 /dev/video0，改这里
CAMERA_DEVICE = "/dev/video0"

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CAMERA_FPS = 15

# 黑色阈值
# 黑框断裂：调大到 100、110
# 白纸阴影被识别成黑色：调小到 70、80
BLACK_V_UPPER = 90

# 你在 Windows 上调好的参数
OUTER_SCALE_X = 1.12
OUTER_SCALE_Y = 1.08

# 是否显示二值化窗口
SHOW_MASK = True


def order_points(pts):
    """
    对四个角点排序：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下

    return rect


def expand_inner_to_outer(inner_points, scale_x=OUTER_SCALE_X, scale_y=OUTER_SCALE_Y):
    """
    根据黑框内孔角点反推 A4 外框角点。
    """
    pts = inner_points.astype(np.float32)
    center = np.mean(pts, axis=0)

    outer = pts.copy()

    for i in range(4):
        outer[i, 0] = center[0] + (pts[i, 0] - center[0]) * scale_x
        outer[i, 1] = center[1] + (pts[i, 1] - center[1]) * scale_y

    return outer


def make_black_mask(raw_image):
    """
    提取黑色区域。
    黑色物体在 mask 中为白色 255。
    白色纸面在 mask 中为黑色 0。
    """
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, BLACK_V_UPPER])

    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((3, 3), np.uint8)

    # 去小噪声
    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # 连接黑框断裂处
    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return black_mask


def detect_rectangle(raw_image):
    """
    检测 A4 黑框。

    当前策略：
    1. 黑框在二值图中是白色边框；
    2. A4 中间白纸在二值图中是黑色区域；
    3. 不直接找外框，因为外框容易和干扰物粘连；
    4. 找黑框内部的“内孔矩形”；
    5. 再由内孔矩形反推 A4 外框。
    """

    black_mask = make_black_mask(raw_image)

    contours, hierarchy = cv2.findContours(
        black_mask.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return None, None, black_mask, None

    hierarchy = hierarchy[0]

    img_h, img_w = black_mask.shape
    img_area = img_h * img_w

    candidates = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # 内孔面积太小，不可能是 A4 中间区域
        if area < img_area * 0.03:
            continue

        # 面积太大也不合理
        if area > img_area * 0.75:
            continue

        # hierarchy[i] = [next, previous, first_child, parent]
        parent = hierarchy[i][3]

        # 只找有父轮廓的轮廓，也就是黑框里的“洞”
        if parent == -1:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        inner_points = order_points(box)

        rect_w = rect[1][0]
        rect_h = rect[1][1]

        if rect_w < 40 or rect_h < 40:
            continue

        short_side = min(rect_w, rect_h)
        long_side = max(rect_w, rect_h)

        if long_side <= 0:
            continue

        ratio = short_side / long_side

        # A4 内孔比例大概接近 0.66
        # 有畸变，所以范围放宽
        if not (0.45 < ratio < 0.90):
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            continue

        roi = black_mask[y:y + h, x:x + w]

        if roi.size == 0:
            continue

        center_roi = roi[
            int(h * 0.25):int(h * 0.75),
            int(w * 0.25):int(w * 0.75)
        ]

        if center_roi.size == 0:
            continue

        # 内孔中心应该基本是黑色，因为白纸在 black_mask 中是 0
        center_black_ratio = cv2.countNonZero(center_roi) / center_roi.size

        # 现在 A4 中间没有图形，所以可以严格一点
        # 后续如果里面有黑色图形，可以调到 0.12 或 0.18
        if center_black_ratio > 0.06:
            continue

        outer_points = expand_inner_to_outer(inner_points)

        # 防止反推出来的外框跑出画面太多
        if np.any(outer_points[:, 0] < -20) or np.any(outer_points[:, 0] > img_w + 20):
            continue

        if np.any(outer_points[:, 1] < -20) or np.any(outer_points[:, 1] > img_h + 20):
            continue

        ratio_score = 1.0 - abs(ratio - 0.66)
        score = area * ratio_score

        candidates.append({
            "score": score,
            "area": area,
            "ratio": ratio,
            "inner_points": inner_points,
            "outer_points": outer_points,
            "center_black_ratio": center_black_ratio
        })

    if len(candidates) == 0:
        return None, None, black_mask, None

    candidates.sort(key=lambda c: c["score"], reverse=True)

    best = candidates[0]

    ordered_points = best["outer_points"]
    inner_points = best["inner_points"]

    cx = int(np.mean(ordered_points[:, 0]))
    cy = int(np.mean(ordered_points[:, 1]))

    img_center_x = raw_image.shape[1] / 2
    img_center_y = raw_image.shape[0] / 2

    offset_x_ratio = (cx - img_center_x) / img_center_x
    offset_y_ratio = -(cy - img_center_y) / img_center_y

    center_offset_ratio = (offset_x_ratio, offset_y_ratio)

    debug_info = {
        "inner_points": inner_points,
        "ratio": best["ratio"],
        "area": best["area"],
        "center_black_ratio": best["center_black_ratio"]
    }

    return ordered_points, center_offset_ratio, black_mask, debug_info


def draw_result(frame, ordered_points, center_offset, debug_info, fps):
    """
    绘制检测结果。
    绿色：反推出的 A4 外框
    蓝色：检测到的内孔矩形
    红点：外框四个角点
    紫点：外框中心点
    """
    show = frame.copy()

    if ordered_points is None:
        cv2.putText(
            show,
            "NO A4 FRAME",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        cv2.putText(
            show,
            "FPS: %.1f" % fps,
            (5, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

        return show

    outer = ordered_points.astype(np.int32)

    # 绿色外框
    cv2.polylines(show, [outer], True, (0, 255, 0), 2)

    # 蓝色内孔
    if debug_info is not None and "inner_points" in debug_info:
        inner = debug_info["inner_points"].astype(np.int32)
        cv2.polylines(show, [inner], True, (255, 0, 0), 1)

    labels = ["TL", "TR", "BR", "BL"]

    for i, p in enumerate(outer):
        x, y = p
        cv2.circle(show, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            show,
            labels[i],
            (x + 3, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

    cx = int(np.mean(outer[:, 0]))
    cy = int(np.mean(outer[:, 1]))

    cv2.circle(show, (cx, cy), 5, (255, 0, 255), -1)

    if center_offset is not None:
        text = "offset: %.2f, %.2f" % (center_offset[0], center_offset[1])
        cv2.putText(
            show,
            text,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1
        )

    if debug_info is not None:
        text2 = "ratio: %.2f center: %.2f" % (
            debug_info["ratio"],
            debug_info["center_black_ratio"]
        )
        cv2.putText(
            show,
            text2,
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

    cv2.putText(
        show,
        "FPS: %.1f" % fps,
        (5, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )

    return show


def open_camera():
    """
    香橙派打开摄像头。
    默认使用 /dev/video0 + V4L2。
    """

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("摄像头打开失败，请检查设备号：", CAMERA_DEVICE)
        print("可以在终端运行：v4l2-ctl --list-devices")
        return None

    # UVC 摄像头一般 MJPG 更流畅
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # 减小缓存，降低延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 打印实际摄像头参数，方便确认是否设置成功
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print("摄像头打开成功")
    print("设备:", CAMERA_DEVICE)
    print("实际分辨率: %.0f x %.0f" % (real_w, real_h))
    print("实际FPS:", real_fps)

    return cap


def main():
    cap = open_camera()

    if cap is None:
        return

    last_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取摄像头失败")
            break

        # 强制缩放，保证算法输入固定
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        ordered_points, center_offset, black_mask, debug_info = detect_rectangle(frame)

        frame_count += 1
        now = time.time()

        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            frame_count = 0
            last_time = now

        result = draw_result(
            frame,
            ordered_points,
            center_offset,
            debug_info,
            fps
        )

        cv2.imshow("result", result)

        if SHOW_MASK:
            cv2.imshow("black_mask", black_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()