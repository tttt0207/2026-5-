# -*- coding: utf-8 -*-
"""
board_tracker_vision_best.py

独立版：画板黑框识别 + 画板坐标系建立 + 激光点识别 + 目标轨迹点跟踪 + 输出 dx/dy 差值

适用场景：
1. 画板是白色 A4/UV 纸，四边有 2cm 黑色边框；
2. 摄像头看到画板和激光点；
3. 视觉端只负责计算当前激光点到当前目标点的误差 dx/dy；
4. STM32 端后续根据 dx/dy 做二维 PID。

坐标定义：
- 透视矫正后标准 A4 为 420 x 594 像素；
- 1cm = 20px；
- 画板中心为原点；
- x 右正，y 上正；
- dx = target_x_cm - laser_x_cm；
- dy = target_y_cm - laser_y_cm。

按键：
- q：退出
- r：重新从第 0 个目标点开始
- p：暂停/继续目标点切换

注意：
- 这份代码会按 6 字节协议持续发送 dx/dy 到 STM32。
- 串口帧：0x01 dx_H dx_L dy_H dy_L 0x05。
"""

import cv2
import numpy as np
import time
import os
import math
import struct
import serial


# =========================
# 摄像头参数
# =========================
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 15
TARGET_PROCESS_FPS = 15

USE_GUI = True
SHOW_RESULT = True
SHOW_WARP = True
SHOW_MASK = True


# =========================
# A4 / 画板参数
# =========================
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

WARP_WIDTH = 420
WARP_HEIGHT = 594

CM_PER_PIXEL_X = A4_WIDTH_CM / WARP_WIDTH      # 0.05 cm/px
CM_PER_PIXEL_Y = A4_HEIGHT_CM / WARP_HEIGHT    # 0.05 cm/px
PX_PER_CM = WARP_WIDTH / A4_WIDTH_CM           # 20 px/cm

# 题目要求：画板/目标物四边黑框线宽 2cm
BORDER_CM = 2.0
BORDER_PX_X = int(BORDER_CM * PX_PER_CM)
BORDER_PX_Y = int(BORDER_CM * PX_PER_CM)


# =========================
# 黑框检测参数
# =========================
# 黑色阈值。黑框识别不完整就调大；阴影误识别就调小。
BLACK_V_UPPER = 90

# 黑框大致居中的允许偏差。只是进入绘制模式的门限，真正坐标由透视矫正计算。
# 数值越小，result 里的黄色中心允许框越小，对正要求越严格。
# 0.05 表示允许黑框中心偏离画面半宽/半高的 5%。
CENTER_TOLERANCE_X_RATIO = 0.05
CENTER_TOLERANCE_Y_RATIO = 0.05
CENTER_STABLE_FRAMES = 3

# 内孔反推外框比例。2cm 黑框下，A4内孔约为 17cm x 25.7cm。
# scale = 外尺寸 / 内孔尺寸：x约 21/17=1.235，y约 29.7/25.7=1.156。
OUTER_SCALE_X = A4_WIDTH_CM / (A4_WIDTH_CM - 2 * BORDER_CM)
OUTER_SCALE_Y = A4_HEIGHT_CM / (A4_HEIGHT_CM - 2 * BORDER_CM)


# =========================
# 激光点检测参数
# =========================
# 先用“高亮 + 有颜色饱和度”检测，适合红光/蓝紫光/荧光点。
# 如果你的激光点不明显，需要调 LASER_V_MIN / LASER_S_MIN。
LASER_V_MIN = 150
LASER_S_MIN = 40
LASER_MIN_AREA = 2
LASER_MAX_AREA = 12000

# 如果环境里有很多亮点，可限制只在黑框内部找激光点。
LASER_SEARCH_INNER_ONLY = True


# =========================
# 目标图形参数
# =========================
# 先用这里手动选择。后续可把你图形识别测距得到的 shape_info 传给 build_target_points_from_shape_info()。
TARGET_SHAPE = "square"       # square / triangle / trapezoid / circle / point
#TARGET_SHAPE = "triangle"

# 基本要求：正方形可先用 12cm 调试；三角形基本要求边长 6cm；梯形基本要求 15/20/15cm。
TARGET_SQUARE_SIDE_CM = 12.0
TARGET_TRIANGLE_SIDE_CM = 6.0
TARGET_TRAPEZOID_TOP_CM = 15.0
TARGET_TRAPEZOID_BOTTOM_CM = 20.0
TARGET_TRAPEZOID_HEIGHT_CM = 15.0
TARGET_CIRCLE_DIAMETER_CM = 12.0

# 线段离散步长。越小越平滑，但目标点更多。
LINE_STEP_CM = 0.5
CIRCLE_STEP_DEG = 5.0

# 到达当前目标点的判断阈值。
ARRIVE_DISTANCE_CM = 0.30

# 目标点切换最小间隔，避免一帧跳过很多点。
MIN_SWITCH_INTERVAL = 0.08


# =========================
# 输出/串口参数
# =========================
PRINT_INTERVAL = 0.10

# 串口发送开关：True 表示程序运行时会一直发送 6 字节数据帧。
ENABLE_SERIAL = True
SERIAL_PORT = "/dev/ttyS2"
SERIAL_BAUD = 115200


# =========================
# 基础工具函数
# =========================
def order_points(pts):
    """四点排序：左上、右上、右下、左下。"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def cm_to_warp_px(x_cm, y_cm):
    """画板中心坐标 cm -> 透视矫正图像像素。"""
    x_px = int(round(WARP_WIDTH / 2 + x_cm * PX_PER_CM))
    y_px = int(round(WARP_HEIGHT / 2 - y_cm * PX_PER_CM))
    return x_px, y_px


def warp_px_to_cm(x_px, y_px):
    """透视矫正图像像素 -> 画板中心坐标 cm。"""
    x_cm = (x_px - WARP_WIDTH / 2) / PX_PER_CM
    y_cm = (WARP_HEIGHT / 2 - y_px) / PX_PER_CM
    return x_cm, y_cm


def dist_cm(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def interpolate_line(p1, p2, step_cm=LINE_STEP_CM):
    """两点之间按 step_cm 插入目标点，不包含 p1，包含 p2。"""
    length = dist_cm(p1, p2)
    if length < 1e-6:
        return [p2]

    n = max(1, int(math.ceil(length / step_cm)))
    points = []
    for i in range(1, n + 1):
        t = i / n
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        points.append((x, y))
    return points


def polyline_points(vertices, closed=True, step_cm=LINE_STEP_CM):
    """把多边形顶点离散为连续目标点。"""
    if len(vertices) == 0:
        return []

    points = [vertices[0]]
    end = len(vertices) if closed else len(vertices) - 1

    for i in range(end):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        points.extend(interpolate_line(p1, p2, step_cm))

    return points


# =========================
# 目标轨迹生成
# =========================
def build_square_points(side_cm):
    s = side_cm / 2.0
    vertices = [(-s, s), (s, s), (s, -s), (-s, -s)]
    return polyline_points(vertices, closed=True)


def build_triangle_points(side_cm):
    h = side_cm * math.sqrt(3) / 2.0
    vertices = [
        (0.0, 2.0 * h / 3.0),
        (-side_cm / 2.0, -h / 3.0),
        (side_cm / 2.0, -h / 3.0),
    ]
    return polyline_points(vertices, closed=True)


def build_trapezoid_points(top_cm, bottom_cm, height_cm):
    vertices = [
        (-top_cm / 2.0, height_cm / 2.0),
        (top_cm / 2.0, height_cm / 2.0),
        (bottom_cm / 2.0, -height_cm / 2.0),
        (-bottom_cm / 2.0, -height_cm / 2.0),
    ]
    return polyline_points(vertices, closed=True)


def build_circle_points(diameter_cm):
    r = diameter_cm / 2.0
    points = []
    deg = 0.0
    while deg < 360.0:
        theta = math.radians(deg)
        points.append((r * math.cos(theta), r * math.sin(theta)))
        deg += CIRCLE_STEP_DEG
    if points:
        points.append(points[0])
    return points


def build_point_points():
    return [(0.0, 0.0)]


def build_target_points(shape_name):
    shape_name = shape_name.lower()

    if shape_name == "square":
        return build_square_points(TARGET_SQUARE_SIDE_CM)

    if shape_name == "triangle":
        return build_triangle_points(TARGET_TRIANGLE_SIDE_CM)

    if shape_name == "trapezoid":
        return build_trapezoid_points(
            TARGET_TRAPEZOID_TOP_CM,
            TARGET_TRAPEZOID_BOTTOM_CM,
            TARGET_TRAPEZOID_HEIGHT_CM,
        )

    if shape_name == "circle":
        return build_circle_points(TARGET_CIRCLE_DIAMETER_CM)

    if shape_name == "point":
        return build_point_points()

    print("未知 TARGET_SHAPE，默认画点")
    return build_point_points()


def build_target_points_from_shape_info(shape_info):
    """
    后续和你现有“图形识别+测距”代码合并时，可以用这个函数。

    shape_info 示例：
    {"shape":"square", "side_cm":12.0}
    {"shape":"triangle", "side_cm":8.0}
    {"shape":"circle", "diameter_cm":10.0}
    {"shape":"trapezoid", "top_cm":15.0, "bottom_cm":20.0, "height_cm":15.0}
    """
    if shape_info is None or "shape" not in shape_info:
        return build_target_points(TARGET_SHAPE)

    name = shape_info["shape"]

    if name == "square":
        return build_square_points(float(shape_info.get("side_cm", TARGET_SQUARE_SIDE_CM)))

    if name == "triangle":
        return build_triangle_points(float(shape_info.get("side_cm", TARGET_TRIANGLE_SIDE_CM)))

    if name == "circle":
        return build_circle_points(float(shape_info.get("diameter_cm", TARGET_CIRCLE_DIAMETER_CM)))

    if name == "trapezoid":
        return build_trapezoid_points(
            float(shape_info.get("top_cm", TARGET_TRAPEZOID_TOP_CM)),
            float(shape_info.get("bottom_cm", TARGET_TRAPEZOID_BOTTOM_CM)),
            float(shape_info.get("height_cm", TARGET_TRAPEZOID_HEIGHT_CM)),
        )

    return build_target_points(TARGET_SHAPE)


# =========================
# 黑框检测与透视矫正
# =========================
def make_black_mask(raw_image):
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, BLACK_V_UPPER], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    return mask


def expand_inner_to_outer(inner_points):
    pts = order_points(inner_points)
    center = np.mean(pts, axis=0)
    outer = pts.copy()

    for i in range(4):
        outer[i, 0] = center[0] + (pts[i, 0] - center[0]) * OUTER_SCALE_X
        outer[i, 1] = center[1] + (pts[i, 1] - center[1]) * OUTER_SCALE_Y

    return order_points(outer)


def detect_board_frame(raw_image):
    """
    检测只有黑框的画板。
    返回：outer_points, inner_points, center_offset, black_mask, debug_info。
    """
    black_mask = make_black_mask(raw_image)

    contours, hierarchy = cv2.findContours(
        black_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return None, None, None, black_mask, None

    hierarchy = hierarchy[0]
    img_h, img_w = black_mask.shape
    img_area = img_h * img_w

    candidates = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        parent = hierarchy[i][3]

        # 内孔是有父轮廓的洞。只找面积足够大的洞。
        if parent == -1:
            continue
        if area < img_area * 0.03 or area > img_area * 0.75:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw < 40 or rh < 40:
            continue

        ratio = min(rw, rh) / max(rw, rh)
        # A4内孔约 17/25.7=0.66，放宽。
        if not (0.45 < ratio < 0.90):
            continue

        box = cv2.boxPoints(rect)
        inner_points = order_points(box)
        outer_points = expand_inner_to_outer(inner_points)

        # 外框不能离图像太离谱。
        if np.any(outer_points[:, 0] < -30) or np.any(outer_points[:, 0] > img_w + 30):
            continue
        if np.any(outer_points[:, 1] < -30) or np.any(outer_points[:, 1] > img_h + 30):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = black_mask[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        center_roi = roi[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        if center_roi.size == 0:
            continue

        center_black_ratio = cv2.countNonZero(center_roi) / center_roi.size
        # 画板内部应基本是白色，允许激光点/噪声存在。
        if center_black_ratio > 0.30:
            continue

        ratio_score = 1.0 - abs(ratio - 0.66)
        score = area * ratio_score * (1.0 - center_black_ratio)

        candidates.append({
            "score": score,
            "area": area,
            "ratio": ratio,
            "inner_points": inner_points,
            "outer_points": outer_points,
            "center_black_ratio": center_black_ratio,
        })

    if not candidates:
        return None, None, None, black_mask, None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    outer_points = best["outer_points"]
    inner_points = best["inner_points"]

    cx = float(np.mean(outer_points[:, 0]))
    cy = float(np.mean(outer_points[:, 1]))

    img_center_x = raw_image.shape[1] / 2.0
    img_center_y = raw_image.shape[0] / 2.0

    offset_x_ratio = (cx - img_center_x) / img_center_x
    offset_y_ratio = -(cy - img_center_y) / img_center_y
    center_offset = (offset_x_ratio, offset_y_ratio)

    debug_info = {
        "ratio": best["ratio"],
        "area": best["area"],
        "center_black_ratio": best["center_black_ratio"],
    }

    return outer_points, inner_points, center_offset, black_mask, debug_info


def warp_board_by_inner_points(raw_image, inner_points):
    dst_inner = np.array([
        [BORDER_PX_X, BORDER_PX_Y],
        [WARP_WIDTH - 1 - BORDER_PX_X, BORDER_PX_Y],
        [WARP_WIDTH - 1 - BORDER_PX_X, WARP_HEIGHT - 1 - BORDER_PX_Y],
        [BORDER_PX_X, WARP_HEIGHT - 1 - BORDER_PX_Y],
    ], dtype=np.float32)

    src_inner = order_points(inner_points)
    M = cv2.getPerspectiveTransform(src_inner, dst_inner)
    warped = cv2.warpPerspective(raw_image, M, (WARP_WIDTH, WARP_HEIGHT))
    return warped, M


# =========================
# 激光点检测
# =========================
def detect_laser_point(warped_board):
    """
    在透视矫正后的画板中找激光点。

    这一版保留最开始稳定的“高亮 + 饱和度”思路，
    只做两个小修正：
    1. 放宽 LASER_MAX_AREA，避免三角形/近距离时大光斑被过滤；
    2. laser_mask 只显示最终选中的那一块，方便判断 board_warp 里的绿色点为什么有没有。

    返回：laser_px, laser_cm, selected_mask。
    """
    hsv = cv2.cvtColor(warped_board, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 高亮且有一定饱和度。
    mask1 = cv2.inRange(v, LASER_V_MIN, 255)
    mask2 = cv2.inRange(s, LASER_S_MIN, 255)
    laser_mask = cv2.bitwise_and(mask1, mask2)

    # 限制只在 A4 黑框内部找，避免黑框外部亮点干扰。
    if LASER_SEARCH_INNER_ONLY:
        inner_mask = np.zeros_like(laser_mask)
        x1 = BORDER_PX_X + 5
        y1 = BORDER_PX_Y + 5
        x2 = WARP_WIDTH - BORDER_PX_X - 5
        y2 = WARP_HEIGHT - BORDER_PX_Y - 5
        cv2.rectangle(inner_mask, (x1, y1), (x2, y2), 255, -1)
        laser_mask = cv2.bitwise_and(laser_mask, inner_mask)

    kernel3 = np.ones((3, 3), np.uint8)
    laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    laser_mask = cv2.morphologyEx(laser_mask, cv2.MORPH_CLOSE, kernel3, iterations=2)

    contours, _ = cv2.findContours(laser_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    selected_mask = np.zeros_like(laser_mask)

    if len(contours) == 0:
        return None, None, selected_mask

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < LASER_MIN_AREA or area > LASER_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        ratio = min(w, h) / max(w, h)
        if ratio < 0.20:
            continue

        candidates.append((area, c))

    if not candidates:
        # 这里返回空 selected_mask。
        # 如果 raw mask 里明明有光斑但这里没有绿点，通常就是面积阈值/形状阈值过滤了。
        return None, None, selected_mask

    candidates.sort(key=lambda item: item[0], reverse=True)
    contour = candidates[0][1]

    cv2.drawContours(selected_mask, [contour], -1, 255, -1)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

    laser_cm = warp_px_to_cm(cx, cy)
    return (cx, cy), laser_cm, selected_mask


# =========================
# 显示与输出
# =========================
def draw_result(frame, outer_points, inner_points, center_offset, center_flag, stable_count, fps):
    show = frame.copy()
    h, w = show.shape[:2]

    # 画面中心十字和允许误差框。
    cx = w // 2
    cy = h // 2
    tol_x = int(w / 2 * CENTER_TOLERANCE_X_RATIO)
    tol_y = int(h / 2 * CENTER_TOLERANCE_Y_RATIO)

    cv2.line(show, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
    cv2.line(show, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
    cv2.rectangle(show, (cx - tol_x, cy - tol_y), (cx + tol_x, cy + tol_y), (0, 255, 255), 1)

    if outer_points is None:
        cv2.putText(show, "NO BOARD FRAME", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(show, "FPS: %.1f" % fps, (5, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return show

    outer = outer_points.astype(np.int32)
    cv2.polylines(show, [outer], True, (0, 255, 0), 2)

    if inner_points is not None:
        inner = inner_points.astype(np.int32)
        cv2.polylines(show, [inner], True, (255, 0, 0), 1)

    bx = int(np.mean(outer[:, 0]))
    by = int(np.mean(outer[:, 1]))
    cv2.circle(show, (bx, by), 5, (255, 0, 255), -1)

    if center_offset is not None:
        cv2.putText(show, "offset: %.2f %.2f" % center_offset, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    cv2.putText(show, "CENTER_FLAG: %d stable:%d/%d" % (center_flag, stable_count, CENTER_STABLE_FRAMES),
                (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if center_flag else (0, 0, 255), 2)

    cv2.putText(show, "FPS: %.1f" % fps, (5, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return show


def draw_warp(warped, target_points, current_index, laser_px, laser_cm, target_cm, dx, dy, board_valid, laser_valid, drawing_done):
    show = warped.copy()

    # 画中心坐标轴。
    cv2.line(show, (0, WARP_HEIGHT // 2), (WARP_WIDTH, WARP_HEIGHT // 2), (200, 200, 200), 1)
    cv2.line(show, (WARP_WIDTH // 2, 0), (WARP_WIDTH // 2, WARP_HEIGHT), (200, 200, 200), 1)

    # 画内部有效区域。
    cv2.rectangle(
        show,
        (BORDER_PX_X, BORDER_PX_Y),
        (WARP_WIDTH - BORDER_PX_X, WARP_HEIGHT - BORDER_PX_Y),
        (0, 255, 255), 1
    )

    # 画整条目标轨迹。
    if target_points:
        pts_px = np.array([cm_to_warp_px(x, y) for x, y in target_points], dtype=np.int32)
        if len(pts_px) >= 2:
            cv2.polylines(show, [pts_px], False, (255, 0, 255), 1)

        # 当前目标点。
        if 0 <= current_index < len(target_points):
            tx, ty = cm_to_warp_px(*target_points[current_index])
            cv2.circle(show, (tx, ty), 5, (0, 0, 255), -1)
            cv2.putText(show, "T%d" % current_index, (tx + 5, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 画激光点。
    if laser_px is not None:
        cv2.circle(show, laser_px, 6, (0, 255, 0), 2)
        cv2.circle(show, laser_px, 2, (0, 255, 0), -1)

    if target_cm is not None:
        cv2.putText(show, "target: %.1f %.1f cm" % target_cm, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    if laser_cm is not None:
        cv2.putText(show, "laser : %.1f %.1f cm" % laser_cm, (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if dx is not None and dy is not None:
        cv2.putText(show, "dx dy : %.1f %.1f cm" % (dx, dy), (8, 79),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    if drawing_done:
        status = "DONE"
        color = (0, 255, 0)
    elif not board_valid:
        status = "NO BOARD"
        color = (0, 0, 255)
    elif not laser_valid:
        status = "NO LASER"
        color = (0, 0, 255)
    else:
        status = "TRACKING"
        color = (0, 255, 255)

    cv2.putText(show, status, (8, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return show


def build_packet(dx_cm, dy_cm):
    """
    按题目要求生成 6 字节串口数据帧：
    [0] 帧头 0x01
    [1] dx * 100 的高八位
    [2] dx * 100 的低八位
    [3] dy * 100 的高八位
    [4] dy * 100 的低八位
    [5] 帧尾 0x05

    dx/dy 单位：cm。
    先保留两位小数，再乘 100，转为 int16。
    负数用 int16 补码发送。
    """
    dx_i = int(round(round(dx_cm, 2) * 100))
    dy_i = int(round(round(dy_cm, 2) * 100))

    dx_i = max(-32768, min(32767, dx_i))
    dy_i = max(-32768, min(32767, dy_i))

    dx_u = dx_i & 0xFFFF
    dy_u = dy_i & 0xFFFF

    return bytes([
        0x01,
        (dx_u >> 8) & 0xFF,
        dx_u & 0xFF,
        (dy_u >> 8) & 0xFF,
        dy_u & 0xFF,
        0x05,
    ])


def open_serial_port():
    if not ENABLE_SERIAL:
        return None

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print("串口已打开: %s, %d" % (SERIAL_PORT, SERIAL_BAUD))
        return ser
    except Exception as e:
        print("串口打开失败，不发送串口数据:", e)
        return None


def print_control_info(dx, dy, target_index, total_targets, flag, laser_cm, target_cm, packet):
    if dx is None or dy is None:
        print("flag=%d  target=%d/%d  dx=None  dy=None" % (flag, target_index, total_targets))
        return

    lx, ly = laser_cm
    tx, ty = target_cm
    print(
        "flag=%d  target=%d/%d  laser=(%.1f, %.1f)cm  target=(%.1f, %.1f)cm  dx=%.1fcm  dy=%.1fcm  packet=%s" % (
            flag, target_index, total_targets, lx, ly, tx, ty, dx, dy, packet.hex(" ")
        )
    )


# =========================
# 摄像头
# =========================
def open_camera():
    print("正在打开摄像头: cv2.VideoCapture(%d)" % CAMERA_ID)
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("摄像头打开失败，请检查 /dev/video* 或是否被其它程序占用")
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok = False
    frame = None
    for _ in range(30):
        ret, temp = cap.read()
        if ret and temp is not None:
            ok = True
            frame = temp
            break
        time.sleep(0.1)

    if not ok:
        print("摄像头已打开，但读取画面失败")
        cap.release()
        return None

    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print("摄像头成功: %.0f x %.0f, FPS=%.1f" % (real_w, real_h, real_fps))
    return cap


# =========================
# 主程序
# =========================
def main():
    target_points = build_target_points(TARGET_SHAPE)
    print("TARGET_SHAPE:", TARGET_SHAPE)
    print("目标点数量:", len(target_points))
    print("A4坐标: 中心为原点，x右正，y上正，单位cm")
    print("黑框线宽: %.1fcm，透视矫正尺寸: %d x %d，比例: %.1fpx/cm" % (
        BORDER_CM, WARP_WIDTH, WARP_HEIGHT, PX_PER_CM
    ))

    cap = open_camera()
    if cap is None:
        return

    ser = open_serial_port()

    gui_enabled = USE_GUI and (os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None)
    if USE_GUI and not gui_enabled:
        print("未检测到 DISPLAY/WAYLAND_DISPLAY，自动关闭窗口显示")

    if gui_enabled:
        if SHOW_RESULT:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        if SHOW_WARP:
            cv2.namedWindow("board_warp", cv2.WINDOW_NORMAL)
        if SHOW_MASK:
            cv2.namedWindow("laser_mask", cv2.WINDOW_NORMAL)

    current_index = 0
    paused = False
    drawing_done = False
    stable_center_count = 0

    frame_count = 0
    fps = 0.0
    fps_start = time.time()
    last_print = 0.0
    last_switch = 0.0

    target_period = 1.0 / TARGET_PROCESS_FPS

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print("读取摄像头失败")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            outer_points, inner_points, center_offset, black_mask, debug_info = detect_board_frame(frame)

            board_detected = outer_points is not None and inner_points is not None and center_offset is not None
            center_flag = 0

            if board_detected:
                ox, oy = center_offset
                if abs(ox) <= CENTER_TOLERANCE_X_RATIO and abs(oy) <= CENTER_TOLERANCE_Y_RATIO:
                    stable_center_count += 1
                else:
                    stable_center_count = 0

                if stable_center_count >= CENTER_STABLE_FRAMES:
                    center_flag = 1
            else:
                stable_center_count = 0

            board_valid = board_detected and center_flag == 1

            warped = None
            laser_px = None
            laser_cm = None
            laser_mask = None
            target_cm = None
            dx = None
            dy = None
            laser_valid = False
            flag = 0

            if board_detected:
                warped, _ = warp_board_by_inner_points(frame, inner_points)
                laser_px, laser_cm, laser_mask = detect_laser_point(warped)
                laser_valid = laser_cm is not None

                if not board_valid:
                    flag = 1     # 识别到画板，但未居中稳定
                elif not laser_valid:
                    flag = 2     # 画板有效，但没找到激光点
                elif drawing_done:
                    flag = 5     # 绘制完成
                else:
                    if current_index < len(target_points):
                        target_cm = target_points[current_index]
                        dx = target_cm[0] - laser_cm[0]
                        dy = target_cm[1] - laser_cm[1]
                        err = math.hypot(dx, dy)

                        flag = 3  # 正在跟踪

                        now_switch = time.time()
                        if (not paused) and err <= ARRIVE_DISTANCE_CM and (now_switch - last_switch >= MIN_SWITCH_INTERVAL):
                            current_index += 1
                            last_switch = now_switch
                            flag = 4  # 当前目标点到达

                            if current_index >= len(target_points):
                                drawing_done = True
                                current_index = len(target_points) - 1
                                flag = 5
                    else:
                        drawing_done = True
                        flag = 5

                # 如果刚切换点/完成后 target_cm 为空，补当前目标点用于显示。
                if target_cm is None and target_points and current_index < len(target_points):
                    target_cm = target_points[current_index]

            else:
                flag = 0         # 未识别到画板

            # 构造并发送 6 字节串口数据帧。dx/dy 为空时发 0。
            if dx is None or dy is None:
                packet = build_packet(0.0, 0.0)
            else:
                packet = build_packet(dx, dy)

            if ser is not None:
                ser.write(packet)

            # 终端打印。
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print_control_info(
                    dx, dy, current_index, len(target_points), flag,
                    laser_cm if laser_cm is not None else (0.0, 0.0),
                    target_cm if target_cm is not None else (0.0, 0.0),
                    packet
                )
                last_print = now

            # 显示。
            if gui_enabled and SHOW_RESULT:
                result_show = draw_result(frame, outer_points, inner_points, center_offset, center_flag, stable_center_count, fps)
                cv2.imshow("result", result_show)

            if gui_enabled and SHOW_WARP and warped is not None:
                warp_show = draw_warp(
                    warped, target_points, current_index, laser_px, laser_cm,
                    target_cm, dx, dy, board_valid, laser_valid, drawing_done
                )
                cv2.imshow("board_warp", warp_show)

            if gui_enabled and SHOW_MASK and laser_mask is not None:
                cv2.imshow("laser_mask", laser_mask)

            frame_count += 1
            if now - fps_start >= 1.0:
                fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            elapsed = time.time() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if gui_enabled:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    current_index = 0
                    drawing_done = False
                    last_switch = time.time()
                    print("已重置目标点")
                elif key == ord("p"):
                    paused = not paused
                    print("暂停切换" if paused else "继续切换")

    except KeyboardInterrupt:
        print("程序被手动停止")

    finally:
        if 'ser' in locals() and ser is not None:
            ser.close()
        cap.release()
        if gui_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
