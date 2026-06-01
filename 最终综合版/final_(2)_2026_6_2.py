import cv2
import numpy as np
import time
import os
import math
import serial
import threading


# =========================
# 香橙派 / Linux 参数区
# =========================

# 优先使用 /dev/video0。
# 如果你的摄像头实际是 /dev/video1，就改成 "/dev/video1"。
CAMERA_DEVICE = "/dev/video0"

# 备用编号，CAMERA_DEVICE 打不开时会尝试这个编号。
CAMERA_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAMERA_FPS = 15
TARGET_PROCESS_FPS = 15

# 黑色阈值
# 黑色图形/黑框识别不完整：调大，例如 100、110
# 白纸阴影也变黑：调小，例如 70、80
BLACK_V_UPPER = 90

# 当前黑框参数
# 注意：这里不能引用后面才定义的 A4_WIDTH_CM/BORDER_CM。
# 2cm 黑框下，内孔约 17cm x 25.7cm，所以外框/内孔比例约为下面两个值。
OUTER_SCALE_X = 21.0 / (21.0 - 2 * 2.0)      # 约 1.235
OUTER_SCALE_Y = 29.7 / (29.7 - 2 * 2.0)      # 约 1.156

# 只显示这三个窗口
# 如果你是桌面/PyCharm运行，可以保持 True。
# 如果以后用 systemd 后台自启动，没有显示器，可以把 USE_GUI 改成 False。
USE_GUI = True
SHOW_RESULT = True
SHOW_SHAPE = True
SHOW_SHAPE_BINARY = True


# =========================
# A4 标准尺寸参数
# =========================

A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

WARP_WIDTH = 420
WARP_HEIGHT = 594

CM_PER_PIXEL_X = A4_WIDTH_CM / WARP_WIDTH      # 0.05 cm/px
CM_PER_PIXEL_Y = A4_HEIGHT_CM / WARP_HEIGHT    # 0.05 cm/px

# A4 外黑框宽度：按你现在反馈“所有黑色边框都是 1cm”设置为 1cm。
# 如果你后面又换回题目要求的 2cm A4 外黑框，只需要把这里改成 2.0。
BORDER_CM = 2.0
BORDER_PX_X = int(BORDER_CM / CM_PER_PIXEL_X)
BORDER_PX_Y = int(BORDER_CM / CM_PER_PIXEL_Y)

# 图形线条：1cm。尺寸测量按黑线“中心线”计算。
FIGURE_LINE_WIDTH_CM = 1.0
FIGURE_LINE_WIDTH_PX = int(FIGURE_LINE_WIDTH_CM / CM_PER_PIXEL_X)  # 20px


# =========================
# 基础工具函数
# =========================

def order_points(pts):
    """
    四点排序：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def distance(p1, p2):
    return np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32))


def angle_cos(p0, p1, p2):
    """
    计算 p0-p1-p2 夹角余弦绝对值。
    越接近 0，越接近 90 度。
    """
    v1 = np.array(p0, dtype=np.float32) - np.array(p1, dtype=np.float32)
    v2 = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 1.0

    return abs(np.dot(v1, v2) / norm)


def expand_inner_to_outer(inner_points, scale_x=OUTER_SCALE_X, scale_y=OUTER_SCALE_Y):
    """
    根据黑框内孔角点反推 A4 外框角点。
    主要用于 result 窗口画绿色外框。
    """
    pts = inner_points.astype(np.float32)
    center = np.mean(pts, axis=0)

    outer = pts.copy()

    for i in range(4):
        outer[i, 0] = center[0] + (pts[i, 0] - center[0]) * scale_x
        outer[i, 1] = center[1] + (pts[i, 1] - center[1]) * scale_y

    return outer


# =========================
# 黑框检测
# =========================

def make_black_mask(raw_image):
    """
    提取黑色区域：
    黑色区域在 mask 中为 255
    白色区域在 mask 中为 0
    """
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, BLACK_V_UPPER])

    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((3, 3), np.uint8)

    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return black_mask


def detect_rectangle(raw_image, require_empty_inside=False):
    """
    检测 A4 黑框。

    require_empty_inside=False：允许 A4 内部存在黑色图形，用于目标物识别。
    require_empty_inside=True ：要求 A4 内部基本无图形，用于发挥 24/25 搜索真正画板，
                                避免把带图形的目标物误判成画板。

    思路：
    1. 黑框在二值图中是白色环。
    2. 中间白纸区域在二值图中是黑色洞。
    3. 找黑框内部的“洞”。
    4. 由内孔反推 A4 外框。
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

        if area < img_area * 0.03:
            continue

        if area > img_area * 0.75:
            continue

        parent = hierarchy[i][3]

        # 只要有父轮廓的“洞”
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

        # A4 内孔比例约 0.66，放宽一些
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

        center_black_ratio = cv2.countNonZero(center_roi) / center_roi.size

        if require_empty_inside:
            # 搜索画板时使用：真正画板内部应几乎没有黑色图形。
            # 这样可以排除带黑色图形的目标物 A4，也能减少光照误判。
            if center_black_ratio > EMPTY_BOARD_CENTER_BLACK_RATIO_MAX:
                continue
        else:
            # 识别目标物时使用：目标物内部可能有黑色图形，所以必须放宽。
            if center_black_ratio > 0.65:
                continue

        outer_points = expand_inner_to_outer(inner_points)

        if np.any(outer_points[:, 0] < -20) or np.any(outer_points[:, 0] > img_w + 20):
            continue

        if np.any(outer_points[:, 1] < -20) or np.any(outer_points[:, 1] > img_h + 20):
            continue

        ratio_score = 1.0 - abs(ratio - 0.66)

        center_score = max(0.3, 1.0 - center_black_ratio)
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


# =========================
# A4 透视矫正
# =========================

def warp_a4_by_inner_points(raw_image, inner_points):
    """
    使用黑框内孔四角点进行透视矫正。

    标准 A4：
    420px x 594px
    1cm = 20px
    黑边框 2cm = 40px
    """

    dst_inner_points = np.array([
        [BORDER_PX_X, BORDER_PX_Y],
        [WARP_WIDTH - 1 - BORDER_PX_X, BORDER_PX_Y],
        [WARP_WIDTH - 1 - BORDER_PX_X, WARP_HEIGHT - 1 - BORDER_PX_Y],
        [BORDER_PX_X, WARP_HEIGHT - 1 - BORDER_PX_Y]
    ], dtype="float32")

    src_inner_points = order_points(inner_points.astype("float32"))

    M = cv2.getPerspectiveTransform(src_inner_points, dst_inner_points)
    warped = cv2.warpPerspective(raw_image, M, (WARP_WIDTH, WARP_HEIGHT))

    return warped, M


def warp_a4_by_outer_points(raw_image, ordered_points):
    """
    备用：使用外框四角点进行透视矫正。
    """
    dst_points = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1]
    ], dtype="float32")

    src_points = ordered_points.astype("float32")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(raw_image, M, (WARP_WIDTH, WARP_HEIGHT))

    return warped, M


# =========================
# 图形识别：裁剪区域和四边形分类
# =========================

def get_inner_region(warped_a4):
    """
    只裁剪 A4 黑框内部区域。
    margin 跟随 BORDER_CM 自动变化，并额外多留 15px，
    避免 A4 外黑框残留进入 shape_binary 干扰内部图形。
    """
    margin_x = BORDER_PX_X + 15
    margin_y = BORDER_PX_Y + 15

    x1 = margin_x
    y1 = margin_y
    x2 = WARP_WIDTH - margin_x
    y2 = WARP_HEIGHT - margin_y

    inner = warped_a4[y1:y2, x1:x2]

    return inner, x1, y1


def classify_quad(pts, contour=None):
    """
    四边形分类：正方形 or 梯形

    这版重点解决：
    梯形容易被识别成正方形的问题。

    修改思路：
    1. 先看“上底/下底差异”，差异明显就优先判为梯形；
    2. 正方形判断不再使用很宽的兜底阈值；
    3. 正方形必须同时满足：四边接近相等、对角线接近相等、角度接近 90 度。
    """
    pts = order_points(pts.astype(np.float32))
    tl, tr, br, bl = pts

    top = distance(tl, tr)
    bottom = distance(bl, br)
    left = distance(tl, bl)
    right = distance(tr, br)

    sides = [top, right, bottom, left]
    max_side = max(sides)
    min_side = min(sides)

    if max_side <= 0:
        return "trapezoid", {}

    side_ratio = min_side / max_side

    diag1 = distance(tl, br)
    diag2 = distance(tr, bl)

    if max(diag1, diag2) > 0:
        diag_ratio = min(diag1, diag2) / max(diag1, diag2)
    else:
        diag_ratio = 0

    cos1 = angle_cos(bl, tl, tr)
    cos2 = angle_cos(tl, tr, br)
    cos3 = angle_cos(tr, br, bl)
    cos4 = angle_cos(br, bl, tl)

    max_cos = max(cos1, cos2, cos3, cos4)

    rect_ratio = 0
    if contour is not None:
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]

        if rw > 0 and rh > 0:
            rect_ratio = min(rw, rh) / max(rw, rh)

    if max(top, bottom) > 0:
        top_bottom_diff = abs(top - bottom) / max(top, bottom)
    else:
        top_bottom_diff = 1

    if max(left, right) > 0:
        left_right_diff = abs(left - right) / max(left, right)
    else:
        left_right_diff = 1

    height_px = (left + right) / 2.0

    # 梯形特征：上底和下底差异明显。
    # 这里要放在正方形前面，否则梯形容易被宽松的正方形条件抢走。
    # 如果你的正方形在极端角度下偶尔变梯形，可以把 0.16 调到 0.20。
    if top_bottom_diff > 0.16:
        return "trapezoid", {
            "top_px": top,
            "bottom_px": bottom,
            "height_px": height_px,

            "top_cm": top * CM_PER_PIXEL_X,
            "bottom_cm": bottom * CM_PER_PIXEL_X,
            "height_cm": height_px * CM_PER_PIXEL_Y
        }

    # 正方形条件收紧：
    # 原来 side_ratio > 0.55、rect_ratio > 0.70 太宽，梯形很容易混进来。
    is_square = (
        side_ratio > 0.78 and
        diag_ratio > 0.78 and
        max_cos < 0.55 and
        rect_ratio > 0.76 and
        top_bottom_diff < 0.16 and
        left_right_diff < 0.25
    )

    if is_square:
        side_px = sum(sides) / 4.0

        return "square", {
            "side_px": side_px,
            "side_cm": side_px * CM_PER_PIXEL_X
        }

    # 其余四边形按梯形处理
    return "trapezoid", {
        "top_px": top,
        "bottom_px": bottom,
        "height_px": height_px,

        "top_cm": top * CM_PER_PIXEL_X,
        "bottom_cm": bottom * CM_PER_PIXEL_X,
        "height_cm": height_px * CM_PER_PIXEL_Y
    }

def approx_polygon_by_target_vertices(contour, target_vertices):
    """
    多次尝试不同 epsilon，尽量把轮廓拟合成指定顶点数。
    使用凸包可以减少边缘毛刺导致的多余顶点。
    """
    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, True)

    if perimeter <= 0:
        return None

    eps_list = [
        0.012, 0.015, 0.018, 0.02, 0.025,
        0.03, 0.035, 0.04, 0.045, 0.05,
        0.06, 0.07, 0.08, 0.09
    ]

    for eps in eps_list:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)

        if len(approx) == target_vertices:
            return approx

    return None


def is_reasonable_triangle(pts, contour=None):
    """
    判断三角形是否合理。
    用于 tri_ok 兜底。
    """
    if pts is None or len(pts) != 3:
        return False

    pts = pts.reshape(3, 2).astype(np.float32)

    s1 = distance(pts[0], pts[1])
    s2 = distance(pts[1], pts[2])
    s3 = distance(pts[2], pts[0])

    sides = [s1, s2, s3]
    max_side = max(sides)
    min_side = min(sides)

    if max_side <= 0:
        return False

    side_ratio = min_side / max_side

    # 等边三角形，畸变后放宽到 0.55
    if side_ratio < 0.55:
        return False

    tri_area = cv2.contourArea(pts.astype(np.float32))

    if tri_area < 300:
        return False

    # 如果传入 contour，判断拟合三角形面积和原轮廓面积是否接近
    # 避免把圆形硬拟合成三角形
    if contour is not None:
        contour_area = cv2.contourArea(contour)

        if tri_area <= 0:
            return False

        area_ratio = contour_area / tri_area

        if area_ratio < 0.60 or area_ratio > 1.25:
            return False

    return True


def is_reasonable_square(pts, contour=None):
    """
    判断四边形是否更像正方形。
    这个函数用于 quad_ok 兜底。

    注意：
    这里不能太宽，否则梯形会提前进入“正方形兜底”。
    最终 square/trapezoid 仍由 classify_quad() 再判断一次。
    """
    if pts is None or len(pts) != 4:
        return False

    pts = order_points(pts.reshape(4, 2).astype(np.float32))
    tl, tr, br, bl = pts

    top = distance(tl, tr)
    bottom = distance(bl, br)
    left = distance(tl, bl)
    right = distance(tr, br)

    sides = [top, right, bottom, left]
    max_side = max(sides)
    min_side = min(sides)

    if max_side <= 0:
        return False

    side_ratio = min_side / max_side

    if max(top, bottom) > 0:
        top_bottom_diff = abs(top - bottom) / max(top, bottom)
    else:
        top_bottom_diff = 1

    # 上下底差异明显，优先认为不是正方形。
    if top_bottom_diff > 0.18:
        return False

    diag1 = distance(tl, br)
    diag2 = distance(tr, bl)

    if max(diag1, diag2) > 0:
        diag_ratio = min(diag1, diag2) / max(diag1, diag2)
    else:
        diag_ratio = 0

    cos1 = angle_cos(bl, tl, tr)
    cos2 = angle_cos(tl, tr, br)
    cos3 = angle_cos(tr, br, bl)
    cos4 = angle_cos(br, bl, tl)

    max_cos = max(cos1, cos2, cos3, cos4)

    rect_ratio = 0

    if contour is not None:
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]

        if rw > 0 and rh > 0:
            rect_ratio = min(rw, rh) / max(rw, rh)

    if side_ratio > 0.78 and diag_ratio > 0.78 and max_cos < 0.55:
        return True

    if rect_ratio > 0.82 and side_ratio > 0.70 and max_cos < 0.65:
        return True

    return False

# =========================
# 图形识别：中心线测量
# =========================

def find_shape_outer_inner_contour(binary):
    """
    找空心图形的外轮廓和内轮廓。

    binary 中：
    白色 = 黑色图形线条
    黑色 = 白纸背景 / 空心内部

    返回：
    outer_contour：黑色线条外边缘
    inner_contour：黑色线条内边缘，也就是空心洞的边界
    """

    contours, hierarchy = cv2.findContours(
        binary.copy(),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return None, None

    hierarchy = hierarchy[0]

    candidates = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        parent = hierarchy[i][3]
        child = hierarchy[i][2]

        # 只找最外层轮廓，并且它内部必须有洞
        if parent != -1:
            continue

        if child == -1:
            continue

        if area < 200:
            continue

        child_area = cv2.contourArea(contours[child])

        if child_area < 50:
            continue

        candidates.append((area, contour, contours[child]))

    if len(candidates) == 0:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][1], candidates[0][2]


def find_largest_external_contour(binary):
    """
    兜底：如果没有找到内轮廓，就找最大外轮廓。
    """
    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]

def find_combined_shape_contour(binary):
    """
    合并所有图形黑色轮廓点。
    解决竖直方向时空心正方形/三角形边框分裂成多个轮廓，导致完全不画框的问题。
    """

    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    img_h, img_w = binary.shape
    img_area = img_h * img_w

    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        # 过滤小噪声
        if area < 30:
            continue

        if w < 5 or h < 5:
            continue

        # 过滤贴边残留，避免 A4 黑框干扰
        if x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2:
            continue

        valid_contours.append(c)

    if len(valid_contours) == 0:
        return None

    # 把所有有效轮廓点合并
    all_points = np.vstack(valid_contours)

    x, y, w, h = cv2.boundingRect(all_points)

    # 整体区域太小，认为不是有效图形
    if w < 40 or h < 40:
        return None

    # 整体区域太大，可能是黑框残留
    if w * h > img_area * 0.70:
        return None

    # 用凸包形成完整外轮廓
    hull = cv2.convexHull(all_points)

    return hull

def find_largest_reasonable_contour(binary):
    """
    更稳的兜底外轮廓查找：
    不要求一定有内孔，只找面积合适、不是整张图边框残留的最大轮廓。
    作用：当空心图形在竖直方向轻微断环，RETR_CCOMP 找不到内孔时，仍然能进入三角形/正方形分类。
    """
    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    img_h, img_w = binary.shape
    img_area = img_h * img_w

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 200:
            continue

        if area > img_area * 0.60:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # 过滤贴边的 A4 黑框残留
        if x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2:
            continue

        candidates.append((area, c))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_shape_contour_strong(binary):
    """
    强化版图形轮廓查找。

    关键思路：
    1. 不强制要求空心图形必须先形成 inner_contour；
    2. 用更强的纵向/横向闭运算补竖直方向断边；
    3. 找最大合理外轮廓，保证先能画出框、进入分类。

    返回：
    contour：找到的外轮廓
    work：增强后的二值图，用于 shape_binary 调试显示
    """
    work = binary.copy()

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))

    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel3, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel5, iterations=1)

    # 只轻微膨胀一次，避免把边框粘成整块，但能连接细小缺口。
    work = cv2.dilate(work, kernel3, iterations=1)

    contours, _ = cv2.findContours(
        work.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, work

    img_h, img_w = work.shape
    img_area = img_h * img_w

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 300:
            continue

        if area > img_area * 0.65:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 过滤贴边的 A4 黑框残留。
        if x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2:
            continue

        if w < 40 or h < 40:
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]

        if rw <= 0 or rh <= 0:
            continue

        ratio = min(rw, rh) / max(rw, rh)

        # 过滤过细的噪声线条。
        if ratio < 0.35:
            continue

        score = area * ratio
        candidates.append((score, area, contour))

    if len(candidates) == 0:
        return None, work

    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][2], work


def get_triangle_side_centerline(outer_contour, inner_contour):
    """
    等边三角形中心线边长：
    取外三角形边长和内三角形边长的平均值。
    """

    if outer_contour is None or inner_contour is None:
        return None

    outer_perimeter = cv2.arcLength(outer_contour, True)
    inner_perimeter = cv2.arcLength(inner_contour, True)

    outer_approx = cv2.approxPolyDP(outer_contour, 0.04 * outer_perimeter, True)
    inner_approx = cv2.approxPolyDP(inner_contour, 0.04 * inner_perimeter, True)

    if len(outer_approx) != 3 or len(inner_approx) != 3:
        return None

    outer_pts = outer_approx.reshape(3, 2).astype(np.float32)
    inner_pts = inner_approx.reshape(3, 2).astype(np.float32)

    outer_sides = [
        distance(outer_pts[0], outer_pts[1]),
        distance(outer_pts[1], outer_pts[2]),
        distance(outer_pts[2], outer_pts[0])
    ]

    inner_sides = [
        distance(inner_pts[0], inner_pts[1]),
        distance(inner_pts[1], inner_pts[2]),
        distance(inner_pts[2], inner_pts[0])
    ]

    outer_avg = sum(outer_sides) / 3.0
    inner_avg = sum(inner_sides) / 3.0

    return (outer_avg + inner_avg) / 2.0

def get_triangle_side_by_min_enclosing_centerline(outer_contour, inner_contour):
    """
    三角形兜底测量：
    当 approxPolyDP 拟合不出 3 个点时，用 minEnclosingTriangle 测中心线边长。
    """
    if outer_contour is None:
        return None

    ret_outer, outer_tri = cv2.minEnclosingTriangle(outer_contour)

    if outer_tri is None:
        return None

    outer_pts = outer_tri.reshape(3, 2).astype(np.float32)

    outer_sides = [
        distance(outer_pts[0], outer_pts[1]),
        distance(outer_pts[1], outer_pts[2]),
        distance(outer_pts[2], outer_pts[0])
    ]

    outer_avg = sum(outer_sides) / 3.0

    if inner_contour is not None:
        ret_inner, inner_tri = cv2.minEnclosingTriangle(inner_contour)

        if inner_tri is not None:
            inner_pts = inner_tri.reshape(3, 2).astype(np.float32)

            inner_sides = [
                distance(inner_pts[0], inner_pts[1]),
                distance(inner_pts[1], inner_pts[2]),
                distance(inner_pts[2], inner_pts[0])
            ]

            inner_avg = sum(inner_sides) / 3.0

            return (outer_avg + inner_avg) / 2.0

    return max(0, outer_avg - FIGURE_LINE_WIDTH_PX)

def get_circle_diameter_centerline(outer_contour, inner_contour):
    """
    圆形中心线直径：
    取外圆直径和内圆直径的平均值。
    """

    if outer_contour is None or inner_contour is None:
        return None

    (_, _), outer_radius = cv2.minEnclosingCircle(outer_contour)
    (_, _), inner_radius = cv2.minEnclosingCircle(inner_contour)

    outer_diameter = outer_radius * 2.0
    inner_diameter = inner_radius * 2.0

    return (outer_diameter + inner_diameter) / 2.0


def get_quad_centerline_info(outer_contour, inner_contour):
    """
    四边形中心线尺寸：
    对正方形：边长 = 外边长和内边长平均
    对梯形：上底、下底、高分别取外轮廓和内轮廓平均
    """

    if outer_contour is None or inner_contour is None:
        return None

    outer_perimeter = cv2.arcLength(outer_contour, True)
    inner_perimeter = cv2.arcLength(inner_contour, True)

    outer_approx = cv2.approxPolyDP(outer_contour, 0.035 * outer_perimeter, True)
    inner_approx = cv2.approxPolyDP(inner_contour, 0.035 * inner_perimeter, True)

    if len(outer_approx) != 4 or len(inner_approx) != 4:
        return None

    outer_pts = order_points(outer_approx.reshape(4, 2).astype(np.float32))
    inner_pts = order_points(inner_approx.reshape(4, 2).astype(np.float32))

    otl, otr, obr, obl = outer_pts
    itl, itr, ibr, ibl = inner_pts

    outer_top = distance(otl, otr)
    outer_bottom = distance(obl, obr)
    outer_left = distance(otl, obl)
    outer_right = distance(otr, obr)
    outer_height = (outer_left + outer_right) / 2.0

    inner_top = distance(itl, itr)
    inner_bottom = distance(ibl, ibr)
    inner_left = distance(itl, ibl)
    inner_right = distance(itr, ibr)
    inner_height = (inner_left + inner_right) / 2.0

    center_top_px = (outer_top + inner_top) / 2.0
    center_bottom_px = (outer_bottom + inner_bottom) / 2.0
    center_height_px = (outer_height + inner_height) / 2.0

    outer_side_avg = (outer_top + outer_right + outer_bottom + outer_left) / 4.0
    inner_side_avg = (inner_top + inner_right + inner_bottom + inner_left) / 4.0
    center_side_px = (outer_side_avg + inner_side_avg) / 2.0

    return {
        "center_top_px": center_top_px,
        "center_bottom_px": center_bottom_px,
        "center_height_px": center_height_px,
        "center_side_px": center_side_px,
        "outer_pts": outer_pts,
        "inner_pts": inner_pts
    }

def get_square_side_by_min_rect_centerline(outer_contour, inner_contour):
    """
    正方形兜底测量：
    当四边形顶点拟合不稳定时，用最小外接旋转矩形测中心线边长。
    """
    if outer_contour is None:
        return None

    outer_rect = cv2.minAreaRect(outer_contour)
    outer_w, outer_h = outer_rect[1]

    if outer_w <= 0 or outer_h <= 0:
        return None

    outer_side = (outer_w + outer_h) / 2.0

    if inner_contour is not None:
        inner_rect = cv2.minAreaRect(inner_contour)
        inner_w, inner_h = inner_rect[1]

        if inner_w > 0 and inner_h > 0:
            inner_side = (inner_w + inner_h) / 2.0
            return (outer_side + inner_side) / 2.0

    # 没有内轮廓时，退化为外边长减线宽影响
    return max(0, outer_side - FIGURE_LINE_WIDTH_PX)

def detect_shape_in_warped(warped_a4):
    """
    在矫正后的 A4 图像中识别：
    圆形、等边三角形、正方形、梯形。

    尺寸测量方式：
    空心图形黑线宽度约 1cm。
    这里不测黑线外轮廓，而是测黑线中心线尺寸：
    中心线尺寸 = (外轮廓尺寸 + 内轮廓尺寸) / 2
    """

    inner, offset_x, offset_y = get_inner_region(warped_a4)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 黑色图形变白，白纸变黑
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 竖直方向识别不到，通常是黑线局部断开，导致空心图形没有形成闭环。
    # 这里分别用普通核、纵向核、横向核补断边。
    # 如果后续发现图形被粘粗太多，可以先把 dilate 那一行注释掉。
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)

    # 轻微加粗，进一步保证空心图形闭合
    binary = cv2.dilate(binary, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=1)

    debug_show = warped_a4.copy()

    # 黄色框：用于图形识别的裁剪区域
    cv2.rectangle(
        debug_show,
        (offset_x, offset_y),
        (offset_x + inner.shape[1], offset_y + inner.shape[0]),
        (0, 255, 255),
        2
    )

    outer_contour, inner_contour = find_shape_outer_inner_contour(binary)

    # 关键修改：
    # 不再强制依赖“空心图形必须有内孔”。
    # 如果竖直方向断环/分裂导致找不到 inner_contour，
    # 先尝试把所有黑色轮廓合并成一个整体轮廓。
    strong_binary = binary.copy()

    if outer_contour is None:
        outer_contour = find_combined_shape_contour(binary)
        inner_contour = None

    # 如果合并轮廓仍失败，再用强化二值图找外轮廓。
    if outer_contour is None:
        outer_contour, strong_binary = find_shape_contour_strong(binary)
        inner_contour = None

    # 如果强化版仍失败，再用普通合理外轮廓兜底。
    if outer_contour is None:
        outer_contour = find_largest_reasonable_contour(binary)
        inner_contour = None

    # 最后一层兜底：直接找最大外轮廓。
    if outer_contour is None:
        outer_contour = find_largest_external_contour(binary)
        inner_contour = None

    if outer_contour is None:
        cv2.putText(
            debug_show,
            "Shape: none",
            (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        return None, debug_show, strong_binary

    contour = outer_contour

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area < 300 or perimeter < 50:
        cv2.putText(
            debug_show,
            "Shape: too small",
            (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        return None, debug_show, strong_binary

    # 外轮廓转回 warped_a4 坐标
    contour_global = contour.copy()
    contour_global[:, 0, 0] += offset_x
    contour_global[:, 0, 1] += offset_y

    cv2.drawContours(debug_show, [contour_global], -1, (0, 255, 0), 2)

    # 内轮廓也画出来，方便观察中心线测量是否可靠
    if inner_contour is not None:
        inner_contour_global = inner_contour.copy()
        inner_contour_global[:, 0, 0] += offset_x
        inner_contour_global[:, 0, 1] += offset_y
        cv2.drawContours(debug_show, [inner_contour_global], -1, (255, 0, 0), 2)

    # 多边形拟合用外轮廓
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

    if len(approx) > 8:
        approx2 = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    else:
        approx2 = approx

    vertices = len(approx2)

    # 新增：同时尝试三角形拟合和四边形拟合
    tri_approx = approx_polygon_by_target_vertices(contour, 3)
    quad_approx = approx_polygon_by_target_vertices(contour, 4)

    tri_ok = is_reasonable_triangle(tri_approx, contour)
    quad_ok = is_reasonable_square(quad_approx, contour)

    # 正方形兜底：approxPolyDP 没拟合出 4 点时，
    # 只有最小外接矩形非常接近正方形，才允许进入四边形分支。
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rect_square_ok = False

    if rw > 0 and rh > 0:
        rect_ratio = min(rw, rh) / max(rw, rh)

        if rect_ratio > 0.82:
            rect_square_ok = True

    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    # 中心点用外轮廓质心，空心图形中心位置不受影响
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cx_inner = int(M["m10"] / M["m00"])
        cy_inner = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx_inner = x + w // 2
        cy_inner = y + h // 2

    cx = cx_inner + offset_x
    cy = cy_inner + offset_y

    center_x_cm = cx * CM_PER_PIXEL_X
    center_y_cm = cy * CM_PER_PIXEL_Y

    # 以 A4 中心为原点，右正、上正
    center_rel_x_cm = center_x_cm - A4_WIDTH_CM / 2.0
    center_rel_y_cm = A4_HEIGHT_CM / 2.0 - center_y_cm

    shape_info = {
        "shape": "unknown",

        "center_px": (cx, cy),
        "center_cm": (center_x_cm, center_y_cm),
        "center_rel_cm": (center_rel_x_cm, center_rel_y_cm),

        "area": area,
        "perimeter": perimeter,
        "vertices": vertices,
        "circularity": circularity,

        "measure_mode": "centerline",
        "figure_line_width_cm": FIGURE_LINE_WIDTH_CM
    }

    # 圆形
    if circularity > 0.72 and vertices >= 6:
        (x, y), radius = cv2.minEnclosingCircle(contour)

        diameter_px = get_circle_diameter_centerline(outer_contour, inner_contour)

        # 如果没有内轮廓，退化为外径减 1cm
        if diameter_px is None:
            diameter_px = max(0, 2 * radius - FIGURE_LINE_WIDTH_PX)

        diameter_cm = diameter_px * CM_PER_PIXEL_X

        shape_info["shape"] = "circle"
        shape_info["diameter_px"] = diameter_px
        shape_info["diameter_cm"] = diameter_cm

        # 画外圆
        cv2.circle(
            debug_show,
            (int(x + offset_x), int(y + offset_y)),
            int(radius),
            (255, 0, 255),
            2
        )

    # 三角形
    elif vertices == 3 or tri_ok:
        if tri_ok:
            pts = tri_approx.reshape(3, 2).astype(np.float32)

        elif vertices == 3:
            pts = approx2.reshape(3, 2).astype(np.float32)

        else:
            ret_tri, tri_pts = cv2.minEnclosingTriangle(contour)

            if tri_pts is not None:
                pts = tri_pts.reshape(3, 2).astype(np.float32)
            else:
                pts = approx2.reshape(3, 2).astype(np.float32)

        side1 = distance(pts[0], pts[1])
        side2 = distance(pts[1], pts[2])
        side3 = distance(pts[2], pts[0])

        outer_side_avg = (side1 + side2 + side3) / 3.0

        side_center_px = get_triangle_side_centerline(outer_contour, inner_contour)

        # 兜底1：用 minEnclosingTriangle 测中心线
        if side_center_px is None:
            side_center_px = get_triangle_side_by_min_enclosing_centerline(
                outer_contour,
                inner_contour
            )

        # 兜底2：仍失败时，用外边长减线宽
        if side_center_px is None:
            side_center_px = max(0, outer_side_avg - FIGURE_LINE_WIDTH_PX)

        shape_info["shape"] = "triangle"
        shape_info["side_px"] = side_center_px
        shape_info["side_cm"] = side_center_px * CM_PER_PIXEL_X

        pts_global = pts.copy()
        pts_global[:, 0] += offset_x
        pts_global[:, 1] += offset_y

        cv2.polylines(
            debug_show,
            [pts_global.astype(np.int32)],
            True,
            (255, 0, 255),
            2
        )

    # 四边形：正方形 or 梯形
    elif vertices == 4 or quad_ok or rect_square_ok:
        if quad_ok:
            pts = quad_approx.reshape(4, 2).astype(np.float32)

        elif vertices == 4:
            pts = approx2.reshape(4, 2).astype(np.float32)

        else:
            # approx 没有四个点时，用最小外接矩形四角兜底。
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            pts = np.array(box, dtype=np.float32)

        shape_name, extra = classify_quad(pts, contour)
        shape_info["shape"] = shape_name

        quad_center_info = get_quad_centerline_info(outer_contour, inner_contour)

        if quad_center_info is not None:
            if shape_name == "square":
                shape_info["side_px"] = quad_center_info["center_side_px"]
                shape_info["side_cm"] = quad_center_info["center_side_px"] * CM_PER_PIXEL_X

            else:
                shape_info["top_px"] = quad_center_info["center_top_px"]
                shape_info["bottom_px"] = quad_center_info["center_bottom_px"]
                shape_info["height_px"] = quad_center_info["center_height_px"]

                shape_info["top_cm"] = quad_center_info["center_top_px"] * CM_PER_PIXEL_X
                shape_info["bottom_cm"] = quad_center_info["center_bottom_px"] * CM_PER_PIXEL_X
                shape_info["height_cm"] = quad_center_info["center_height_px"] * CM_PER_PIXEL_Y

        else:
            # 兜底：用外轮廓测量结果，再减线宽影响
            if shape_name == "square":
                side_px = get_square_side_by_min_rect_centerline(
                    outer_contour,
                    inner_contour
                )

                if side_px is None:
                    side_px = max(0, extra["side_px"] - FIGURE_LINE_WIDTH_PX)

                shape_info["side_px"] = side_px
                shape_info["side_cm"] = side_px * CM_PER_PIXEL_X

            else:
                top_px = max(0, extra["top_px"] - FIGURE_LINE_WIDTH_PX)
                bottom_px = max(0, extra["bottom_px"] - FIGURE_LINE_WIDTH_PX)
                height_px = max(0, extra["height_px"] - FIGURE_LINE_WIDTH_PX)

                shape_info["top_px"] = top_px
                shape_info["bottom_px"] = bottom_px
                shape_info["height_px"] = height_px

                shape_info["top_cm"] = top_px * CM_PER_PIXEL_X
                shape_info["bottom_cm"] = bottom_px * CM_PER_PIXEL_X
                shape_info["height_cm"] = height_px * CM_PER_PIXEL_Y

        pts_global = pts.copy()
        pts_global[:, 0] += offset_x
        pts_global[:, 1] += offset_y
        pts_global = order_points(pts_global.astype(np.float32))

        cv2.polylines(
            debug_show,
            [pts_global.astype(np.int32)],
            True,
            (255, 0, 255),
            2
        )

    # 兜底圆形
    else:
        if circularity > 0.65:
            (x, y), radius = cv2.minEnclosingCircle(contour)

            diameter_px = get_circle_diameter_centerline(outer_contour, inner_contour)

            if diameter_px is None:
                diameter_px = max(0, 2 * radius - FIGURE_LINE_WIDTH_PX)

            diameter_cm = diameter_px * CM_PER_PIXEL_X

            shape_info["shape"] = "circle"
            shape_info["diameter_px"] = diameter_px
            shape_info["diameter_cm"] = diameter_cm

            cv2.circle(
                debug_show,
                (int(x + offset_x), int(y + offset_y)),
                int(radius),
                (255, 0, 255),
                2
            )

        else:
            shape_info["shape"] = "unknown"

    # 显示结果
    cv2.circle(debug_show, (cx, cy), 5, (0, 0, 255), -1)

    text1 = "Shape: %s" % shape_info["shape"]
    text2 = "Center: %.1f, %.1f cm" % (
        shape_info["center_rel_cm"][0],
        shape_info["center_rel_cm"][1]
    )
    text3 = "V: %d  C: %.2f" % (
        shape_info["vertices"],
        shape_info["circularity"]
    )

    if shape_info["shape"] == "circle":
        text4 = "Centerline Diameter: %.1f cm" % shape_info["diameter_cm"]

    elif shape_info["shape"] == "triangle":
        text4 = "Centerline Side: %.1f cm" % shape_info["side_cm"]

    elif shape_info["shape"] == "square":
        text4 = "Centerline Side: %.1f cm" % shape_info["side_cm"]

    elif shape_info["shape"] == "trapezoid":
        text4 = "Centerline Top: %.1f  Bottom: %.1f  H: %.1f cm" % (
            shape_info["top_cm"],
            shape_info["bottom_cm"],
            shape_info["height_cm"]
        )

    else:
        text4 = "Size: unknown"

    cv2.putText(debug_show, text1, (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.putText(debug_show, text2, (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.putText(debug_show, text3, (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.putText(debug_show, text4, (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.putText(debug_show, "Measure: centerline",
                (10, 350), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2)

    return shape_info, debug_show, strong_binary


def round_shape_info(shape_info):
    """
    让终端最终输出只保留一位小数。
    不改变内部计算精度，只改变 print 出来的显示效果。
    """
    if shape_info is None:
        return None

    rounded = {}

    for key, value in shape_info.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), 1)

        elif isinstance(value, tuple):
            rounded[key] = tuple(
                round(float(v), 1) if isinstance(v, (float, np.floating)) else v
                for v in value
            )

        else:
            rounded[key] = value

    return rounded


# =========================
# 显示函数
# =========================

def draw_result(frame, ordered_points, center_offset, debug_info, fps):
    show = frame.copy()

    if ordered_points is None:
        cv2.putText(show, "NO A4 FRAME",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

        cv2.putText(show, "FPS: %.1f" % fps,
                    (5, 460), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

        return show

    outer = ordered_points.astype(np.int32)

    # 绿色：A4 外框
    cv2.polylines(show, [outer], True, (0, 255, 0), 2)

    # 蓝色：黑框内孔
    if debug_info is not None and "inner_points" in debug_info:
        inner = debug_info["inner_points"].astype(np.int32)
        cv2.polylines(show, [inner], True, (255, 0, 0), 1)

    labels = ["TL", "TR", "BR", "BL"]

    for i, p in enumerate(outer):
        x, y = p
        cv2.circle(show, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(show, labels[i],
                    (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1)

    cx = int(np.mean(outer[:, 0]))
    cy = int(np.mean(outer[:, 1]))

    cv2.circle(show, (cx, cy), 5, (255, 0, 255), -1)

    if center_offset is not None:
        text = "offset: %.2f, %.2f" % (
            center_offset[0],
            center_offset[1]
        )

        cv2.putText(show, text,
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 255), 2)

    if debug_info is not None:
        text2 = "ratio: %.2f center: %.2f" % (
            debug_info["ratio"],
            debug_info["center_black_ratio"]
        )

        cv2.putText(show, text2,
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

    cv2.putText(show, "FPS: %.1f" % fps,
                (5, 460), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)

    return show


# =========================
# 摄像头和主循环
# =========================

def open_camera():
    """
    香橙派稳定版摄像头打开函数。

    这版按你之前移植成功的简单方式来：
    1. 不用 Windows 的 cv2.CAP_DSHOW；
    2. 不用字符串路径 /dev/video0 去打开；
    3. 不自动遍历 /dev/video0、/dev/video1，避免把可用设备试挂；
    4. 只用 cv2.VideoCapture(0) 打开真正的视频流节点。

    前提：你已经用 v4l2-ctl 或 ffmpeg 测试过 video0 可以用。
    """
    print("正在打开摄像头...")
    print("使用方式: cv2.VideoCapture(0)")
    print("如果此程序打不开，先确认没有其它程序占用摄像头")

    # 关键：香橙派上优先用最简单的编号方式。
    # 你之前能成功显示画面的脚本，一般就是这种写法。
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("摄像头打开失败")
        print("请检查：")
        print("  1. ls /dev/video*")
        print("  2. v4l2-ctl --list-devices")
        print("  3. 是否还有 PyCharm、旧的开机自启动程序、cheese 等占用摄像头")
        return None

    # 设置参数。注意：部分 UVC 摄像头不一定接受所有 set，失败也不会报错。
    # MJPG 一般比 YUYV 更适合 USB 摄像头高帧率传输。
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # 部分 OpenCV 版本支持，减少延迟；不支持也不影响。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 先连续读几帧。很多 USB 摄像头刚打开时前几帧会失败，不能立刻判定失败。
    frame = None
    ok = False
    for i in range(30):
        ret, temp = cap.read()
        if ret and temp is not None:
            frame = temp
            ok = True
            break
        time.sleep(0.1)

    if not ok:
        print("摄像头已打开，但是连续读取画面失败")
        print("建议先执行：")
        print("  pkill -f camera_test.py")
        print("  pkill -f run.sh")
        print("然后重新插拔摄像头，或重启香橙派后再运行")
        cap.release()
        return None

    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("摄像头打开并读取画面成功")
    print("CAMERA_ID:", CAMERA_ID)
    print("请求分辨率: %d x %d" % (FRAME_WIDTH, FRAME_HEIGHT))
    print("请求FPS:", CAMERA_FPS)
    print("实际分辨率: %.0f x %.0f" % (real_w, real_h))
    print("摄像头报告FPS:", real_fps)
    print("实际FOURCC:", fourcc_str)
    print("程序处理FPS限制:", TARGET_PROCESS_FPS)
    print("标准 A4: %.1fcm x %.1fcm" % (A4_WIDTH_CM, A4_HEIGHT_CM))
    print("标准黑框: %.1fcm = %dpx" % (BORDER_CM, BORDER_PX_X))
    print("图形线宽: %.1fcm = %dpx，尺寸按中心线测量" % (FIGURE_LINE_WIDTH_CM, FIGURE_LINE_WIDTH_PX))

    return cap

# ============================================================
# 串口通信 + 状态机巡线绘图
# ============================================================

# 串口参数：仍然使用原来的 TX/RX 串口
ENABLE_SERIAL = True
SERIAL_PORT = "/dev/ttyS2"
SERIAL_BAUD = 115200

# 串口固定发送周期：50ms
SERIAL_INTERVAL = 0.050

# 终端打印周期
PRINT_INTERVAL = 0.20

# 串口调试打印开关
# True：打印 STM32 发来的帧、香橙派实际发出的帧，方便调试。
# 如果后期嫌终端刷屏，可以把 DEBUG_SERIAL_PRINT 改成 False。
DEBUG_SERIAL_PRINT = False

# 是否打印串口原始接收字节。
# 一般只看解析出的 6 字节帧即可；如果怀疑帧头错位/丢字节，再改成 True。
DEBUG_SERIAL_RAW_PRINT = True

# 发送数据打印节流。True 表示相同的 TX 帧也每 50ms 打印一次；False 表示相同帧只打印第一次。
DEBUG_SERIAL_PRINT_REPEAT_TX = True

# 到达目标点阈值，单位 cm
ARRIVE_DISTANCE_CM = 0.30

# 到起点后不要立刻开激光：必须在允许小范围波动内连续稳定一段时间。
# START_STABLE_DISTANCE_CM 可以略大于 ARRIVE_DISTANCE_CM，避免云台轻微抖动导致一直等不到开激光。
# START_STABLE_TIME_SEC 建议 0.30~0.60，越大越稳，但起笔等待越久。
START_STABLE_DISTANCE_CM = 0.40
START_STABLE_TIME_SEC = 0.40

# 搜索黑框阶段用更宽的中心范围，避免云台停不下来
SEARCH_CENTER_TOLERANCE_X_RATIO = 0.18
SEARCH_CENTER_TOLERANCE_Y_RATIO = 0.18
SEARCH_STABLE_FRAMES = 2

# 发挥 25 先找目标物时，目标物需要大致进入画面中心后才通知 STM32。
# 这里复用 ratio 偏差，0.18 表示中心偏差不超过画面半宽/半高的 18%。
TARGET_CENTER_TOLERANCE_X_RATIO = 0.18
TARGET_CENTER_TOLERANCE_Y_RATIO = 0.18
TARGET_CENTER_STABLE_FRAMES = 2

# 目标物已找到帧不需要 STM32 回 ACK，为了防止单帧丢失，连续发送一小段时间。
TARGET_FOUND_NOTIFY_TIME = 0.20

# 发挥 24/25 搜索“真正画板”时使用：画板内部应基本没有黑色图形。
# 如果误把目标物当画板：调小到 0.06；如果光照/污点导致画板漏检：调大到 0.10~0.12。
EMPTY_BOARD_CENTER_BLACK_RATIO_MAX = 0.08


# 目标轨迹离散间隔，单位 cm
LINE_STEP_CM = 0.5
CIRCLE_STEP_DEG = 5.0

# =========================
# 通信协议帧
# =========================
FRAME_LEN = 6

LASER_ON_FRAME = bytes([0x03, 0x01, 0x00, 0x00, 0x00, 0x30])
LASER_OFF_FRAME = bytes([0x03, 0x02, 0x00, 0x00, 0x00, 0x30])
STOP_SCAN_FRAME = bytes([0x03, 0x03, 0x00, 0x00, 0x00, 0x30])
BOARD_READY_FRAME = bytes([0x03, 0x04, 0x00, 0x00, 0x00, 0x30])

# 发挥第五问新增：香橙派识别到目标物，并且目标物大致进入画面中心后，通知 STM32。
# STM32 收到后可以短暂停止云台，随后继续扫描寻找空白画板。
TARGET_FOUND_FRAME = bytes([0x03, 0x05, 0x00, 0x00, 0x00, 0x30])

# 系统状态
SYS_IDLE = 0
SYS_RECOGNIZE_TARGET = 1
SYS_NOTIFY_TARGET_FOUND = 2
SYS_WAIT_BOARD_READY = 3
SYS_SEARCH_BOARD = 4
SYS_MOVE_TO_START = 5
SYS_WAIT_LASER_ON_ACK = 6
SYS_DRAWING = 7
SYS_WAIT_LASER_OFF_ACK = 8

STATE_NAME = {
    SYS_IDLE: "IDLE",
    SYS_RECOGNIZE_TARGET: "RECOGNIZE_TARGET",
    SYS_NOTIFY_TARGET_FOUND: "NOTIFY_TARGET_FOUND",
    SYS_WAIT_BOARD_READY: "WAIT_BOARD_READY",
    SYS_SEARCH_BOARD: "SEARCH_BOARD",
    SYS_MOVE_TO_START: "MOVE_TO_START",
    SYS_WAIT_LASER_ON_ACK: "WAIT_LASER_ON_ACK",
    SYS_DRAWING: "DRAWING",
    SYS_WAIT_LASER_OFF_ACK: "WAIT_LASER_OFF_ACK",
}



def bytes_to_hex(data):
    """bytes -> 'AA 12 00 00 ED 55' 这种显示格式。"""
    if data is None:
        return "None"
    return " ".join("%02X" % b for b in data)


def describe_frame(frame):
    """把常用 6 字节帧翻译成便于调试查看的文字。"""
    if frame is None:
        return "None"

    if len(frame) != FRAME_LEN:
        return "UNKNOWN_LEN"

    if frame[0] == 0xAA and frame[5] == 0x55:
        state_code = frame[1]
        check_ok = (frame[4] == ((state_code ^ 0xFF) & 0xFF))
        return "TASK_CMD state=0x%02X check=%s" % (state_code, "OK" if check_ok else "BAD")

    if frame == LASER_ON_FRAME:
        return "CTRL LASER_ON / LASER_ON_ACK"

    if frame == LASER_OFF_FRAME:
        return "CTRL LASER_OFF / LASER_OFF_ACK"

    if frame == STOP_SCAN_FRAME:
        return "CTRL STOP_SCAN"

    if frame == BOARD_READY_FRAME:
        return "CTRL BOARD_READY"

    if frame == TARGET_FOUND_FRAME:
        return "CTRL TARGET_FOUND"

    if frame == bytes([0x07, 0x01, 0x00, 0x00, 0x02, 0xF7]):
        return "OLD BOARD_READY_FRAME"

    if frame == bytes([0xF7, 0x01, 0x00, 0x00, 0x02, 0x07]):
        return "OLD STOP_SCAN_FRAME"

    if frame[0] in (0x01, 0x02) and frame[5] in (0x05, 0x04):
        dx_raw = (frame[1] << 8) | frame[2]
        dy_raw = (frame[3] << 8) | frame[4]
        if dx_raw >= 0x8000:
            dx_raw -= 0x10000
        if dy_raw >= 0x8000:
            dy_raw -= 0x10000
        dx_cm = dx_raw / 100.0
        dy_cm = dy_raw / 100.0
        mode = "DRAW" if frame[0] == 0x01 else "MOVE_TO_START"
        return "%s dx=%.2fcm dy=%.2fcm" % (mode, dx_cm, dy_cm)

    return "UNKNOWN_FRAME"


def debug_print_rx(frame, note="RX"):
    if DEBUG_SERIAL_PRINT:
        print("[%s] %s  %s" % (note, bytes_to_hex(frame), describe_frame(frame)))


def debug_print_tx(frame, sys_state=None, task_state=None):
    if not DEBUG_SERIAL_PRINT:
        return

    prefix = "[TX]"
    if sys_state is not None:
        prefix += " SYS=%s" % STATE_NAME.get(sys_state, "?")
    if task_state is not None:
        prefix += " TASK=0x%02X" % task_state

    print("%s  %s  %s" % (prefix, bytes_to_hex(frame), describe_frame(frame)))


def open_serial_port():
    if not ENABLE_SERIAL:
        return None

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0)
        print("串口已打开: %s, %d" % (SERIAL_PORT, SERIAL_BAUD))
        return ser
    except Exception as e:
        print("串口打开失败:", e)
        return None


def make_task_command(state_code):
    """
    STM32 -> 香橙派：题目启动帧
    格式：AA state 00 00 check 55
    check = state ^ 0xFF
    """
    return bytes([0xAA, state_code & 0xFF, 0x00, 0x00, (state_code ^ 0xFF) & 0xFF, 0x55])


def parse_task_command(frame):
    if len(frame) != FRAME_LEN:
        return None

    if frame[0] != 0xAA or frame[5] != 0x55:
        return None

    state_code = frame[1]
    check = frame[4]

    if check != ((state_code ^ 0xFF) & 0xFF):
        return None

    if state_code not in (0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24, 0x25):
        return None

    return state_code


def build_xy_packet(dx_cm, dy_cm, head, tail):
    """
    构造 dx/dy 六字节帧：
    head dx_H dx_L dy_H dy_L tail

    dx/dy 单位 cm，保留两位小数后乘 100，按 int16 补码发送。
    """
    dx_i = int(round(round(dx_cm, 2) * 100))
    dy_i = int(round(round(dy_cm, 2) * 100))

    dx_i = max(-32768, min(32767, dx_i))
    dy_i = max(-32768, min(32767, dy_i))

    dx_u = dx_i & 0xFFFF
    dy_u = dy_i & 0xFFFF

    return bytes([
        head & 0xFF,
        (dx_u >> 8) & 0xFF,
        dx_u & 0xFF,
        (dy_u >> 8) & 0xFF,
        dy_u & 0xFF,
        tail & 0xFF,
    ])


def build_move_packet(dx_cm, dy_cm):
    # 空走到起点，激光关闭
    return build_xy_packet(dx_cm, dy_cm, 0x02, 0x04)


def build_draw_packet(dx_cm, dy_cm):
    # 正式画图，激光打开
    return build_xy_packet(dx_cm, dy_cm, 0x01, 0x05)


def serial_send_worker(ser, latest_packet_holder, packet_lock, stop_event):
    """
    固定 50ms 发送一次 latest_packet_holder["packet"]。
    当 packet 为 None 时不发送，表示 IDLE 等状态下保持安静。
    """
    next_send_time = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()

        if now >= next_send_time:
            with packet_lock:
                packet = latest_packet_holder.get("packet", None)

            if packet is not None:
                try:
                    ser.write(packet)

                    if DEBUG_SERIAL_PRINT:
                        # 发送线程里只知道当前要发的 packet，不直接读主状态，
                        # 所以这里打印实际发出的 6 字节数据和帧含义。
                        if DEBUG_SERIAL_PRINT_REPEAT_TX:
                            debug_print_tx(packet)
                        else:
                            last_packet = latest_packet_holder.get("last_printed_tx", None)
                            if last_packet != packet:
                                debug_print_tx(packet)
                                latest_packet_holder["last_printed_tx"] = packet

                except Exception as e:
                    print("串口发送失败:", e)
                    time.sleep(0.05)

            next_send_time += SERIAL_INTERVAL
            if next_send_time < now - SERIAL_INTERVAL:
                next_send_time = now + SERIAL_INTERVAL
        else:
            time.sleep(min(next_send_time - now, 0.001))


def is_known_frame(frame):
    if len(frame) != FRAME_LEN:
        return False

    # 题目启动帧
    if frame[0] == 0xAA and frame[5] == 0x55:
        return True

    # 控制帧
    if frame[0] == 0x03 and frame[5] == 0x30:
        return True

    # 兼容你之前临时定义过的 ready/stop 方向帧。如果后续全部改成 03xx30，可删掉。
    if frame == bytes([0x07, 0x01, 0x00, 0x00, 0x02, 0xF7]):
        return True
    if frame == bytes([0xF7, 0x01, 0x00, 0x00, 0x02, 0x07]):
        return True

    return False


def serial_receive_worker(ser, rx_queue, rx_lock, stop_event):
    """
    串口接收线程：一直读取 STM32 发来的数据，并按 6 字节帧放入队列。
    主状态机只会在对应状态处理对应帧，因此 STM32 重复发送不会造成重复执行。
    """
    buf = bytearray()

    while not stop_event.is_set():
        try:
            n = ser.in_waiting
            if n:
                new_data = ser.read(n)
                if DEBUG_SERIAL_RAW_PRINT and new_data:
                    print("[RX_RAW] %s" % bytes_to_hex(new_data))
                buf.extend(new_data)
            else:
                time.sleep(0.002)
                continue

            # 从缓冲区中找 6 字节帧
            while len(buf) >= FRAME_LEN:
                candidate = bytes(buf[:FRAME_LEN])

                if is_known_frame(candidate):
                    with rx_lock:
                        rx_queue.append(candidate)
                    debug_print_rx(candidate, note="RX_FRAME")
                    del buf[:FRAME_LEN]
                else:
                    # 帧头没对齐，丢掉一个字节继续找
                    del buf[0]

                # 防止异常数据过多导致缓存无限增长
                if len(buf) > 200:
                    buf.clear()

        except Exception as e:
            print("串口接收失败:", e)
            time.sleep(0.05)


def pop_rx_frames(rx_queue, rx_lock):
    with rx_lock:
        frames = list(rx_queue)
        rx_queue.clear()
    return frames


def clear_rx(ser, rx_queue, rx_lock):
    with rx_lock:
        rx_queue.clear()
    if ser is not None:
        try:
            ser.reset_input_buffer()
        except Exception:
            pass


def set_latest_packet(latest_packet_holder, packet_lock, packet):
    with packet_lock:
        latest_packet_holder["packet"] = packet


# =========================
# 坐标和轨迹生成
# =========================
def cm_to_warp_px_board(x_cm, y_cm):
    x_px = int(round(WARP_WIDTH / 2 + x_cm / CM_PER_PIXEL_X))
    y_px = int(round(WARP_HEIGHT / 2 - y_cm / CM_PER_PIXEL_Y))
    return x_px, y_px


def warp_px_to_cm_board(x_px, y_px):
    x_cm = (x_px - WARP_WIDTH / 2) * CM_PER_PIXEL_X
    y_cm = (WARP_HEIGHT / 2 - y_px) * CM_PER_PIXEL_Y
    return x_cm, y_cm


def image_center_to_board_cm(M):
    """
    将 result 画面黄色十字中心映射到透视矫正后的 A4 坐标系。
    因为你的激光笔固定在画面中心，所以这里就是当前激光点坐标。
    """
    src = np.array([[[FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, M)
    x_px = float(dst[0, 0, 0])
    y_px = float(dst[0, 0, 1])
    return warp_px_to_cm_board(x_px, y_px), (int(round(x_px)), int(round(y_px)))


def dist_cm2(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def interpolate_line_cm(p1, p2, step_cm=LINE_STEP_CM):
    length = dist_cm2(p1, p2)
    if length < 1e-6:
        return [p2]

    n = max(1, int(math.ceil(length / step_cm)))
    points = []
    for i in range(1, n + 1):
        t = i / n
        points.append((
            p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t
        ))
    return points


def polyline_points_cm(vertices, closed=True, step_cm=LINE_STEP_CM):
    if not vertices:
        return []

    points = [vertices[0]]
    end = len(vertices) if closed else len(vertices) - 1

    for i in range(end):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        points.extend(interpolate_line_cm(p1, p2, step_cm))

    return points


def build_fixed_task_points(state_code):
    """
    基础部分固定图形：
    状态12：以 (-3,3) 为起点，边长 6cm 正方形
    状态13：按你的方法，以 (0,0) 为起点画边长 6cm 等边三角形
    状态14：按你给的侧放梯形四点
    """
    if state_code == 0x12:
        vertices = [(-3.0, 3.0), (3.0, 3.0), (3.0, -3.0), (-3.0, -3.0)]
        return polyline_points_cm(vertices, closed=True)

    if state_code == 0x13:
        h = 6.0 * math.sqrt(3.0) / 2.0
        vertices = [(0.0, 0.0), (6.0, 0.0), (3.0, h)]
        return polyline_points_cm(vertices, closed=True)

    #if state_code == 0x14:
    #    vertices = [
    #        (-7.5, 10.0),
    #        (7.5, 7.5),
    #        (7.5, -7.5),
    #        (-7.5, -10.0),
    #    ]
    #
    #if state_code == 0x14:
     #   vertices = [
     #       (-6.5, 9.0),
     #       (6.5, 6.5),
     #       (6.5, -6.5),
     #       (-6.5, -9.0),
     #   ]
     #   return polyline_points_cm(vertices, closed=True)

    if state_code == 0x14:
       vertices = [
           (-5.0, 8.0),
           (5.0, 5.0),
           (5.0, -5.0),
           (-5.0, -8.0),
       ]
       return polyline_points_cm(vertices, closed=True)
    return []


def build_points_from_shape_info_for_draw(shape_info):
    """
    发挥部分：根据识别出的目标物形状和尺寸生成画板上的轨迹。
    轨迹默认以画板中心为图形中心。
    """
    if shape_info is None:
        return []

    shape = shape_info.get("shape", "unknown")

    if shape == "square":
        side = float(shape_info.get("side_cm", 10.0))
        s = side / 2.0
        vertices = [(-s, s), (s, s), (s, -s), (-s, -s)]
        return polyline_points_cm(vertices, closed=True)

    if shape == "triangle":
        side = float(shape_info.get("side_cm", 10.0))
        h = side * math.sqrt(3.0) / 2.0
        vertices = [
            (0.0, 2.0 * h / 3.0),
            (-side / 2.0, -h / 3.0),
            (side / 2.0, -h / 3.0),
        ]
        return polyline_points_cm(vertices, closed=True)

    if shape == "trapezoid":
        top = float(shape_info.get("top_cm", 12.0))
        bottom = float(shape_info.get("bottom_cm", 16.0))
        height = float(shape_info.get("height_cm", 10.0))

        # 默认横放梯形。如果你发现 16cm 以上太贴边，可以在这里改成侧放。
        vertices = [
            (-top / 2.0, height / 2.0),
            (top / 2.0, height / 2.0),
            (bottom / 2.0, -height / 2.0),
            (-bottom / 2.0, -height / 2.0),
        ]
        return polyline_points_cm(vertices, closed=True)

    if shape == "circle":
        diameter = float(shape_info.get("diameter_cm", 10.0))
        r = diameter / 2.0
        points = []
        deg = 0.0
        while deg < 360.0:
            theta = math.radians(deg)
            points.append((r * math.cos(theta), r * math.sin(theta)))
            deg += CIRCLE_STEP_DEG
        if points:
            points.append(points[0])
        return points

    return []


def is_board_roughly_centered(center_offset):
    if center_offset is None:
        return False
    ox, oy = center_offset
    return (abs(ox) <= SEARCH_CENTER_TOLERANCE_X_RATIO and
            abs(oy) <= SEARCH_CENTER_TOLERANCE_Y_RATIO)


# =========================
# 显示辅助
# =========================
def draw_tracker_warp(warped_a4, target_points, current_index, laser_px, laser_cm,
                      target_cm, dx, dy, sys_state, task_state, shape_info):
    show = warped_a4.copy()

    cv2.line(show, (0, WARP_HEIGHT // 2), (WARP_WIDTH, WARP_HEIGHT // 2), (200, 200, 200), 1)
    cv2.line(show, (WARP_WIDTH // 2, 0), (WARP_WIDTH // 2, WARP_HEIGHT), (200, 200, 200), 1)

    if target_points:
        pts_px = np.array([cm_to_warp_px_board(x, y) for x, y in target_points], dtype=np.int32)
        if len(pts_px) >= 2:
            cv2.polylines(show, [pts_px], False, (255, 0, 255), 1)

        if 0 <= current_index < len(target_points):
            tx, ty = cm_to_warp_px_board(*target_points[current_index])
            cv2.circle(show, (tx, ty), 5, (0, 0, 255), -1)

    if laser_px is not None:
        cv2.circle(show, laser_px, 7, (0, 255, 255), 2)
        cv2.circle(show, laser_px, 2, (0, 255, 255), -1)

    cv2.putText(show, "SYS: %s  TASK: 0x%02X" % (STATE_NAME.get(sys_state, "?"), task_state),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

    if laser_cm is not None:
        cv2.putText(show, "center/laser: %.2f %.2f cm" % laser_cm,
                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

    if target_cm is not None:
        cv2.putText(show, "target: %.2f %.2f cm" % target_cm,
                    (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

    if dx is not None and dy is not None:
        cv2.putText(show, "dx dy: %.2f %.2f cm" % (dx, dy),
                    (8, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 0), 2)

    if shape_info is not None:
        cv2.putText(show, "shape: %s" % shape_info.get("shape", "none"),
                    (8, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)

    return show


def draw_result_with_cross(frame, ordered_points, center_offset, debug_info, fps, sys_state, task_state):
    show = draw_result(frame, ordered_points, center_offset, debug_info, fps)

    h, w = show.shape[:2]
    cx = w // 2
    cy = h // 2

    cv2.line(show, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
    cv2.line(show, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

    cv2.putText(show, "SYS:%s TASK:0x%02X" % (STATE_NAME.get(sys_state, "?"), task_state),
                (5, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

    return show


# =========================
# 主程序
# =========================
def main():
    print("当前程序：状态12~25 串口状态机版本")
    print("串口发送周期：%.0f ms" % (SERIAL_INTERVAL * 1000))
    print("题目命令：AA state 00 00 check 55")
    print("起点空走：02 dxH dxL dyH dyL 04")
    print("正式画图：01 dxH dxL dyH dyL 05")
    print("开激光/关激光/停止扫描/画板ready/目标物已找到：03 01/02/03/04/05 00 00 00 30")
    print("起点开激光条件：误差 <= %.2fcm 且连续稳定 %.2fs" % (START_STABLE_DISTANCE_CM, START_STABLE_TIME_SEC))

    cap = open_camera()
    if cap is None:
        return

    ser = open_serial_port()

    latest_packet_holder = {"packet": None}
    packet_lock = threading.Lock()
    stop_event = threading.Event()

    rx_queue = []
    rx_lock = threading.Lock()

    send_thread = None
    recv_thread = None

    if ser is not None:
        send_thread = threading.Thread(
            target=serial_send_worker,
            args=(ser, latest_packet_holder, packet_lock, stop_event),
            daemon=True
        )
        recv_thread = threading.Thread(
            target=serial_receive_worker,
            args=(ser, rx_queue, rx_lock, stop_event),
            daemon=True
        )
        send_thread.start()
        recv_thread.start()
        print("串口发送/接收线程已启动")

    gui_enabled = USE_GUI and (os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None)

    if USE_GUI and not gui_enabled:
        print("未检测到 DISPLAY/WAYLAND_DISPLAY，已自动关闭图像窗口显示")

    if gui_enabled:
        if SHOW_RESULT:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        if SHOW_SHAPE:
            cv2.namedWindow("shape_result", cv2.WINDOW_NORMAL)
        if SHOW_SHAPE_BINARY:
            cv2.namedWindow("shape_binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("board_warp", cv2.WINDOW_NORMAL)

    sys_state = SYS_IDLE
    task_state = 0x00
    target_points = []
    current_index = 0
    shape_info_saved = None

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    last_print_time = 0.0
    last_switch_time = 0.0
    search_center_stable = 0
    target_center_stable = 0
    target_found_notify_start = 0.0
    start_stable_since = 0.0
    start_stable_printed = False

    target_period = 1.0 / TARGET_PROCESS_FPS

    # 当前视觉结果
    ordered_points = None
    center_offset = None
    debug_info = None
    warped_a4 = None
    M_warp = None
    shape_show = None
    shape_binary = None
    laser_cm = None
    laser_px = None
    target_cm = None
    dx = None
    dy = None

    try:
        while True:
            loop_start = time.time()

            # 读取串口帧，并按当前系统状态处理
            frames = pop_rx_frames(rx_queue, rx_lock)

            for frame6 in frames:
                debug_print_rx(frame6, note="RX_HANDLE SYS=%s" % STATE_NAME.get(sys_state, "?"))

                # 兼容旧 ready 帧，统一转为 BOARD_READY_FRAME
                if frame6 == bytes([0x07, 0x01, 0x00, 0x00, 0x02, 0xF7]):
                    frame6 = BOARD_READY_FRAME

                if sys_state == SYS_IDLE:
                    cmd_state = parse_task_command(frame6)
                    if cmd_state is not None:
                        task_state = cmd_state
                        target_points = []
                        current_index = 0
                        shape_info_saved = None
                        search_center_stable = 0
                        target_center_stable = 0
                        target_found_notify_start = 0.0
                        start_stable_since = 0.0
                        start_stable_printed = False
                        last_switch_time = time.time()

                        print("收到题目命令: 0x%02X" % task_state)

                        if task_state in (0x12, 0x13, 0x14):
                            target_points = build_fixed_task_points(task_state)
                            if len(target_points) > 0:
                                sys_state = SYS_MOVE_TO_START
                                start_stable_since = 0.0
                                start_stable_printed = False
                                print("基础题轨迹生成完成，目标点数量:", len(target_points))
                            else:
                                sys_state = SYS_IDLE
                                task_state = 0x00

                        elif task_state in (0x21, 0x22, 0x23, 0x24, 0x25):
                            sys_state = SYS_RECOGNIZE_TARGET
                            print("进入发挥题识别目标物阶段")

                        clear_rx(ser, rx_queue, rx_lock)

                elif sys_state == SYS_WAIT_BOARD_READY:
                    if frame6 == BOARD_READY_FRAME:
                        sys_state = SYS_MOVE_TO_START
                        start_stable_since = 0.0
                        start_stable_printed = False
                        print("收到画板 ready，开始移动到起点")
                        clear_rx(ser, rx_queue, rx_lock)

                elif sys_state == SYS_WAIT_LASER_ON_ACK:
                    if frame6 == LASER_ON_FRAME:
                        sys_state = SYS_DRAWING
                        current_index = 0
                        last_switch_time = time.time()
                        print("收到开激光 ACK，开始正式画图")
                        clear_rx(ser, rx_queue, rx_lock)

                elif sys_state == SYS_WAIT_LASER_OFF_ACK:
                    if frame6 == LASER_OFF_FRAME:
                        print("收到关激光 ACK，任务 0x%02X 完成，回到空闲" % task_state)
                        sys_state = SYS_IDLE
                        task_state = 0x00
                        target_points = []
                        current_index = 0
                        shape_info_saved = None
                        start_stable_since = 0.0
                        start_stable_printed = False
                        set_latest_packet(latest_packet_holder, packet_lock, None)
                        clear_rx(ser, rx_queue, rx_lock)

            # 读取摄像头
            ret, frame = cap.read()

            if not ret or frame is None:
                print("读取摄像头失败")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # 发挥 24/25 搜索画板阶段，要求黑框内部基本无图形，避免把目标物误判成画板。
            require_empty_board = (sys_state == SYS_SEARCH_BOARD)
            ordered_points, center_offset, black_mask, debug_info = detect_rectangle(
                frame,
                require_empty_inside=require_empty_board
            )

            warped_a4 = None
            M_warp = None
            shape_show = None
            shape_binary = None
            laser_cm = None
            laser_px = None
            target_cm = None
            dx = None
            dy = None

            if ordered_points is not None:
                if debug_info is not None and "inner_points" in debug_info:
                    warped_a4, M_warp = warp_a4_by_inner_points(frame, debug_info["inner_points"])
                else:
                    warped_a4, M_warp = warp_a4_by_outer_points(frame, ordered_points)

                if M_warp is not None:
                    laser_cm, laser_px = image_center_to_board_cm(M_warp)

            # 发挥题：识别目标物
            if sys_state == SYS_RECOGNIZE_TARGET:
                if warped_a4 is not None:
                    shape_info, shape_show, shape_binary = detect_shape_in_warped(warped_a4)

                    valid_shape = (
                        shape_info is not None and
                        shape_info.get("shape") in ("square", "triangle", "trapezoid", "circle")
                    )

                    if valid_shape:
                        # 发挥第五问：目标物和画板都可能在任意位置。
                        # 必须等目标物大致进入画面中心，才认为“找到目标物”，并通知 STM32。
                        # 这样 STM32 可以短暂停止云台，随后继续扫描寻找空白画板。
                        if task_state == 0x25:
                            target_centered = (
                                center_offset is not None and
                                abs(center_offset[0]) <= TARGET_CENTER_TOLERANCE_X_RATIO and
                                abs(center_offset[1]) <= TARGET_CENTER_TOLERANCE_Y_RATIO
                            )

                            if target_centered:
                                target_center_stable += 1
                            else:
                                target_center_stable = 0

                            if target_center_stable < TARGET_CENTER_STABLE_FRAMES:
                                set_latest_packet(latest_packet_holder, packet_lock, None)
                            else:
                                new_target_points = build_points_from_shape_info_for_draw(shape_info)

                                if len(new_target_points) > 0:
                                    target_points = new_target_points
                                    shape_info_saved = shape_info
                                    target_found_notify_start = time.time()
                                    sys_state = SYS_NOTIFY_TARGET_FOUND

                                    set_latest_packet(latest_packet_holder, packet_lock, TARGET_FOUND_FRAME)

                                    print("发挥25：目标物已居中并识别成功:", round_shape_info(shape_info))
                                    print("发挥25：发送目标物已找到帧: %s" % bytes_to_hex(TARGET_FOUND_FRAME))
                                    print("生成轨迹点数量:", len(target_points))

                                    clear_rx(ser, rx_queue, rx_lock)
                        else:
                            target_points = build_points_from_shape_info_for_draw(shape_info)

                            if len(target_points) > 0:
                                shape_info_saved = shape_info
                                print("目标物识别成功:", round_shape_info(shape_info))
                                print("生成轨迹点数量:", len(target_points))

                                # 状态21~23：等待 STM32 固定角度转到画板后发 ready
                                if task_state in (0x21, 0x22, 0x23):
                                    sys_state = SYS_WAIT_BOARD_READY
                                    print("等待 STM32 固定转角到位 ready: 03 04 00 00 00 30")

                                # 状态24：STM32 慢速扫描，香橙派检测到黑框大致居中后发停止扫描
                                elif task_state == 0x24:
                                    sys_state = SYS_SEARCH_BOARD
                                    search_center_stable = 0
                                    print("进入搜索画板阶段，检测黑框大致居中后发送停止扫描")

                                clear_rx(ser, rx_queue, rx_lock)
                    else:
                        if task_state == 0x25:
                            target_center_stable = 0
                else:
                    target_center_stable = 0

                if sys_state == SYS_RECOGNIZE_TARGET:
                    set_latest_packet(latest_packet_holder, packet_lock, None)

            # 发挥25：目标物已找到通知帧连续发送一小段时间，不等待 ACK。
            elif sys_state == SYS_NOTIFY_TARGET_FOUND:
                set_latest_packet(latest_packet_holder, packet_lock, TARGET_FOUND_FRAME)

                if time.time() - target_found_notify_start >= TARGET_FOUND_NOTIFY_TIME:
                    sys_state = SYS_SEARCH_BOARD
                    search_center_stable = 0
                    target_center_stable = 0
                    target_found_notify_start = 0.0
                    set_latest_packet(latest_packet_holder, packet_lock, None)
                    print("发挥25：目标物通知完成，继续搜索空白画板")

            # 发挥24/25：搜索黑框，大致居中后让 STM32 停止扫描
            elif sys_state == SYS_SEARCH_BOARD:
                if is_board_roughly_centered(center_offset):
                    search_center_stable += 1
                else:
                    search_center_stable = 0

                if search_center_stable >= SEARCH_STABLE_FRAMES:
                    set_latest_packet(latest_packet_holder, packet_lock, STOP_SCAN_FRAME)
                    sys_state = SYS_WAIT_BOARD_READY
                    print("黑框大致居中，发送停止扫描帧，等待 STM32 停稳 ready")
                    clear_rx(ser, rx_queue, rx_lock)
                else:
                    set_latest_packet(latest_packet_holder, packet_lock, None)

            # 移动到起点：激光关闭，发送 02...04
            # 新增保护：到达起点后必须在允许误差范围内连续稳定一段时间，才请求开激光。
            # 这样可以避免云台刚好扫过起点的一瞬间就开激光，导致 UV 纸留下偏起笔痕迹。
            elif sys_state == SYS_MOVE_TO_START:
                if target_points and laser_cm is not None:
                    target_cm = target_points[0]
                    dx = target_cm[0] - laser_cm[0]
                    dy = target_cm[1] - laser_cm[1]
                    err = math.hypot(dx, dy)

                    # 稳定等待期间仍然持续发送 0x02 空走控制帧，激光保持关闭，
                    # STM32 继续用 PID 把误差压小。
                    set_latest_packet(latest_packet_holder, packet_lock, build_move_packet(dx, dy))

                    now_stable = time.time()

                    if err <= START_STABLE_DISTANCE_CM:
                        if start_stable_since <= 0.0:
                            start_stable_since = now_stable
                            start_stable_printed = False
                            print("已进入起点稳定范围，开始计时：err=%.2fcm，要求稳定 %.2fs" % (err, START_STABLE_TIME_SEC))

                        stable_time = now_stable - start_stable_since

                        if (not start_stable_printed) and stable_time >= START_STABLE_TIME_SEC:
                            start_stable_printed = True

                        if stable_time >= START_STABLE_TIME_SEC:
                            sys_state = SYS_WAIT_LASER_ON_ACK
                            set_latest_packet(latest_packet_holder, packet_lock, LASER_ON_FRAME)
                            print("起点已稳定 %.2fs，发送开激光请求" % stable_time)
                            clear_rx(ser, rx_queue, rx_lock)
                    else:
                        if start_stable_since > 0.0:
                            print("起点稳定被打断：err=%.2fcm > %.2fcm，重新等待" % (err, START_STABLE_DISTANCE_CM))
                        start_stable_since = 0.0
                        start_stable_printed = False
                else:
                    start_stable_since = 0.0
                    start_stable_printed = False
                    set_latest_packet(latest_packet_holder, packet_lock, build_move_packet(0.0, 0.0))

            # 等待开激光确认：重复发送开激光请求
            elif sys_state == SYS_WAIT_LASER_ON_ACK:
                set_latest_packet(latest_packet_holder, packet_lock, LASER_ON_FRAME)

            # 正式画图：激光已打开，发送 01...05
            elif sys_state == SYS_DRAWING:
                if target_points and laser_cm is not None and current_index < len(target_points):
                    target_cm = target_points[current_index]
                    dx = target_cm[0] - laser_cm[0]
                    dy = target_cm[1] - laser_cm[1]
                    err = math.hypot(dx, dy)

                    set_latest_packet(latest_packet_holder, packet_lock, build_draw_packet(dx, dy))

                    now_switch = time.time()
                    if err <= ARRIVE_DISTANCE_CM and (now_switch - last_switch_time >= 0.05):
                        current_index += 1
                        last_switch_time = now_switch

                        if current_index >= len(target_points):
                            current_index = len(target_points) - 1
                            sys_state = SYS_WAIT_LASER_OFF_ACK
                            set_latest_packet(latest_packet_holder, packet_lock, LASER_OFF_FRAME)
                            print("已到达终点，发送关激光请求")
                            clear_rx(ser, rx_queue, rx_lock)
                else:
                    set_latest_packet(latest_packet_holder, packet_lock, build_draw_packet(0.0, 0.0))

            # 等待关激光确认：重复发送关激光请求
            elif sys_state == SYS_WAIT_LASER_OFF_ACK:
                set_latest_packet(latest_packet_holder, packet_lock, LASER_OFF_FRAME)

            # 空闲：不发送任何数据
            elif sys_state == SYS_IDLE:
                set_latest_packet(latest_packet_holder, packet_lock, None)

            # 终端打印
            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL:
                pkt = latest_packet_holder.get("packet", None)
                pkt_s = "None" if pkt is None else pkt.hex(" ")
                print("SYS=%s TASK=0x%02X idx=%d/%d laser=%s target=%s dx=%s dy=%s packet=%s" % (
                    STATE_NAME.get(sys_state, "?"),
                    task_state,
                    current_index,
                    len(target_points),
                    "None" if laser_cm is None else "(%.2f,%.2f)" % laser_cm,
                    "None" if target_cm is None else "(%.2f,%.2f)" % target_cm,
                    "None" if dx is None else "%.2f" % dx,
                    "None" if dy is None else "%.2f" % dy,
                    pkt_s
                ))
                last_print_time = now

            # 显示窗口
            if gui_enabled and SHOW_RESULT:
                result = draw_result_with_cross(
                    frame,
                    ordered_points,
                    center_offset,
                    debug_info,
                    fps,
                    sys_state,
                    task_state
                )
                cv2.imshow("result", result)

            if gui_enabled and warped_a4 is not None:
                warp_show = draw_tracker_warp(
                    warped_a4,
                    target_points,
                    current_index,
                    laser_px,
                    laser_cm,
                    target_cm,
                    dx,
                    dy,
                    sys_state,
                    task_state,
                    shape_info_saved
                )
                cv2.imshow("board_warp", warp_show)

            if gui_enabled and SHOW_SHAPE and shape_show is not None:
                cv2.imshow("shape_result", shape_show)

            if gui_enabled and SHOW_SHAPE_BINARY and shape_binary is not None:
                cv2.imshow("shape_binary", shape_binary)

            # FPS 统计
            frame_count += 1
            if now - fps_start_time >= 1.0:
                fps = frame_count / (now - fps_start_time)
                frame_count = 0
                fps_start_time = now

            elapsed = time.time() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if gui_enabled:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    sys_state = SYS_IDLE
                    task_state = 0x00
                    target_points = []
                    current_index = 0
                    shape_info_saved = None
                    search_center_stable = 0
                    target_center_stable = 0
                    target_found_notify_start = 0.0
                    start_stable_since = 0.0
                    start_stable_printed = False
                    set_latest_packet(latest_packet_holder, packet_lock, None)
                    clear_rx(ser, rx_queue, rx_lock)
                    print("手动重置为空闲状态")

    except KeyboardInterrupt:
        print("程序被手动停止")

    finally:
        stop_event.set()

        if send_thread is not None:
            send_thread.join(timeout=0.2)
        if recv_thread is not None:
            recv_thread.join(timeout=0.2)

        if ser is not None:
            ser.close()

        cap.release()

        if gui_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
