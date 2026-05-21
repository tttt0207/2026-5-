import cv2
import numpy as np
import time
import os


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
OUTER_SCALE_X = 1.08
OUTER_SCALE_Y = 1.04

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
BORDER_CM = 1.0
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


def detect_rectangle(raw_image):
    """
    检测 A4 黑框。

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

        # 这里要放宽，因为 A4 里面有黑色几何图形
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

def main():
    cap = open_camera()

    if cap is None:
        return

    # 如果没有桌面显示环境，自动关闭 imshow，避免 systemd 自启动时报错。
    # 你在香橙派桌面/PyCharm里运行时，一般 DISPLAY 会存在，窗口会正常显示。
    gui_enabled = USE_GUI and (os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None)

    if USE_GUI and not gui_enabled:
        print("未检测到 DISPLAY/WAYLAND_DISPLAY，已自动关闭图像窗口显示")
        print("如果你是在桌面环境运行但仍不显示，可在终端执行：echo $DISPLAY")

    if gui_enabled:
        if SHOW_RESULT:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        if SHOW_SHAPE:
            cv2.namedWindow("shape_result", cv2.WINDOW_NORMAL)
        if SHOW_SHAPE_BINARY:
            cv2.namedWindow("shape_binary", cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    last_print_time = 0

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

            ordered_points, center_offset, black_mask, debug_info = detect_rectangle(frame)

            if gui_enabled and SHOW_RESULT:
                result = draw_result(
                    frame,
                    ordered_points,
                    center_offset,
                    debug_info,
                    fps
                )
                cv2.imshow("result", result)

            if ordered_points is not None:
                # 优先用黑框内孔四点做透视矫正。
                # ordered_points 是由内孔外扩估算出来的外框点，不如真实内孔角点稳定。
                if debug_info is not None and "inner_points" in debug_info:
                    warped_a4, _ = warp_a4_by_inner_points(
                        frame,
                        debug_info["inner_points"]
                    )
                else:
                    warped_a4, _ = warp_a4_by_outer_points(
                        frame,
                        ordered_points
                    )

                shape_info, shape_show, shape_binary = detect_shape_in_warped(warped_a4)

                if gui_enabled and SHOW_SHAPE:
                    cv2.imshow("shape_result", shape_show)

                if gui_enabled and SHOW_SHAPE_BINARY:
                    cv2.imshow("shape_binary", shape_binary)

                # 每秒打印一次识别结果，避免刷屏
                now_print = time.time()
                if shape_info is not None and now_print - last_print_time >= 1.0:
                    print(round_shape_info(shape_info))
                    last_print_time = now_print

            frame_count += 1
            now = time.time()

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

    except KeyboardInterrupt:
        print("程序被手动停止")

    finally:
        cap.release()
        if gui_enabled:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
