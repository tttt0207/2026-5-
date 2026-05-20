import cv2
import numpy as np
import time


# =========================
# Windows 参数区
# =========================

CAMERA_ID = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAMERA_FPS = 30
TARGET_PROCESS_FPS = 30

# 黑色阈值
# 黑色图形/黑框识别不完整：调大，例如 100、110
# 白纸阴影也变黑：调小，例如 70、80
BLACK_V_UPPER = 90

# 当前黑框参数
OUTER_SCALE_X = 1.08
OUTER_SCALE_Y = 1.04

# 只显示这三个窗口
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

# A4 黑框：2cm
BORDER_CM = 2.0
BORDER_PX_X = int(BORDER_CM / CM_PER_PIXEL_X)  # 40px
BORDER_PX_Y = int(BORDER_CM / CM_PER_PIXEL_Y)  # 40px

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
        if center_black_ratio > 0.20:
            continue

        outer_points = expand_inner_to_outer(inner_points)

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
    margin_x = 30
    margin_y = 30

    x1 = margin_x
    y1 = margin_y
    x2 = WARP_WIDTH - margin_x
    y2 = WARP_HEIGHT - margin_y

    inner = warped_a4[y1:y2, x1:x2]

    return inner, x1, y1


def classify_quad(pts):
    """
    四边形分类：正方形 or 梯形
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

    cos1 = angle_cos(bl, tl, tr)
    cos2 = angle_cos(tl, tr, br)
    cos3 = angle_cos(tr, br, bl)
    cos4 = angle_cos(br, bl, tl)

    max_cos = max(cos1, cos2, cos3, cos4)

    # 正方形：四边接近相等，四个角接近 90 度
    if min_side / max_side > 0.75 and max_cos < 0.35:
        side_px = sum(sides) / 4.0

        return "square", {
            "side_px": side_px,
            "side_cm": side_px * CM_PER_PIXEL_X
        }

    # 梯形：四边形但不是正方形
    height_px = (distance(tl, bl) + distance(tr, br)) / 2.0

    return "trapezoid", {
        "top_px": top,
        "bottom_px": bottom,
        "height_px": height_px,

        "top_cm": top * CM_PER_PIXEL_X,
        "bottom_cm": bottom * CM_PER_PIXEL_X,
        "height_cm": height_px * CM_PER_PIXEL_Y
    }


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

        if area < 300:
            continue

        child_area = cv2.contourArea(contours[child])

        if child_area < 100:
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

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

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

    # 如果没有找到内孔，兜底使用最大外轮廓。
    # 正常空心图形应该能找到 inner_contour。
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
        return None, debug_show, binary

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
        return None, debug_show, binary

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
    elif vertices == 3:
        pts = approx2.reshape(3, 2).astype(np.float32)

        side1 = distance(pts[0], pts[1])
        side2 = distance(pts[1], pts[2])
        side3 = distance(pts[2], pts[0])

        outer_side_avg = (side1 + side2 + side3) / 3.0

        side_center_px = get_triangle_side_centerline(outer_contour, inner_contour)

        # 如果内轮廓拟合失败，退化为外边长减 1cm
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
    elif vertices == 4:
        pts = approx2.reshape(4, 2).astype(np.float32)

        shape_name, extra = classify_quad(pts)
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
    text2 = "Center: %.2f, %.2f cm" % (
        shape_info["center_rel_cm"][0],
        shape_info["center_rel_cm"][1]
    )
    text3 = "V: %d  C: %.2f" % (
        shape_info["vertices"],
        shape_info["circularity"]
    )

    if shape_info["shape"] == "circle":
        text4 = "Centerline Diameter: %.2f cm" % shape_info["diameter_cm"]

    elif shape_info["shape"] == "triangle":
        text4 = "Centerline Side: %.2f cm" % shape_info["side_cm"]

    elif shape_info["shape"] == "square":
        text4 = "Centerline Side: %.2f cm" % shape_info["side_cm"]

    elif shape_info["shape"] == "trapezoid":
        text4 = "Centerline Top: %.2f  Bottom: %.2f  H: %.2f cm" % (
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

    return shape_info, debug_show, binary


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
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("摄像头打开失败，请尝试把 CAMERA_ID 改成 0、1、2、3")
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print("摄像头打开成功")
    print("CAMERA_ID:", CAMERA_ID)
    print("请求分辨率: %d x %d" % (FRAME_WIDTH, FRAME_HEIGHT))
    print("请求FPS:", CAMERA_FPS)
    print("实际分辨率: %.0f x %.0f" % (real_w, real_h))
    print("摄像头报告FPS:", real_fps)
    print("程序处理FPS限制:", TARGET_PROCESS_FPS)
    print("标准 A4: %.1fcm x %.1fcm" % (A4_WIDTH_CM, A4_HEIGHT_CM))
    print("标准黑框: %.1fcm = %dpx" % (BORDER_CM, BORDER_PX_X))
    print("图形线宽: %.1fcm = %dpx，尺寸按中心线测量" % (FIGURE_LINE_WIDTH_CM, FIGURE_LINE_WIDTH_PX))

    return cap


def main():
    cap = open_camera()

    if cap is None:
        return

    frame_count = 0
    fps = 0.0
    fps_start_time = time.time()
    last_print_time = 0

    target_period = 1.0 / TARGET_PROCESS_FPS

    while True:
        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print("读取摄像头失败")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        ordered_points, center_offset, black_mask, debug_info = detect_rectangle(frame)

        if SHOW_RESULT:
            result = draw_result(
                frame,
                ordered_points,
                center_offset,
                debug_info,
                fps
            )
            cv2.imshow("result", result)

        if ordered_points is not None:
            warped_a4, _ = warp_a4_by_outer_points(
                frame,
                ordered_points
            )

            if SHOW_SHAPE:
                shape_info, shape_show, shape_binary = detect_shape_in_warped(warped_a4)

                cv2.imshow("shape_result", shape_show)

                if SHOW_SHAPE_BINARY:
                    cv2.imshow("shape_binary", shape_binary)

                # 每秒打印一次识别结果，避免刷屏
                now_print = time.time()
                if shape_info is not None and now_print - last_print_time >= 1.0:
                    print(shape_info)
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

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
