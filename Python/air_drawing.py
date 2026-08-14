import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

BRUSH_SIZE = 6
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 30

COLORS = {
    "BLUE": (130, 35, 0),
    "GREEN": (0, 105, 0),
    "YELLOW": (0, 150, 150),
    "ORANGE": (0, 70, 180),
    "BROWN": (20, 45, 85),
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "RED": (0, 0, 140),
    "PURPLE": (105, 20, 95),
    "PINK": (130, 25, 150),
}

current_color_name = "RED"
toolbar_visible = False
toolbar_hold_frames = 0
TOOLBAR_CONFIRMATION = 10
toolbar_rects = None

# Palette interaction settings.
# These were missing in the previous version.
PALETTE_HOLD_FRAMES = 10
PALETTE_COOLDOWN_FRAMES = 18
palette_hold_frames = 0
palette_selection_cooldown = 0

# Phase 4 UI feedback.
COLOR_MESSAGE_DURATION = 45
color_message_text = ""
color_message_timer = 0

# Accessible color selection.
# ============================================================
# PALETTE AUTO-SELECTION
# ============================================================

hovered_color_name = None
recent_colors = []
MAX_RECENT_COLORS = 3

# Color must be continuously hovered for this long.
# 0.65 sec = fast but still intentional.
PALETTE_SELECT_TIME = 0.65

# Current color being "loaded"
palette_loading_color = None
palette_loading_start = None

# Prevents the palette from immediately selecting another
# color after one has already been selected.
#palette_selection_lock = False

TWO_FINGER_CONFIRMATION = 5
two_finger_count = 0
previous_two_finger_active = False

# Level 7 - snap-to-clear
fade_clear_active = False
FADE_RATE = 0.82

# Level 8: particle fade


two_finger_active = False

# Lower = more responsive
# Higher = smoother but more delayed
SMOOTHING = 0.38
STROKE_SMOOTHING_WINDOW = 2
STROKE_MIN_DISTANCE = 1.0

# Prevents a bad tracking jump from creating
# a huge unwanted line
MAX_JUMP = 90

# Gesture stability
START_CONFIRMATION = 3
STOP_CONFIRMATION = 3


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,

    min_hand_detection_confidence=0.50,
    min_hand_presence_confidence=0.50,
    min_tracking_confidence=0.50
)


landmarker = HandLandmarker.create_from_options(
    options
)


if __name__ == "__main__":

    # ============================================================
    # CAMERA
    # ============================================================

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        landmarker.close()

        raise SystemExit


    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    print(
        f"Camera resolution: "
        f"{actual_width} x {actual_height}"
    )


# ============================================================
# DRAWING VARIABLES
# ============================================================

canvas = None
canvas_mask = None

previous_point = None

smoothed_point = None
stroke_points = []

drawing = False

valid_frames = 0

invalid_frames = 0

undo_stack = []
redo_stack = []
mask_undo_stack = []
mask_redo_stack = []
MAX_HISTORY = 30


# ============================================================
# TIME
# ============================================================

start_time = time.monotonic()


if __name__ == "__main__":

    # ============================================================
    # WINDOW
    # ============================================================

    WINDOW_NAME = "Air Drawing - Level 11 Fixed Controls"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):

    """
    Calculates angle ABC.
    """

    ba = np.array(
        [
            a.x - b.x,
            a.y - b.y
        ],
        dtype=np.float32
    )

    bc = np.array(
        [
            c.x - b.x,
            c.y - b.y
        ],
        dtype=np.float32
    )

    ba_length = np.linalg.norm(ba)
    bc_length = np.linalg.norm(bc)

    if ba_length == 0 or bc_length == 0:

        return 0.0

    cosine = np.dot(
        ba,
        bc
    ) / (
        ba_length *
        bc_length
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


def point_distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def is_index_draw_gesture(hand):
    """
    Reliable drawing gesture for V1.

    DRAW only when the index finger is clearly extended.
    The other fingers are explicitly checked so a closed fist,
    thumb-only gesture, two-finger gesture, or open hand does not draw.

    MediaPipe landmark IDs:
      0  wrist
      5-8   index
      9-12  middle
      13-16 ring
      17-20 pinky
    """

    wrist = hand[0]

    # ---- Index geometry ----
    index_mcp = hand[5]
    index_pip = hand[6]
    index_dip = hand[7]
    index_tip = hand[8]

    pip_angle = calculate_angle(
        index_mcp,
        index_pip,
        index_dip
    )

    dip_angle = calculate_angle(
        index_pip,
        index_dip,
        index_tip
    )

    # How far the index tip is from the wrist/palm.
    index_tip_wrist = point_distance(wrist, index_tip)
    index_pip_wrist = point_distance(wrist, index_pip)

    # How far the fingertip is from its MCP.
    index_tip_mcp = point_distance(index_mcp, index_tip)
    index_pip_mcp = point_distance(index_mcp, index_pip)

    # A genuinely extended index has a substantially longer
    # fingertip-to-wrist distance and fingertip-to-MCP distance.
    index_extended = (
        pip_angle > 150
        and
        dip_angle > 150
        and
        index_tip_wrist > index_pip_wrist * 1.18
        and
        index_tip_mcp > index_pip_mcp * 1.20
    )

    if not index_extended:
        return False

    # ---- Other fingers ----
    # They must NOT be clearly extended. This prevents:
    # ✌️, 🖐️ and similar gestures from drawing.
    middle_extended = is_finger_extended_simple(
        hand, 9, 10, 11, 12
    )

    ring_extended = is_finger_extended_simple(
        hand, 13, 14, 15, 16
    )

    pinky_extended = is_finger_extended_simple(
        hand, 17, 18, 19, 20
    )

    return (
        not middle_extended
        and
        not ring_extended
        and
        not pinky_extended
    )


def is_finger_extended_simple(hand, mcp, pip, dip, tip):
    """Supporting test used only for middle/ring/pinky."""

    pip_angle = calculate_angle(
        hand[mcp],
        hand[pip],
        hand[dip]
    )

    dip_angle = calculate_angle(
        hand[pip],
        hand[dip],
        hand[tip]
    )

    tip_wrist = point_distance(
        hand[0],
        hand[tip]
    )

    pip_wrist = point_distance(
        hand[0],
        hand[pip]
    )

    tip_mcp = point_distance(
        hand[mcp],
        hand[tip]
    )

    pip_mcp = point_distance(
        hand[mcp],
        hand[pip]
    )

    return (
        pip_angle > 150
        and
        dip_angle > 150
        and
        tip_wrist > pip_wrist * 1.15
        and
        tip_mcp > pip_mcp * 1.15
    )


def get_palm_center(hand, frame_width, frame_height):
    """Return the center of the palm for palette selection."""

    palm_ids = (
        0,   # wrist
        5,   # index MCP
        9,   # middle MCP
        13,  # ring MCP
        17   # pinky MCP
    )

    x = sum(hand[i].x for i in palm_ids) / len(palm_ids)
    y = sum(hand[i].y for i in palm_ids) / len(palm_ids)

    return (
        int(np.clip(x, 0.0, 1.0) * frame_width),
        int(np.clip(y, 0.0, 1.0) * frame_height)
    )


def is_open_palm_gesture(hand):
    return (
        is_finger_extended_simple(hand, 5, 6, 7, 8)
        and is_finger_extended_simple(hand, 9, 10, 11, 12)
        and is_finger_extended_simple(hand, 13, 14, 15, 16)
        and is_finger_extended_simple(hand, 17, 18, 19, 20)
    )


def is_two_finger_gesture(hand):
    """Detect ✌️: index + middle extended, ring + pinky folded."""
    index_extended = is_finger_extended_simple(hand, 5, 6, 7, 8)
    middle_extended = is_finger_extended_simple(hand, 9, 10, 11, 12)
    ring_extended = is_finger_extended_simple(hand, 13, 14, 15, 16)
    pinky_extended = is_finger_extended_simple(hand, 17, 18, 19, 20)

    return (
        index_extended
        and middle_extended
        and not ring_extended
        and not pinky_extended
    )


def index_only_gesture(hand):
    if is_two_finger_gesture(hand):
        return False
    return is_index_draw_gesture(hand)

def reset_line():

    """
    Completely disconnect the next stroke.
    """

    global previous_point
    global smoothed_point
    global stroke_points

    previous_point = None
    smoothed_point = None
    stroke_points = []


def reset_gesture():

    global drawing
    global valid_frames
    global invalid_frames
    global two_finger_count
    global previous_two_finger_active
    global two_finger_active

    drawing = False

    valid_frames = 0
    invalid_frames = 0
    two_finger_count = 0
    previous_two_finger_active = False
    two_finger_active = False
    # Do not clear particles here; they are part of the fade animation.

    reset_line()


# ============================================================
# LEVEL 6 GESTURE STATE
# ============================================================

def update_two_finger_state(hand):
    global two_finger_count

    if is_two_finger_gesture(hand):
        two_finger_count += 1
    else:
        two_finger_count = 0

    return two_finger_count >= TWO_FINGER_CONFIRMATION


# ============================================================
# LEVEL 5.1 STROKE SMOOTHING
# ============================================================

def smooth_stroke_point(x, y):
    global stroke_points

    stroke_points.append((x, y))

    if len(stroke_points) > STROKE_SMOOTHING_WINDOW:
        stroke_points.pop(0)

    weights = list(range(1, len(stroke_points) + 1))
    total = sum(weights)

    sx = sum(
        p[0] * w for p, w in zip(stroke_points, weights)
    ) / total

    sy = sum(
        p[1] * w for p, w in zip(stroke_points, weights)
    ) / total

    return int(round(sx)), int(round(sy))


def point_distance_pixels(p1, p2):
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )


# ============================================================
# LEVEL 5 DRAWING TOOLS
# ============================================================

def save_canvas_state():
    global undo_stack, redo_stack
    global mask_undo_stack, mask_redo_stack

    if canvas is not None:
        undo_stack.append(canvas.copy())
        mask_undo_stack.append(canvas_mask.copy())

        if len(undo_stack) > MAX_HISTORY:
            undo_stack.pop(0)

        if len(mask_undo_stack) > MAX_HISTORY:
            mask_undo_stack.pop(0)

        redo_stack.clear()
        mask_redo_stack.clear()


def undo():
    global canvas, canvas_mask
    if undo_stack:
        redo_stack.append(canvas.copy())
        mask_redo_stack.append(canvas_mask.copy())

        canvas = undo_stack.pop()

        if mask_undo_stack:
            canvas_mask = mask_undo_stack.pop()


def redo():
    global canvas, canvas_mask
    if redo_stack:
        undo_stack.append(canvas.copy())
        mask_undo_stack.append(canvas_mask.copy())

        canvas = redo_stack.pop()

        if mask_redo_stack:
            canvas_mask = mask_redo_stack.pop()


def clear_canvas():
    global canvas, canvas_mask
    if canvas is not None:
        save_canvas_state()
        canvas = np.zeros_like(canvas)

        if canvas_mask is not None:
            canvas_mask = np.zeros_like(canvas_mask)
    

def brush_color():
    return COLORS[current_color_name]


def brush_thickness():
    return BRUSH_SIZE


def draw_dot(point):
    radius = max(brush_thickness() // 2, 1)

    cv2.circle(
        canvas,
        point,
        radius,
        brush_color(),
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        canvas_mask,
        point,
        radius,
        255,
        -1,
        cv2.LINE_AA
    )


def draw_segment(p1, p2):
    cv2.line(
        canvas,
        p1,
        p2,
        brush_color(),
        brush_thickness(),
        cv2.LINE_AA
    )

    cv2.line(
        canvas_mask,
        p1,
        p2,
        255,
        brush_thickness(),
        cv2.LINE_AA
    )


def add_recent_color(name):
    global recent_colors

    if name in recent_colors:
        recent_colors.remove(name)

    recent_colors.insert(0, name)

    if len(recent_colors) > MAX_RECENT_COLORS:
        recent_colors = recent_colors[:MAX_RECENT_COLORS]


def set_hovered_color(name):
    global hovered_color_name
    hovered_color_name = name


def clear_hovered_color():
    global hovered_color_name
    hovered_color_name = None


def show_color_selection_message(name):
    global color_message_text
    global color_message_timer

    color_message_text = f"{name} SELECTED"
    color_message_timer = COLOR_MESSAGE_DURATION


def set_color(name):
    global current_color_name
    if name in COLORS:
        current_color_name = name
        add_recent_color(name)


def brush_up():
    global BRUSH_SIZE
    BRUSH_SIZE = min(MAX_BRUSH_SIZE, BRUSH_SIZE + 2)


def brush_down():
    global BRUSH_SIZE
    BRUSH_SIZE = max(MIN_BRUSH_SIZE, BRUSH_SIZE - 2)


# ============================================================
# MAIN LOOP
# ============================================================

# ============================================================
# LEVEL 11 - FIXED COLOR TOOLBAR
# ============================================================


# ============================================================
# PALETTE AUTO-SELECTION
# ============================================================

# Time required to select a color after the palm stays
# over that color.
PALETTE_SELECT_TIME = 0.65

# Current color being hovered
palette_loading_color = None

# Time when hovering started
palette_loading_start = None

# Prevents repeated selection while the same palm gesture
# is still active.
palette_selection_lock = False


def reset_palette_loading():
    """
    Cancel the current palette loading operation.
    """

    global palette_loading_color
    global palette_loading_start

    palette_loading_color = None
    palette_loading_start = None


def update_palette_loading(selected_color):
    """
    Start or continue the loading timer for the color
    currently underneath the palm.

    Returns:
        True  -> selection completed
        False -> still loading / no selection
    """

    global palette_loading_color
    global palette_loading_start

    # Nothing is being hovered
    if selected_color is None:
        reset_palette_loading()
        return False

    now = time.monotonic()

    # Palm moved to a different color.
    # Restart the timer.
    if palette_loading_color != selected_color:

        palette_loading_color = selected_color
        palette_loading_start = now

        return False

    # Safety check
    if palette_loading_start is None:

        palette_loading_start = now

        return False

    # Calculate how long the color has been hovered.
    elapsed = now - palette_loading_start

    # Selection completed.
    if elapsed >= PALETTE_SELECT_TIME:

        reset_palette_loading()

        return True

    return False


def get_palette_loading_progress():
    """
    Returns the current palette loading progress.

    0.0 = just started
    1.0 = completely loaded
    """

    if (
        palette_loading_color is None
        or
        palette_loading_start is None
    ):
        return 0.0

    elapsed = time.monotonic() - palette_loading_start

    progress = elapsed / PALETTE_SELECT_TIME

    # Keep value between 0 and 1.
    progress = max(
        0.0,
        min(1.0, progress)
    )

    return progress


def draw_toolbar(frame):
    """Minimal professional 10-color floating palette."""

    items = [
        ("BLUE",   (190, 70, 20)),
        ("GREEN",  (40, 155, 45)),
        ("YELLOW", (0, 210, 235)),
        ("ORANGE", (0, 125, 245)),
        ("BROWN",  (55, 90, 135)),
        ("BLACK",  (0, 0, 0)),
        ("WHITE",  (255, 255, 255)),
        ("RED",    (40, 45, 220)),
        ("PURPLE", (155, 55, 150)),
        ("PINK",   (190, 75, 210)),
    ]

    # ============================================================
    # PALETTE LOADING / AUTO SELECTION
    # ============================================================

    def reset_palette_loading():
        global palette_loading_color
        global palette_loading_start

        palette_loading_color = None
        palette_loading_start = None


    def update_palette_loading(selected_color):
        """
        Start / update the loading timer for the color under
        the palm.

        Returns:
            True  -> color selection completed
            False -> still loading / nothing selected
        """

        global palette_loading_color
        global palette_loading_start

        if selected_color is None:
            reset_palette_loading()
            return False

        now = time.monotonic()

        # Palm moved onto a different color.
        # Restart the loading animation.
        if palette_loading_color != selected_color:
            palette_loading_color = selected_color
            palette_loading_start = now
            return False

        # Same color -> continue loading.
        if palette_loading_start is None:
            palette_loading_start = now
            return False

        elapsed = now - palette_loading_start

        if elapsed >= PALETTE_SELECT_TIME:
            reset_palette_loading()
            return True

        return False


    def get_palette_loading_progress():

        if (
            palette_loading_color is None
            or
            palette_loading_start is None
        ):
            return 0.0

        elapsed = time.monotonic() - palette_loading_start

        progress = elapsed / PALETTE_SELECT_TIME

        return max(
            0.0,
            min(1.0, progress)
        )

    # Compact 2 x 5 grid.
    radius = 21
    cell_w = 52
    cell_h = 52
    gap = 5

    columns = 2
    rows = 5

    h, w = frame.shape[:2]

    panel_w = columns * cell_w + gap + 20
    panel_h = rows * cell_h + (rows - 1) * gap + 20

    right = w - 18
    left = right - panel_w
    top = max(18, (h - panel_h) // 2)

    # Soft dark floating panel.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (left, top),
        (right, top + panel_h),
        (22, 22, 22),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame
    )

    # Thin neutral outline.
    cv2.rectangle(
        frame,
        (left, top),
        (right, top + panel_h),
        (75, 75, 75),
        1
    )

    rects = []

    for i, (name, color) in enumerate(items):

        row = i // columns
        col = i % columns

        cx = left + 10 + radius + col * (cell_w + gap)
        cy = top + 10 + radius + row * (cell_h + gap)

        x1 = cx - radius
        y1 = cy - radius
        x2 = cx + radius
        y2 = cy + radius

        # Hit area remains rectangular for reliable palm selection.
        rects.append((name, x1, y1, x2, y2))

        # Very subtle shadow.
        cv2.circle(
            frame,
            (cx + 1, cy + 2),
            radius,
            (0, 0, 0),
            -1,
            cv2.LINE_AA
        )

        # Color circle.
        cv2.circle(
            frame,
            (cx, cy),
            radius - 2,
            color,
            -1,
            cv2.LINE_AA
        )

        # Neutral outline for light colors and black.
        if name == "BLACK":
            border = (175, 175, 175)
            thickness = 2
        elif name == "WHITE":
            border = (120, 120, 120)
            thickness = 2
        elif name == "RED":
            border = (70, 0, 0)
            thickness = 1
        else:
            border = (35, 35, 35)
            thickness = 1

        cv2.circle(
            frame,
            (cx, cy),
            radius - 2,
            border,
            thickness,
            cv2.LINE_AA
        )

        # Hover preview: shows what would be selected.
        # ============================================================
# PALM HOVER + LOADING RING
# ============================================================

    # ============================================================
# PALETTE LOADING INDICATOR
# ============================================================
# When the palm stays over a color, a thin light-green
# progress ring appears around that color.
# ============================================================

        if name == hovered_color_name:

            progress = get_palette_loading_progress()

            if progress > 0.0:

                # Light green in OpenCV BGR format.
                loading_color = (170, 255, 170)

                # Very thin ring outside the color.
                loading_radius = radius + 5

                # Draw only the portion that has loaded.
                cv2.ellipse(
                    frame,
                    (cx, cy),
                    (loading_radius, loading_radius),
                    -90,
                    0,
                    int(360 * progress),
                    loading_color,
                    2,
                    cv2.LINE_AA
                )

            # Loading progress
            progress = get_palette_loading_progress()

            if progress > 0.0:

                # Full ring = 360 degrees.
                end_angle = int(360 * progress)

                cv2.ellipse(
                    frame,
                    (cx, cy),
                    (radius + 6, radius + 6),
                    -90,
                    0,
                    end_angle,
                    (170, 255, 170),
                    4,
                    cv2.LINE_AA
                )

            # Selected color = single clean white ring.
            if name == current_color_name:
                cv2.circle(
                    frame,
                    (cx, cy),
                    radius + 4,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA
                )

    return rects


def toolbar_hit(x, y, rects):
    """Use larger invisible hit zones around compact swatches."""
    padding = 5

    for name, x1, y1, x2, y2 in rects:
        if (
            x1 - padding <= x <= x2 + padding
            and
            y1 - padding <= y <= y2 + padding
        ):
            return name

    return None


def update_palette_visibility(open_palm_detected):
    """
    Open palm ONLY controls palette visibility.

    It NEVER:
      - changes current_color_name
      - resets the selected color
      - changes brush size
    """
    global toolbar_visible
    global toolbar_hold_frames

    if open_palm_detected:
        toolbar_hold_frames += 1

        if toolbar_hold_frames >= PALETTE_HOLD_FRAMES:
            toolbar_visible = True
    else:
        toolbar_hold_frames = 0


def close_palette_after_selection():
    """
    Hide the palette ONLY after a color has actually been selected.
    The selected color remains active.
    """
    global toolbar_visible
    global toolbar_hold_frames
    global toolbar_rects

    toolbar_visible = False
    toolbar_hold_frames = 0
    toolbar_rects = None

# ============================================================
# PHASE 3 — SMART SHAPE RECOGNITION
# ============================================================

SHAPE_RECOGNITION_ENABLED = False
SHAPE_MIN_POINTS = 12
SHAPE_MIN_SIZE = 35
LINE_STRAIGHTNESS_THRESHOLD = 0.96
CIRCLE_CIRCULARITY_THRESHOLD = 0.72
CLOSED_SHAPE_THRESHOLD = 0.08
SHAPE_MESSAGE_DURATION = 45

current_stroke_points = []
last_recognized_shape = None
shape_message_timer = 0


def _shape_distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _shape_path_length(points):
    return sum(_shape_distance(points[i - 1], points[i])
               for i in range(1, len(points)))


def _shape_size(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _shape_is_closed(points):
    if len(points) < 3:
        return False
    length = _shape_path_length(points)
    if length <= 0:
        return False
    return _shape_distance(points[0], points[-1]) / length < CLOSED_SHAPE_THRESHOLD


def _detect_line(points):
    if len(points) < SHAPE_MIN_POINTS:
        return None

    start = np.asarray(points[0], dtype=np.float32)
    end = np.asarray(points[-1], dtype=np.float32)
    direct = np.linalg.norm(end - start)
    path = _shape_path_length(points)

    if direct < SHAPE_MIN_SIZE or path <= 0:
        return None

    if direct / path >= LINE_STRAIGHTNESS_THRESHOLD:
        return ("LINE", (tuple(start.astype(int)), tuple(end.astype(int))))

    return None


def _detect_circle(points):
    if len(points) < SHAPE_MIN_POINTS or not _shape_is_closed(points):
        return None

    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter <= 0 or area <= 0:
        return None

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)

    if circularity < CIRCLE_CIRCULARITY_THRESHOLD:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    if radius < SHAPE_MIN_SIZE / 2:
        return None

    return ("CIRCLE", (int(cx), int(cy), int(radius)))


def _detect_polygon(points):
    if len(points) < SHAPE_MIN_POINTS or not _shape_is_closed(points):
        return None

    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, perimeter * 0.04, True)
    pts = [tuple(map(int, p[0])) for p in approx]

    if len(pts) == 3:
        return ("TRIANGLE", pts)

    if len(pts) == 4:
        return ("RECTANGLE", pts)

    return None


def _detect_arrow(points):
    if len(points) < SHAPE_MIN_POINTS:
        return None

    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(
        contour,
        max(3.0, _shape_path_length(points) * 0.025),
        False
    )

    pts = [tuple(map(int, p[0])) for p in simplified]

    if len(pts) < 4:
        return None

    start = np.asarray(pts[0], dtype=np.float32)
    tip = np.asarray(pts[-1], dtype=np.float32)
    overall = np.linalg.norm(tip - start)

    if overall < SHAPE_MIN_SIZE:
        return None

    # Require a sharp direction change near the end of the stroke.
    best_angle = 0

    for i in range(1, len(pts) - 1):
        a = np.asarray(pts[i - 1], dtype=np.float32)
        b = np.asarray(pts[i], dtype=np.float32)
        c = np.asarray(pts[i + 1], dtype=np.float32)

        v1 = b - a
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            continue

        cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        best_angle = max(best_angle, math.degrees(math.acos(cosine)))

    if best_angle < 35:
        return None

    head_size = max(12, int(overall * 0.15))
    nearby = [
        p for p in pts[:-1]
        if _shape_distance(p, pts[-1]) < head_size * 2
    ]

    if len(nearby) < 2:
        return None

    separation = max(
        _shape_distance(nearby[i], nearby[j])
        for i in range(len(nearby))
        for j in range(i + 1, len(nearby))
    )

    if separation < head_size * 0.35:
        return None

    return ("ARROW", (tuple(start.astype(int)), tuple(tip.astype(int))))


def recognize_shape(points):
    if not SHAPE_RECOGNITION_ENABLED:
        return None

    if len(points) < SHAPE_MIN_POINTS or _shape_size(points) < SHAPE_MIN_SIZE:
        return None

    result = _detect_circle(points)
    if result:
        return result

    result = _detect_polygon(points)
    if result:
        return result

    result = _detect_arrow(points)
    if result:
        return result

    return _detect_line(points)


def _draw_recognized_shape(result):
    if result is None:
        return False

    name, data = result
    color = brush_color()
    thickness = brush_thickness()

    if name == "LINE":
        p1, p2 = data
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.line(canvas_mask, p1, p2, 255, thickness, cv2.LINE_AA)
        return True

    if name == "CIRCLE":
        cx, cy, radius = data
        cv2.circle(canvas, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas_mask, (cx, cy), radius, 255, thickness, cv2.LINE_AA)
        return True

    if name in ("TRIANGLE", "RECTANGLE"):
        pts = np.asarray(data, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        cv2.polylines(canvas_mask, [pts], True, 255, thickness, cv2.LINE_AA)
        return True

    if name == "ARROW":
        start, tip = data
        start = np.asarray(start, dtype=np.float32)
        tip = np.asarray(tip, dtype=np.float32)

        direction = tip - start
        length = np.linalg.norm(direction)

        if length == 0:
            return False

        direction /= length
        perpendicular = np.asarray(
            [-direction[1], direction[0]], dtype=np.float32
        )

        arrow_size = max(15, int(length * 0.12))
        left = tip - direction * arrow_size + perpendicular * arrow_size * 0.65
        right = tip - direction * arrow_size - perpendicular * arrow_size * 0.65

        cv2.line(canvas, tuple(start.astype(int)), tuple(tip.astype(int)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(canvas, tuple(tip.astype(int)), tuple(left.astype(int)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(canvas, tuple(tip.astype(int)), tuple(right.astype(int)),
                 color, thickness, cv2.LINE_AA)

        cv2.line(canvas_mask, tuple(start.astype(int)), tuple(tip.astype(int)),
                 255, thickness, cv2.LINE_AA)
        cv2.line(canvas_mask, tuple(tip.astype(int)), tuple(left.astype(int)),
                 255, thickness, cv2.LINE_AA)
        cv2.line(canvas_mask, tuple(tip.astype(int)), tuple(right.astype(int)),
                 255, thickness, cv2.LINE_AA)

        return True

    return False


def finish_smart_stroke():
    global current_stroke_points
    global last_recognized_shape
    global shape_message_timer

    points = current_stroke_points[:]
    current_stroke_points = []

    result = recognize_shape(points)

    if result is None:
        last_recognized_shape = None
        shape_message_timer = 0
        return

    # undo_stack[-1] is the canvas state saved immediately before this stroke.
    if undo_stack:
        canvas[:] = undo_stack[-1]
        if mask_undo_stack:
            canvas_mask[:] = mask_undo_stack[-1]

    if _draw_recognized_shape(result):
        last_recognized_shape = result[0]
        shape_message_timer = SHAPE_MESSAGE_DURATION


def phase3_shape_status(output):
    global shape_message_timer

    if shape_message_timer > 0 and last_recognized_shape:
        cv2.putText(
            output,
            f"SMART SHAPE: {last_recognized_shape}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        shape_message_timer -= 1


def _process_one_frame(frame):

    """
    Runs one frame through hand tracking, gesture logic, and
    drawing/HUD rendering. Shared by both the desktop OpenCV
    window loop and the web server's process_frame().
    `frame` must be a raw BGR frame (not yet mirrored).
    Returns the fully composited BGR output frame.
    """
    global _, baseline, canvas, canvas_mask, color_message_timer, current_stroke_points
    global cursor_color, cursor_radius, drawing, erase_gesture_valid, erase_radius, gesture_valid
    global hand, ink, invalid_frames, mode_text, movement, mp_image
    global new_x, new_y, notification_text, nx, ny, old_x
    global old_y, open_palm, palette_selection_lock, palm_cursor_radius, palm_x, palm_y
    global ph, preview_text, previous_point, previous_two_finger_active, previous_x, previous_y
    global pw, px, py, raw_x, raw_y, recent_color
    global recent_name, recent_x, recent_y, result, rgb_frame, selected
    global selection_complete, smoothed_point, status_color, status_dot, status_h, status_overlay
    global status_radius, status_text, status_w, status_x, status_y, stroke_x
    global stroke_y, th, timestamp_ms, tip, toast_h, toast_overlay
    global toast_w, toolbar_rects, tw, two_finger_active, valid_frames, x
    global y

    palm_x = -100
    palm_y = -100

    # ====================================================
    # MIRROR CAMERA
    # ====================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ====================================================
    # CREATE CANVAS
    # ====================================================
    #
    # Re-created whenever the incoming frame's resolution
    # changes (not just on the very first frame). Without this
    # check, a canvas built for one resolution gets reused
    # against frames of a different size — e.g. if the browser
    # sends a differently-sized frame than the first one — and
    # every mask/compositing operation below throws a shape
    # mismatch.

    if canvas is None or canvas.shape[:2] != frame.shape[:2]:

        canvas = np.zeros_like(
            frame
        )

        canvas_mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        undo_stack.clear()
        redo_stack.clear()
        mask_undo_stack.clear()
        mask_redo_stack.clear()


    # ====================================================
    # MEDIAPIPE IMAGE
    # ====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ====================================================
    # TIMESTAMP
    # ====================================================

    timestamp_ms = int(
        (
            time.monotonic()
            -
            start_time
        )
        * 1000
    )


    # ====================================================
    # HAND TRACKING
    # ====================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ====================================================
    # HAND FOUND
    # ====================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # =================================================
        # INDEX FINGERTIP
        # =================================================

        tip = hand[8]


        raw_x = int(
            np.clip(
                tip.x,
                0.0,
                1.0
            )
            *
            frame.shape[1]
        )


        raw_y = int(
            np.clip(
                tip.y,
                0.0,
                1.0
            )
            *
            frame.shape[0]
        )


        # =================================================
        # SMOOTHING
        # =================================================

        if smoothed_point is None:

            smoothed_point = (
                raw_x,
                raw_y
            )

        else:

            old_x, old_y = (
                smoothed_point
            )


            new_x = int(
                old_x
                +
                SMOOTHING
                *
                (
                    raw_x
                    -
                    old_x
                )
            )


            new_y = int(
                old_y
                +
                SMOOTHING
                *
                (
                    raw_y
                    -
                    old_y
                )
            )


            smoothed_point = (
                new_x,
                new_y
            )


        x, y = smoothed_point

        stroke_x, stroke_y = smooth_stroke_point(x, y)


        # =================================================
        # CHECK DRAWING / ERASER GESTURE
        # =================================================

        two_finger_active = update_two_finger_state(hand)

        # ✌️ uses the index fingertip as the eraser cursor.
        open_palm = is_open_palm_gesture(hand)

        if not open_palm:
            palette_selection_lock = False
            reset_palette_loading()

        # =================================================
        #
        # OPEN PALM:
        #   - Opens the palette.
        #   - The PALM CENTER selects a color.
        #   - Does NOT draw.
        #
        # INDEX ONLY:
        #   - Draws using the previously selected color.
        #   - Does NOT change color automatically.
        # =================================================

        update_palette_visibility(open_palm)

        gesture_valid = index_only_gesture(hand)

        # ============================================================
        # PALM POSITION
        # Always calculate this before any palette UI uses it.
        # ============================================================

        palm_x, palm_y = get_palm_center(
            hand,
            frame.shape[1],
            frame.shape[0]
        )



        if toolbar_visible:

            # FIRST create the palette hitboxes
            toolbar_rects = draw_toolbar(frame)

            if open_palm:

                gesture_valid = False
                drawing = False
                reset_line()

                palm_x, palm_y = get_palm_center(
                    hand,
                    frame.shape[1],
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (palm_x, palm_y),
                    14,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # NOW use the palette rectangles
                selected = toolbar_hit(
                    palm_x,
                    palm_y,
                    toolbar_rects
                )

                if selected is not None:

                    set_hovered_color(selected)

                    selection_complete = update_palette_loading(
                        selected
                    )

                    if (
                        selection_complete
                        and
                        not palette_selection_lock
                    ):

                        set_color(selected)

                        show_color_selection_message(
                            selected
                        )

                        palette_selection_lock = True

                        close_palette_after_selection()

                else:

                    clear_hovered_color()

                    reset_palette_loading()

                # ============================================================
# PALM AUTO COLOR SELECTION
# ============================================================

                if selected is not None:

                    set_hovered_color(selected)

                    # Start / continue loading around this color.
                    selection_complete = update_palette_loading(
                        selected
                    )

                    # Automatically select after the short hold.
                    if (
                        selection_complete
                        and
                        not palette_selection_lock
                    ):

                        set_color(selected)

                        show_color_selection_message(
                            selected
                        )

                        palette_selection_lock = True

                        close_palette_after_selection()

                else:

                    clear_hovered_color()

                    # Palm moved away from all colors.
                    reset_palette_loading()

        elif open_palm:

            # Palette may be visible/closing;
            # open palm never draws.
            gesture_valid = False
            drawing = False
            reset_line()

    # Keep the color-selection lock while the same
    # open palm is still active.

        # =================================================
        # PHASE 4 — ACTUAL BRUSH-SIZE CURSOR
        # =================================================
        #
        # The circle radius is derived directly from the
        # current brush thickness. Therefore changing +/-
        # immediately changes the cursor size too.
        #
        # Hide it while:
        #   - OPEN PALM is selecting a color
        #   - TWO-FINGER eraser is active
        # =================================================

        if (
            not open_palm
            and
            not two_finger_active
            and
            gesture_valid
        ):
            cursor_radius = max(
                brush_thickness() // 2,
                2
            )

            cursor_color = brush_color()

            # Minimal cursor: one neutral outer ring and one
            # color ring matching the actual brush.
            cv2.circle(
                frame,
                (stroke_x, stroke_y),
                cursor_radius + 2,
                (235, 235, 235),
                1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                (stroke_x, stroke_y),
                cursor_radius,
                cursor_color,
                1,
                cv2.LINE_AA
            )

        # =================================================
        # PALETTE SELECTION MARKER
        # =================================================
        #
        # During open-palm selection, show the palm center
        # instead of the drawing cursor.
        # =================================================

        if open_palm and toolbar_visible:
            # ============================================================
            # PALM SELECTION CURSOR
            # ============================================================

            palm_cursor_radius = 14

            cv2.circle(
                frame,
                (palm_x, palm_y),
                palm_cursor_radius,
                (245, 245, 245),
                2,
                cv2.LINE_AA
            )

            # Small center point
            cv2.circle(
                frame,
                (palm_x, palm_y),
                3,
                (245, 245, 245),
                -1,
                cv2.LINE_AA
            )


        # ✌️ is the ONLY eraser control.
        # It does not change the selected drawing color.
        erase_gesture_valid = two_finger_active
        previous_two_finger_active = two_finger_active


        # =================================================
        # GESTURE STABILITY
        # =================================================

        if gesture_valid:

            valid_frames += 1

            invalid_frames = 0

        else:

            invalid_frames += 1

            valid_frames = 0


        # ✌️ ERASER: erase directly at the filtered fingertip.
        if erase_gesture_valid:
            # Erase only while ✌️ is active.
            erase_radius = max(BRUSH_SIZE * 3, 24)

            cv2.circle(
                canvas,
                (stroke_x, stroke_y),
                erase_radius,
                (0, 0, 0),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                canvas_mask,
                (stroke_x, stroke_y),
                erase_radius,
                0,
                -1,
                cv2.LINE_AA
            )

            current_stroke_points.append((stroke_x, stroke_y))
            previous_point = (stroke_x, stroke_y)
            drawing = False
            reset_line()

        # =================================================
        # START DRAWING
        # =================================================

        if (
            not drawing
            and
            valid_frames >=
            START_CONFIRMATION
        ):

            drawing = True


            # Save once for this whole stroke.
            save_canvas_state()

            # Round brush start.
            current_stroke_points = [(stroke_x, stroke_y)]
            draw_dot((stroke_x, stroke_y))

            # Start a completely new stroke.
            previous_point = (
                stroke_x,
                stroke_y
            )


        # =================================================
        # STOP DRAWING
        # =================================================

        if (
            drawing
            and
            invalid_frames >=
            STOP_CONFIRMATION
        ):

            drawing = False

            finish_smart_stroke()

            reset_line()


        # =================================================
        # DRAWING
        # =================================================

        if drawing:

            if previous_point is not None:

                previous_x, previous_y = (
                    previous_point
                )


                movement = point_distance_pixels(
                    (previous_x, previous_y),
                    (stroke_x, stroke_y)
                )

                # Ignore tiny movements caused by jitter.
                if movement >= STROKE_MIN_DISTANCE:

                    # Ignore only very large tracking jumps.
                    if movement <= MAX_JUMP:

                        draw_segment(
                            (previous_x, previous_y),
                            (stroke_x, stroke_y)
                        )

                        previous_point = (
                            stroke_x,
                            stroke_y
                        )

                        current_stroke_points.append(
                            (stroke_x, stroke_y)
                        )


        # =================================================
        # STATUS
        # =================================================

        if drawing:

            status_text = "DRAWING"

            status_color = (
                0,
                255,
                0
            )

        else:

            status_text = "PAUSED"

            status_color = (
                0,
                255,
                255
            )


    # ====================================================
    # NO HAND
    # ====================================================

    else:

        reset_gesture()

        status_text = "NO HAND"

        status_color = (
            0,
            0,
            255
        )


    # ====================================================
    # CAMERA + DRAWING
    # ====================================================

    # BLACK must be opaque, so do not use addWeighted().
    # Copy drawing pixels wherever the drawing mask exists.
    output = frame.copy()

    ink = canvas_mask > 0
    output[ink] = canvas[ink]



    # ====================================================
    # STATUS
    # ====================================================

    cv2.putText(
        output,
        status_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3,
        cv2.LINE_AA
    )


    cv2.putText(
        output,
        "INDEX = DRAW",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    if two_finger_active:
        mode_text = f"TWO-FINGER ERASER | Size: {BRUSH_SIZE}"
    else:
        mode_text = f"{current_color_name} | Size: {BRUSH_SIZE}"

    cv2.putText(
        output, mode_text, (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    # ====================================================
    # CLEAN STATUS INDICATOR
    # ====================================================
    #
    # One compact pill replaces the previous large HUD and
    # instruction text. It stays unobtrusive while still
    # showing the active color and brush size.
    # ====================================================

    status_x = 24
    status_y = 24
    status_w = 190
    status_h = 42

    status_overlay = output.copy()

    cv2.rectangle(
        status_overlay,
        (status_x, status_y),
        (status_x + status_w, status_y + status_h),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        status_overlay,
        0.82,
        output,
        0.18,
        0,
        output
    )

    cv2.rectangle(
        output,
        (status_x, status_y),
        (status_x + status_w, status_y + status_h),
        (90, 90, 90),
        1
    )

    status_dot = (
        status_x + 20,
        status_y + status_h // 2
    )

    status_radius = max(
        min(brush_thickness() // 2, 9),
        4
    )

    # White halo keeps BLACK and dark colors visible.
    cv2.circle(
        output,
        status_dot,
        status_radius + 2,
        (225, 225, 225),
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        output,
        status_dot,
        status_radius,
        brush_color(),
        -1,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"{current_color_name}  ·  {BRUSH_SIZE}px",
        (status_x + 39, status_y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA
    )

    # Recent colors: last three deliberately selected colors.
    recent_x = status_x + status_w + 10
    recent_y = status_y + status_h // 2

    for recent_name in recent_colors:
        recent_color = COLORS.get(
            recent_name,
            (255, 255, 255)
        )

        cv2.circle(
            output,
            (recent_x, recent_y),
            9,
            (210, 210, 210),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            output,
            (recent_x, recent_y),
            7,
            recent_color,
            -1,
            cv2.LINE_AA
        )

        recent_x += 22

    # ====================================================
    # HOVER COLOR PREVIEW
    # ====================================================

    if toolbar_visible and hovered_color_name is not None:
        preview_text = f"{hovered_color_name}  ·  HOLD TO SELECT"

        (pw, ph), _ = cv2.getTextSize(
            preview_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1
        )

        px = max(24, output.shape[1] - pw - 42)
        py = 24

        cv2.rectangle(
            output,
            (px, py),
            (px + pw + 18, py + ph + 14),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            preview_text,
            (px + 9, py + ph + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA
        )

    # ====================================================
    # MINIMAL COLOR SELECTION TOAST
    # ====================================================

    if color_message_timer > 0:
        notification_text = color_message_text

        (tw, th), baseline = cv2.getTextSize(
            notification_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        # Place the toast just below the status indicator.
        nx = 24
        ny = 82

        toast_w = tw + 28
        toast_h = th + 18

        toast_overlay = output.copy()

        cv2.rectangle(
            toast_overlay,
            (nx, ny),
            (nx + toast_w, ny + toast_h),
            (20, 20, 20),
            -1
        )

        cv2.addWeighted(
            toast_overlay,
            0.78,
            output,
            0.22,
            0,
            output
        )

        cv2.circle(
            output,
            (nx + 11, ny + toast_h // 2),
            4,
            brush_color(),
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            notification_text,
            (nx + 21, ny + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA
        )

        color_message_timer -= 1

    phase3_shape_status(output)

    return output


# ============================================================
# WEB SERVER ENTRY POINTS
# ============================================================
#
# process_frame() and handle_command() are called by
# web_server.py over the WebSocket connection. They are safe
# to import without a camera or display attached — no camera
# or window is created unless this file is run directly
# (python air_drawing.py).
# ============================================================

def process_frame(self,frame_bytes):
    """
    Takes one raw camera frame as JPEG-encoded bytes (as sent
    by the browser), runs it through hand tracking + drawing,
    and returns the composited output frame as JPEG bytes.
    Returns None if the frame couldn't be decoded.
    """

    h, w = frame.shape[:2]
    if self.canvas is None or self.canvas.shape[:2] != (h, w):
        # reinit canvas fresh at new size instead of cv2.resize-ing old canvas down
        new_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        new_canvas_mask = np.zeros((h, w), dtype=np.uint8)
        if self.canvas is not None:
            # optionally preserve old drawing by pasting top-left, not resizing
            oh, ow = self.canvas.shape[:2]
            ph, pw = min(oh, h), min(ow, w)
            new_canvas[:ph, :pw] = self.canvas[:ph, :pw]
            new_canvas_mask[:ph, :pw] = self.canvas_mask[:ph, :pw]
        self.canvas, self.canvas_mask = new_canvas, new_canvas_mask




    global canvas, canvas_mask

    nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    output = _process_one_frame(frame)

    ok, encoded = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if not ok:
        return None

    return encoded.tobytes()


def handle_command(command, payload=None):
    """
    Handles the JSON text commands sent by app.js over the
    WebSocket (undo, redo, clear, select_color, brush_size).
    Returns a dict that gets sent straight back to the browser
    as JSON so the UI can reflect the new state.
    """

    payload = payload or {}

    if command == "undo":
        undo()
        reset_line()
        return {"message": "UNDO"}

    if command == "redo":
        redo()
        reset_line()
        return {"message": "REDO"}

    if command == "clear":
        clear_canvas()
        reset_gesture()
        return {"message": "CANVAS CLEARED"}

    if command == "select_color":
        color = str(payload.get("color", "")).upper()
        if color in COLORS:
            set_color(color)
            return {
                "selectedColor": color,
                "message": f"{color} SELECTED"
            }
        return {"type": "error", "message": f"Unknown color: {color}"}

    if command == "brush_size":
        global BRUSH_SIZE
        try:
            size = int(payload.get("size", BRUSH_SIZE))
        except (TypeError, ValueError):
            return {"type": "error", "message": "Invalid brush size"}
        BRUSH_SIZE = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, size))
        return {"brushSize": BRUSH_SIZE}

    return {"type": "error", "message": f"Unknown command: {command}"}


# ============================================================
# DESKTOP APP (only runs with: python air_drawing.py)
# ============================================================

if __name__ == "__main__":

    try:

        while True:

            success, frame = camera.read()

            if not success:
                continue

            output = _process_one_frame(frame)

            # ====================================================
            # DISPLAY
            # ====================================================

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            # ====================================================
            # KEYBOARD
            # ====================================================

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("c"), ord("C")):
                clear_canvas()
                reset_gesture()

            elif key in (ord("z"), ord("Z")):
                undo()
                reset_line()

            elif key in (ord("x"), ord("X")):
                redo()
                reset_line()

            elif key in (ord("r"), ord("R")):
                set_color("RED")

            elif key in (ord("b"), ord("B")):
                set_color("BLUE")

            elif key in (ord("g"), ord("G")):
                set_color("GREEN")

            elif key in (ord("y"), ord("Y")):
                set_color("YELLOW")

            elif key in (ord("w"), ord("W")):
                set_color("WHITE")

            elif key in (ord("s"), ord("S")):
                SHAPE_RECOGNITION_ENABLED = not SHAPE_RECOGNITION_ENABLED
                print(
                    "Shape recognition:",
                    "ON" if SHAPE_RECOGNITION_ENABLED else "OFF"
                )

            elif key in (ord("+"), ord("=")):
                brush_up()

            elif key in (ord("-"), ord("_")):
                brush_down()

    # ============================================================
    # CLEANUP
    # ============================================================

    finally:

        camera.release()

        landmarker.close()

        cv2.destroyAllWindows()
