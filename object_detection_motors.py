from gpiozero import Motor
from time import sleep
motor1 = Motor(forward=5, backward=6)
motor2 = Motor(forward=13, backward=19)
motorSpeed = 0.5
#Test motor movement at beggining
motor1.forward(motorSpeed)
motor2.forward(motorSpeed)
sleep(2)
motor1.backward(motorSpeed)
motor2.backward(motorSpeed)
sleep(2)
motor1.stop()
motor2.stop()
sleep(2)

import argparse
import sys
from functools import lru_cache

import cv2

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []

# ──────────────────────────────────────────────
# MOTOR CONTROL CONFIG
# ──────────────────────────────────────────────
# Dead zone: how far (as a fraction of frame size) the person's center
# can be from the frame center before we issue a move command.
# 0.10 = 10% of frame width/height on each side → 20% total "do nothing" band.
HORIZONTAL_DEAD_ZONE = 0.10   # fraction of frame width
VERTICAL_DEAD_ZONE   = 0.10   # fraction of frame height

# When multiple people are detected, track the one with the highest confidence.
TRACK_HIGHEST_CONFIDENCE = True


# ──────────────────────────────────────────────
# STUB MOTOR FUNCTIONS — replace with your GPIO code
# ──────────────────────────────────────────────
def motor_left(magnitude: float):
    """
    Called when the person is to the LEFT of center.
    magnitude: 0.0–1.0  (how far off-center, 1.0 = at the edge of the frame)
    """
    print(f"[MOTOR] PAN LEFT  | magnitude={magnitude:.2f}")
    motor2.forward(motorSpeed)


def motor_right(magnitude: float):
    """
    Called when the person is to the RIGHT of center.
    magnitude: 0.0–1.0
    """
    print(f"[MOTOR] PAN RIGHT | magnitude={magnitude:.2f}")
    motor2.backward(motorSpeed)


def motor_up(magnitude: float):
    """
    Called when the person is ABOVE center (center_y < frame mid).
    magnitude: 0.0–1.0
    """
    print(f"[MOTOR] TILT UP   | magnitude={magnitude:.2f}")


def motor_down(magnitude: float):
    """
    Called when the person is BELOW center (center_y > frame mid).
    magnitude: 0.0–1.0
    """
    print(f"[MOTOR] TILT DOWN | magnitude={magnitude:.2f}")


def motor_centered():
    """Called when the person is inside the dead zone — no movement needed."""
    print("[MOTOR] CENTERED  | no movement")
    motor1.stop()
    motor2.stop()


# ──────────────────────────────────────────────
# PERSON TRACKING LOGIC
# ──────────────────────────────────────────────
def get_frame_dimensions():
    """Return (width, height) of the main stream output."""
    cfg = picam2.camera_configuration()
    main = cfg.get("main", {})
    size = main.get("size", (640, 480))   # sensible fallback
    return size  # (width, height)


def find_target_person(detections):
    """
    From a list of Detection objects, return the single person Detection
    we want to track, or None if no person is visible.

    Strategy: pick the highest-confidence person detection.
    You could also pick the largest bounding box, or the one closest
    to the current center — swap out this function to change strategy.
    """
    labels = get_labels()
    person_label = "person"

    people = [
        d for d in detections
        if labels[int(d.category)].lower() == person_label
    ]

    if not people:
        return None

    if TRACK_HIGHEST_CONFIDENCE:
        return max(people, key=lambda d: d.conf)
    else:
        # Largest bounding box by area
        return max(people, key=lambda d: d.box[2] * d.box[3])


def get_person_center(detection):
    """
    Given a Detection, return (center_x, center_y) in pixels.
    detection.box is (x, y, w, h) — top-left corner + width/height.
    """
    x, y, w, h = detection.box
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y


def decide_motor_direction(center_x, center_y, frame_w, frame_h):
    """
    Compare the person's center to the frame center and call the
    appropriate motor function(s).

    Returns a dict with the computed info (handy for debug overlay).
    """
    mid_x = frame_w / 2
    mid_y = frame_h / 2

    # Offset from center, normalised to -1.0 … +1.0
    # Positive offset_x → person is to the RIGHT
    # Positive offset_y → person is BELOW center (y increases downward)
    offset_x = (center_x - mid_x) / mid_x
    offset_y = (center_y - mid_y) / mid_y

    h_action = "center"
    v_action = "center"

    # Horizontal axis
    if offset_x < -HORIZONTAL_DEAD_ZONE:
        magnitude = (abs(offset_x) - HORIZONTAL_DEAD_ZONE) / (1.0 - HORIZONTAL_DEAD_ZONE)
        motor_left(magnitude)
        h_action = f"LEFT  {magnitude:.2f}"
    elif offset_x > HORIZONTAL_DEAD_ZONE:
        magnitude = (offset_x - HORIZONTAL_DEAD_ZONE) / (1.0 - HORIZONTAL_DEAD_ZONE)
        motor_right(magnitude)
        h_action = f"RIGHT {magnitude:.2f}"

    # Vertical axis
    if offset_y < -VERTICAL_DEAD_ZONE:
        magnitude = (abs(offset_y) - VERTICAL_DEAD_ZONE) / (1.0 - VERTICAL_DEAD_ZONE)
        motor_up(magnitude)
        v_action = f"UP    {magnitude:.2f}"
    elif offset_y > VERTICAL_DEAD_ZONE:
        magnitude = (offset_y - VERTICAL_DEAD_ZONE) / (1.0 - VERTICAL_DEAD_ZONE)
        motor_down(magnitude)
        v_action = f"DOWN  {magnitude:.2f}"

    if h_action == "center" and v_action == "center":
        motor_centered()

    return {
        "center_x": center_x,
        "center_y": center_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "h_action": h_action,
        "v_action": v_action,
    }


# ──────────────────────────────────────────────
# ORIGINAL DETECTION CODE (unchanged logic)
# ──────────────────────────────────────────────
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()

    frame_w, frame_h = get_frame_dimensions()
    mid_x = frame_w // 2
    mid_y = frame_h // 2

    # Find the person we're tracking this frame
    target = find_target_person(detections)

    with MappedArray(request, stream) as m:

        # ── Draw frame-center crosshair ──────────────────────
        crosshair_size = 20
        crosshair_color = (0, 255, 255)   # yellow
        cv2.line(m.array,
                 (mid_x - crosshair_size, mid_y),
                 (mid_x + crosshair_size, mid_y),
                 crosshair_color, 1)
        cv2.line(m.array,
                 (mid_x, mid_y - crosshair_size),
                 (mid_x, mid_y + crosshair_size),
                 crosshair_color, 1)

        # ── Draw dead-zone rectangle ─────────────────────────
        dz_x = int(mid_x - HORIZONTAL_DEAD_ZONE * mid_x)
        dz_y = int(mid_y - VERTICAL_DEAD_ZONE   * mid_y)
        dz_x2 = int(mid_x + HORIZONTAL_DEAD_ZONE * mid_x)
        dz_y2 = int(mid_y + VERTICAL_DEAD_ZONE   * mid_y)
        cv2.rectangle(m.array, (dz_x, dz_y), (dz_x2, dz_y2), (0, 255, 255), 1)

        # ── Draw all detections ──────────────────────────────
        for detection in detections:
            x, y, w, h = detection.box
            is_target = (detection is target)

            box_color  = (0, 255, 0) if not is_target else (0, 128, 255)  # green / orange
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            overlay = m.array.copy()
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)

            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, thickness=2)

            # ── If this is the tracked person, draw center dot + info ──
            if is_target:
                cx, cy = get_person_center(detection)
                cx, cy = int(cx), int(cy)

                # Dot at person's center
                cv2.circle(m.array, (cx, cy), 6, (0, 128, 255), -1)

                # Line from frame center to person center
                cv2.line(m.array, (mid_x, mid_y), (cx, cy), (0, 128, 255), 1)

                # Offset text
                info = f"cx={cx} cy={cy}"
                cv2.putText(m.array, info, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    frame_w, frame_h = get_frame_dimensions()

    while True:
        last_results = parse_detections(picam2.capture_metadata())

        # ── Motor control: runs every frame in the main loop ──────────────
        target = find_target_person(last_results)
        if target is not None:
            cx, cy = get_person_center(target)
            info = decide_motor_direction(cx, cy, frame_w, frame_h)
            # info dict available here if you want to log or throttle commands:
            # info = { center_x, center_y, offset_x, offset_y, h_action, v_action }
        else:
            print("[MOTOR] No person detected")