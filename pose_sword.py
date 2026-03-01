import argparse
import json
import math
import socket
import sys
import time
import cv2

import numpy as np

# ─── LED HSV ranges ───────────────────────────────────────────────────────────
YELLOW_LOWER = np.array([77, 180, 113])
YELLOW_UPPER = np.array([97, 255, 255])
ORANGE_LOWER = np.array([105, 178, 121])
ORANGE_UPPER = np.array([124, 255, 255])

last_green_pos    = None
last_red_pos      = None
last_sword_vector = None  # (dx, dy, length, angle, angle_from_vertical)

# COCO keypoint indices
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_HIP      = 11
IDX_R_HIP      = 12


def find_led_center(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [c for c in contours if 5 < cv2.contourArea(c) < 2000]
    if not valid:
        return None
    biggest = max(valid, key=cv2.contourArea)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def detect_leds(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    red_mask   = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    return find_led_center(green_mask), find_led_center(red_mask)


def draw_sword_overlay(frame):
    h, w = frame.shape[:2]
    if last_green_pos:
        cv2.circle(frame, last_green_pos, 10, (0, 255, 0), -1)
        cv2.putText(frame, "TIP", (last_green_pos[0]+12, last_green_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if last_red_pos:
        cv2.circle(frame, last_red_pos, 10, (0, 0, 255), -1)
        cv2.putText(frame, "HANDLE", (last_red_pos[0]+12, last_red_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if last_green_pos and last_red_pos and last_sword_vector:
        cv2.line(frame, last_red_pos, last_green_pos, (255, 255, 255), 2)
        dx, dy, length, angle, angle_from_vertical = last_sword_vector
        cv2.putText(frame, f"Sword: ({dx:.2f}, {dy:.2f}) len={length:.2f} angle={angle:.1f} vert={angle_from_vertical:.1f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

last_boxes     = None
last_scores    = None
last_keypoints = None
WINDOW_SIZE_H_W = (480, 640)


# ─── UDP ──────────────────────────────────────────────────────────────────────
def pixel_to_norm(px, py, w, h):
    return [round((px / w) * 2.0 - 1.0, 4),
            round((py / h) * 2.0 - 1.0, 4)]


def build_and_send_udp(keypoints_3d, w, h):
    def kp(idx):
        if keypoints_3d is None:
            return None
        k = keypoints_3d[idx]
        if k[2] < args.detection_threshold:
            return None
        return pixel_to_norm(float(k[0]), float(k[1]), w, h)

    sword_detected = last_sword_vector is not None
    if sword_detected:
        dx, dy, length, angle, angle_from_vertical = last_sword_vector
        tip    = pixel_to_norm(last_green_pos[0], last_green_pos[1], w, h) if last_green_pos else None
        handle = pixel_to_norm(last_red_pos[0],   last_red_pos[1],   w, h) if last_red_pos else None
        print(f"[SWORD] L={length:.4f}  angle_deg={angle:.2f}  vertical={angle_from_vertical:.2f}")
    else:
        dx, dy, length, angle, angle_from_vertical = 0.0, 0.0, 0.0, 0.0, 0.0
        tip = handle = None

    payload = {
        "shoulders": {
            "left":  kp(IDX_L_SHOULDER),
            "right": kp(IDX_R_SHOULDER)
        },
        "hips": {
            "left":  kp(IDX_L_HIP),
            "right": kp(IDX_R_HIP)
        },
        "sword": {
            "tip":               tip,
            "handle":            handle,
            "direction":         [round(dx, 4), round(dy, 4)],
            "length":            round(length, 4),
            "angle_deg":         round(angle, 2),
            "angle_from_vertical": round(angle_from_vertical, 2),
            "detected":          sword_detected
        },
        "t": round(time.time(), 4)
    }

    try:
        msg = json.dumps(payload).encode("utf-8")
        udp_sock.sendto(msg, (args.unity_ip, args.unity_port))
    except Exception as e:
        print(f"[UDP] Send error: {e}")


def ai_output_tensor_parse(metadata: dict):
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=args.detection_threshold,
            network_postprocess=True
        )
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes     = [np.array(b) for b in boxes]
            last_scores    = np.array(scores)
    return last_boxes, last_scores, last_keypoints


def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(m.array, boxes, scores,
                                  np.zeros(scores.shape), keypoints,
                                  args.detection_threshold, args.detection_threshold,
                                  request.get_metadata(), picam2, stream)


def picamera2_pre_callback(request: CompletedRequest):
    global last_green_pos, last_red_pos, last_sword_vector

    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())

    # ── LED detection on clean frame (before pose overlay) ──
    with MappedArray(request, 'main') as m:
        frame = m.array
        h, w  = frame.shape[:2]
        last_green_pos, last_red_pos = detect_leds(frame)

        if last_green_pos and last_red_pos:
            # Normalized coords
            tx = (last_green_pos[0] / w) * 2 - 1
            ty = (last_green_pos[1] / h) * 2 - 1
            hx = (last_red_pos[0]   / w) * 2 - 1
            hy = (last_red_pos[1]   / h) * 2 - 1

            # Raw vector (tip - handle) in normalized space
            dx_raw = tx - hx
            dy_raw = ty - hy
            length = math.sqrt(dx_raw**2 + dy_raw**2)

            # Normalized direction unit vector
            if length > 0:
                dx, dy = dx_raw / length, dy_raw / length
            else:
                dx, dy = 0.0, 0.0

            # Compass angle (0=right, 90=up, etc.)
            angle = math.degrees(math.atan2(-dy_raw, dx_raw))

            # Tilt from vertical: 0 = sword pointing straight up/down
            #                     90 = sword pointing left/right
            angle_from_vertical = round(math.degrees(math.atan2(abs(dx_raw), abs(dy_raw))), 2)

            last_sword_vector = (dx, dy, length, angle, angle_from_vertical)
        else:
            last_sword_vector = None

        draw_sword_overlay(frame)

    # ── Draw pose skeleton overlay after LED detection ──
    ai_output_tensor_draw(request, boxes, scores, keypoints)

    # ── Send UDP ──
    kp_3d = last_keypoints[0] if last_keypoints is not None and len(last_keypoints) > 0 else None
    build_and_send_udp(kp_3d, WINDOW_SIZE_H_W[1], WINDOW_SIZE_H_W[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--print-intrinsics", action="store_true")
    parser.add_argument("--unity-ip",   type=str, default="144.39.120.28")
    parser.add_argument("--unity-port", type=int, default=5005)
    return parser.parse_args()


def get_drawer():
    categories = [c for c in intrinsics.labels if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)


if __name__ == "__main__":
    args = get_args()

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[UDP] Sending to {args.unity_ip}:{args.unity_port}")

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        exit()

    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    drawer = get_drawer()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={'FrameRate': intrinsics.inference_rate},
        buffer_count=12
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping.")
    finally:
        udp_sock.close()