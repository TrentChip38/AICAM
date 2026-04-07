#!/usr/bin/env python3
"""
=============================================================
  TURRET CONTROLLER
  Flask + MJPEG + IMX500 AI Camera — single file
  - Streams IMX500 camera feed with detection overlays
  - Manual mode: phone buttons drive motors directly
  - AI Turret mode: tracks nearest person automatically
=============================================================

SETUP:
  pip install flask opencv-python gpiozero picamera2 --break-system-packages

RUN:
  python3 turret_server.py

Then on your phone, connect to the same WiFi and go to:
  http://<your-pi-ip>:5000   (find IP with: hostname -I)
"""

import sys
import threading
import time
import argparse
from functools import lru_cache

import board
import busio
import adafruit_lis3dh
import math

import cv2
from flask import Flask, Response, request, jsonify
from gpiozero import Motor
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# =============================================================
#  CONFIGURATION
# =============================================================

STREAM_FPS           = 24
JPEG_QUALITY         = 70
HOST                 = "0.0.0.0"
PORT                 = 5000
MOTOR_SPEED          = 0.5

HORIZONTAL_DEAD_ZONE = 0.10   # fraction of frame width before motors activate
VERTICAL_DEAD_ZONE   = 0.10   # fraction of frame height before motors activate

TRACK_HIGHEST_CONFIDENCE = True

# I2C setup for LIS3DH at address 0x18
i2c = busio.I2C(board.SCL, board.SDA)
lis3dh = adafruit_lis3dh.LIS3DH_I2C(i2c, address=0x18)

def get_accel_direction():
    x, y, z = lis3dh.acceleration  # in m/s^2
    # Simple direction logic (customize as needed)
    if abs(x) > abs(y):
        if x > 2: return "LEFT"
        elif x < -2: return "RIGHT"
    else:
        if y > 2: return "FORWARD"
        elif y < -2: return "BACKWARD"
    return "FLAT"
import math

def get_accel_angles():
    x, y, z = lis3dh.acceleration
    # Pitch: rotation around X axis (forward/backward tilt)
    # Roll:  rotation around Y axis (side-to-side tilt)
    pitch = math.degrees(math.atan2(-x, math.sqrt(y*y + z*z)))
    roll  = math.degrees(math.atan2(y, z))
    return pitch, roll
    
def get_accel_direction():
    pitch, roll = get_accel_angles()
    if abs(pitch) > abs(roll):
        if pitch > 10: return "UP"
        elif pitch < -10: return "DOWN"
    else:
        if roll > 10: return "LEFT"
        elif roll < -10: return "RIGHT"
    return "FLAT"
# =============================================================
#  MOTORS  (gpiozero)
# =============================================================

motor1 = Motor(forward=6,  backward=5)   # tilt / vertical
motor2 = Motor(forward=19, backward=13)  # pan  / horizontal
shooter = Motor(forward=26, backward=20)  # Actually the gun

def stop_all_motors():
    motor1.stop()
    motor2.stop()

def motor_left(magnitude: float):
    print(f"[MOTOR] PAN LEFT  | magnitude={magnitude:.2f}")
    motor2.forward(MOTOR_SPEED)

def motor_right(magnitude: float):
    print(f"[MOTOR] PAN RIGHT | magnitude={magnitude:.2f}")
    motor2.backward(MOTOR_SPEED)

def motor_up(magnitude: float):
    print(f"[MOTOR] TILT UP   | magnitude={magnitude:.2f}")
    motor1.backward(MOTOR_SPEED)

def motor_down(magnitude: float):
    print(f"[MOTOR] TILT DOWN | magnitude={magnitude:.2f}")
    motor1.forward(MOTOR_SPEED)

last_time = 0
def motor_centered():
    global last_time
    print("[MOTOR] CENTERED — stopping")
    stop_all_motors()
    current_time = time.time()
    if current_time - last_time >= 1.0:#Wait this many seconds before firing again
        gun_fire()
        last_time = current_time

def gun_on():
    print("[shooter] firing")
    shooter.forward(0.5)
    
def gun_stop():
    print("[shooter] stopping")
    shooter.stop()
    
def gun_fire(duration=0.8):
    def fire():
        print("Shooting a few times")
        shooter.forward(1)
        time.sleep(duration)
        shooter.stop()
    #Makes a thread that calls fire with duration
    threading.Thread(target=fire, daemon=True).start()

# =============================================================
#  GLOBAL STATE
# =============================================================

ai_mode_enabled = False   # toggled by the AI Turret button on the UI
last_detections = []
last_results    = None
picam2          = None
imx500          = None
intrinsics      = None
args            = None

# Prevent AI loop and button handler from racing on motor calls
motor_lock = threading.Lock()

# =============================================================
#  PERSON TRACKING LOGIC
# =============================================================

def get_frame_dimensions():
    cfg  = picam2.camera_configuration()
    main = cfg.get("main", {})
    return main.get("size", (640, 480))


def find_target_person(detections):
    labels = get_labels()
    people = [d for d in detections if labels[int(d.category)].lower() == "person"]
    if not people:
        return None
    if TRACK_HIGHEST_CONFIDENCE:
        return max(people, key=lambda d: d.conf)
    return max(people, key=lambda d: d.box[2] * d.box[3])


def get_person_center(detection):
    x, y, w, h = detection.box
    return x + w / 2, y + h / 2


def decide_motor_direction(center_x, center_y, frame_w, frame_h):
    mid_x    = frame_w / 2
    mid_y    = frame_h / 2
    offset_x = (center_x - mid_x) / mid_x
    offset_y = (center_y - mid_y) / mid_y

    h_action = "center"
    v_action = "center"

    if offset_x < -HORIZONTAL_DEAD_ZONE:
        mag = (abs(offset_x) - HORIZONTAL_DEAD_ZONE) / (1.0 - HORIZONTAL_DEAD_ZONE)
        motor_left(mag);  h_action = f"LEFT  {mag:.2f}"
    elif offset_x > HORIZONTAL_DEAD_ZONE:
        mag = (offset_x - HORIZONTAL_DEAD_ZONE) / (1.0 - HORIZONTAL_DEAD_ZONE)
        motor_right(mag); h_action = f"RIGHT {mag:.2f}"

    if offset_y < -VERTICAL_DEAD_ZONE:
        mag = (abs(offset_y) - VERTICAL_DEAD_ZONE) / (1.0 - VERTICAL_DEAD_ZONE)
        motor_up(mag);    v_action = f"UP    {mag:.2f}"
    elif offset_y > VERTICAL_DEAD_ZONE:
        mag = (offset_y - VERTICAL_DEAD_ZONE) / (1.0 - VERTICAL_DEAD_ZONE)
        motor_down(mag);  v_action = f"DOWN  {mag:.2f}"

    if h_action == "center" and v_action == "center":
        motor_centered()

    return {"offset_x": offset_x, "offset_y": offset_y,
            "h_action": h_action,  "v_action": v_action}

# =============================================================
#  IMX500 DETECTION
# =============================================================

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf     = conf
        self.box      = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    global last_detections
    np_outputs       = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=args.threshold,
            iou_thres=args.iou, max_out_dets=args.max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > args.threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [l for l in labels if l and l != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Picamera2 pre-callback: draws overlays onto every frame before capture."""
    detections = last_results
    if not detections:
        return

    labels             = get_labels()
    frame_w, frame_h   = get_frame_dimensions()
    mid_x, mid_y       = frame_w // 2, frame_h // 2
    target             = find_target_person(detections)

    with MappedArray(request, stream) as m:

        # Frame-center crosshair
        cv2.line(m.array, (mid_x - 20, mid_y), (mid_x + 20, mid_y), (0, 255, 255), 1)
        cv2.line(m.array, (mid_x, mid_y - 20), (mid_x, mid_y + 20), (0, 255, 255), 1)

        # Dead-zone rectangle
        dz_x  = int(mid_x - HORIZONTAL_DEAD_ZONE * mid_x)
        dz_y  = int(mid_y - VERTICAL_DEAD_ZONE   * mid_y)
        dz_x2 = int(mid_x + HORIZONTAL_DEAD_ZONE * mid_x)
        dz_y2 = int(mid_y + VERTICAL_DEAD_ZONE   * mid_y)
        cv2.rectangle(m.array, (dz_x, dz_y), (dz_x2, dz_y2), (0, 255, 255), 1)

        # AI mode indicator burned into the frame
        if ai_mode_enabled:
            cv2.putText(m.array, "AI TURRET ON", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 64, 255), 2)

        for detection in detections:
            x, y, w, h = detection.box
            is_target  = (detection is target)
            box_color  = (0, 128, 255) if is_target else (0, 255, 0)
            label      = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            overlay = m.array.copy()
            cv2.rectangle(overlay,
                          (x + 5, y + 15 - th), (x + 5 + tw, y + 15 + bl),
                          (255, 255, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.30, m.array, 0.70, 0, m.array)
            cv2.putText(m.array, label, (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, 2)

            if is_target:
                cx, cy = int(x + w / 2), int(y + h / 2)
                cv2.circle(m.array, (cx, cy), 6, (0, 128, 255), -1)
                cv2.line(m.array, (mid_x, mid_y), (cx, cy), (0, 128, 255), 1)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

# =============================================================
#  FRAME GRABBER  — feeds annotated frames to MJPEG stream
# =============================================================

class FrameGrabber:
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def get_jpeg(self):
        with self._lock:
            if self._frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self._frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return buf.tobytes() if ok else None

grabber = FrameGrabber()

# =============================================================
#  FLASK APP
# =============================================================

app = Flask(__name__)

def generate_frames():
    while True:
        jpeg = grabber.get_jpeg()
        if jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(1.0 / STREAM_FPS)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/direction")
def direction():
    dir = get_accel_direction()
    pitch, roll = get_accel_angles()
    return jsonify({
        "direction": dir,
        "pitch": round(pitch, 1),
        "roll": round(roll, 1)
    })

@app.route("/command", methods=["POST"])
def command():
    global ai_mode_enabled
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "")
    event  = data.get("event", "press")   # "press" or "release"

    # ── AI mode toggle ───────────────────────────────────────
    if action == "ai_toggle":
        ai_mode_enabled = not ai_mode_enabled
        stop_all_motors()
        print(f"[AI]    AI Turret mode {'ON' if ai_mode_enabled else 'OFF'}")
        return jsonify({"status": "ok", "ai_mode": ai_mode_enabled})

    # ── Manual buttons (ignored while AI mode is active) ─────
    if ai_mode_enabled:
        return jsonify({"status": "ignored", "reason": "ai_mode_active"})

    with motor_lock:
        if event == "release":
            stop_all_motors()
            print(f"[CMD]   {action} RELEASED → motors stopped")
            return jsonify({"status": "ok", "action": action, "event": "release"})

        action_map = {
            "pan_left":  lambda: motor_left(1.0),
            "pan_right": lambda: motor_right(1.0),
            "tilt_up":   lambda: motor_up(1.0),
            "tilt_down": lambda: motor_down(1.0),
            "fire":      lambda: gun_fire(),  # wire up fire mechanism here
        }

        if action in action_map:
            print(f"[CMD]   {action} PRESSED")
            action_map[action]()
            return jsonify({"status": "ok", "action": action})

    return jsonify({"status": "error", "message": "unknown action"}), 400


@app.route("/status")
def status():
    return jsonify({"ai_mode": ai_mode_enabled})

#What showed the angles and direction up top
# </head>
# <body>
# <div id="angles" style="margin:10px 0;font-size:1.2em;color:#ff9500;">...</div>
# <div id="direction" style="margin:10px 0;font-size:1.2em;color:#00e5ff;">...</div>

@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>
  <title>Turret Control</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#0a0c0f;--panel:#10141a;--border:#1e2a38;
      --accent:#00e5ff;--red:#ff2d55;--green:#39ff14;--orange:#ff9500;
      --text:#c8d8e8;
      --glow:0 0 12px rgba(0,229,255,0.4);
      --red-glow:0 0 16px rgba(255,45,85,0.6);
      --orange-glow:0 0 16px rgba(255,149,0,0.7);
    }
    html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;overflow:hidden}
    .app{display:flex;flex-direction:column;height:100dvh;padding:10px;gap:10px}

    header{display:flex;align-items:center;justify-content:space-between;
      padding:6px 12px;border:1px solid var(--border);background:var(--panel);border-radius:6px}
    header h1{font-size:1.1rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
      color:var(--accent);text-shadow:var(--glow)}
    .status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
      box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

    .video-wrapper{flex:1;border:1px solid var(--border);border-radius:6px;overflow:hidden;
      background:#000;position:relative;min-height:0}
    .video-wrapper img{width:100%;height:100%;object-fit:cover;display:block}
    .corner{position:absolute;width:14px;height:14px;border-color:var(--accent);border-style:solid;opacity:.5}
    .corner.tl{top:8px;left:8px;border-width:2px 0 0 2px}
    .corner.tr{top:8px;right:8px;border-width:2px 2px 0 0}
    .corner.bl{bottom:8px;left:8px;border-width:0 0 2px 2px}
    .corner.br{bottom:8px;right:8px;border-width:0 2px 2px 0}
    .crosshair{position:absolute;inset:0;pointer-events:none;display:flex;align-items:center;justify-content:center}
    .crosshair::before,.crosshair::after{content:'';position:absolute;background:rgba(0,229,255,.5)}
    .crosshair::before{width:1px;height:30px}
    .crosshair::after{width:30px;height:1px}
    .crosshair-ring{width:40px;height:40px;border:1px solid rgba(0,229,255,.4);border-radius:50%}
    .ai-badge{position:absolute;top:8px;left:50%;transform:translateX(-50%);
      background:rgba(255,149,0,.15);border:1px solid var(--orange);color:var(--orange);
      font-family:'Share Tech Mono',monospace;font-size:.7rem;padding:3px 10px;
      border-radius:4px;letter-spacing:.1em;opacity:0;transition:opacity .3s;pointer-events:none}
    .ai-badge.visible{opacity:1}

    .controls{display:flex;flex-direction:column;gap:8px;padding:8px;
      border:1px solid var(--border);background:var(--panel);border-radius:6px}
    .dpad{display:grid;grid-template-columns:1fr auto 1fr;grid-template-rows:auto auto auto;
      gap:8px;align-items:center;justify-items:center}

    .btn{
  -webkit-user-select: none; /* Safari */
  -moz-user-select: none;    /* Firefox */
  -ms-user-select: none;     /* IE10+/Edge */
  user-select: none;         /* Standard */
  -webkit-tap-highlight-color:transparent;user-select:none;cursor:pointer;
      border:1px solid var(--border);border-radius:6px;background:#0d1117;color:var(--text);
      font-family:'Rajdhani',sans-serif;font-size:.75rem;font-weight:600;
      letter-spacing:.08em;text-transform:uppercase;padding:10px 14px;min-width:64px;
      text-align:center;transition:background .1s,border-color .1s,box-shadow .1s;touch-action:manipulation}
    .btn:active,.btn.active{background:rgba(0,229,255,.08);border-color:var(--accent);
      color:var(--accent);box-shadow:var(--glow)}
    .btn-up{grid-column:2;grid-row:1}
    .btn-left{grid-column:1;grid-row:2}
    .btn-right{grid-column:3;grid-row:2}
    .btn-down{grid-column:2;grid-row:3}

    .bottom-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .btn-fire{padding:14px;border-radius:8px;border:1px solid rgba(255,45,85,.4);
      background:rgba(255,45,85,.08);color:var(--red);font-size:1rem;letter-spacing:.2em}
    .btn-fire:active,.btn-fire.active{background:rgba(255,45,85,.2);box-shadow:var(--red-glow)}
    .btn-ai{padding:14px;border-radius:8px;border:1px solid rgba(255,149,0,.3);
      background:rgba(255,149,0,.06);color:var(--orange);font-size:.85rem;letter-spacing:.15em;
      transition:background .2s,box-shadow .2s,border-color .2s}
    .btn-ai.ai-on{background:rgba(255,149,0,.2);border-color:var(--orange);
      box-shadow:var(--orange-glow);animation:ai-pulse 1.5s infinite}
    @keyframes ai-pulse{0%,100%{box-shadow:var(--orange-glow)}50%{box-shadow:none}}

    #toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(20px);
      background:rgba(0,229,255,.1);border:1px solid var(--accent);color:var(--accent);
      font-family:'Share Tech Mono',monospace;font-size:.75rem;padding:6px 16px;
      border-radius:4px;opacity:0;transition:opacity .2s,transform .2s;pointer-events:none;z-index:99}
    #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  </style>
</head>
<body>
<div class="app">
  <header>
    <h1>&#9678; Turret Control</h1>
    <div class="status-dot"></div>
  </header>

  <div class="video-wrapper">
    <img src="/video_feed" alt="Camera feed"/>
    <div class="corner tl"></div><div class="corner tr"></div>
    <div class="corner bl"></div><div class="corner br"></div>
    <div class="crosshair"><div class="crosshair-ring"></div></div>
    <div class="ai-badge" id="aiBadge">◆ AI TRACKING</div>
  </div>

  <div class="controls">
    <div class="dpad">
      <button class="btn btn-up"    data-action="tilt_up">   &#9650; UP    </button>
      <button class="btn btn-left"  data-action="pan_left">  &#9664; LEFT  </button>
      <button class="btn btn-right" data-action="pan_right"> RIGHT &#9654; </button>
      <button class="btn btn-down"  data-action="tilt_down"> &#9660; DOWN  </button>
    </div>
    <div class="bottom-row">
      <button class="btn btn-fire" data-action="fire">  &#9670; FIRE      </button>
      <button class="btn btn-ai"   id="aiBtn">          &#11041; AI TURRET </button>
    </div>
  </div>
</div>
<div id="toast"></div>

<script>
  let aiMode = false;
  const aiBtn   = document.getElementById('aiBtn');
  const aiBadge = document.getElementById('aiBadge');
  const toast   = document.getElementById('toast');
  let toastTimer;

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1400);
  }

  function setAiMode(on) {
    aiMode = on;
    aiBtn.classList.toggle('ai-on', on);
    aiBadge.classList.toggle('visible', on);
    showToast(on ? 'AI TURRET ON' : 'AI TURRET OFF');
  }

  async function updateDirection() {
    try {
        const res = await fetch('/direction');
        const data = await res.json();
        document.getElementById('direction').textContent = 'Direction: ' + data.direction;
        document.getElementById('angles').textContent = 'Pitch: ' + data.pitch + '°  Roll: ' + data.roll + '°';
    } catch (e) {
        document.getElementById('direction').textContent = 'Direction: (error)';
        document.getElementById('angles').textContent = 'Pitch: ... Roll: ...';
    }
    }
 setInterval(updateDirection, 500); // update every 0.5s
 updateDirection();

  async function sendCommand(action, event = 'press') {
    try {
      const res = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, event })
      });
      return await res.json();
    } catch (err) { showToast('No connection'); }
  }

  // AI toggle button
  aiBtn.addEventListener('pointerdown', async e => {
    e.preventDefault();
    const data = await sendCommand('ai_toggle');
    if (data) setAiMode(data.ai_mode);
  });

  // D-pad + fire: hold = motors on, release = motors stop
  document.querySelectorAll('.btn[data-action]:not(#aiBtn)').forEach(btn => {
    const action = btn.dataset.action;

    btn.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (aiMode) { showToast('AI mode active'); return; }
      btn.classList.add('active');
      sendCommand(action, 'press');
    });

    const release = () => {
      if (!btn.classList.contains('active')) return;
      btn.classList.remove('active');
      sendCommand(action, 'release');
    };

    btn.addEventListener('pointerup',     release);
    btn.addEventListener('pointerleave',  release);
    btn.addEventListener('pointercancel', release);
  });

  // Keyboard support for desktop testing
  const keyMap = {
    ArrowUp:'tilt_up', ArrowDown:'tilt_down',
    ArrowLeft:'pan_left', ArrowRight:'pan_right', ' ':'fire'
  };
  const heldKeys = new Set();
  document.addEventListener('keydown', e => {
    if (!keyMap[e.key] || heldKeys.has(e.key)) return;
    e.preventDefault(); heldKeys.add(e.key);
    sendCommand(keyMap[e.key], 'press');
  });
  document.addEventListener('keyup', e => {
    if (!keyMap[e.key]) return;
    heldKeys.delete(e.key);
    sendCommand(keyMap[e.key], 'release');
  });

  // Sync AI state on load
  fetch('/status').then(r => r.json()).then(d => setAiMode(d.ai_mode));
</script>
</body>
</html>"""

# =============================================================
#  ARGS
# =============================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps",                type=int)
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bbox-order",         choices=["yx", "xy"], default="yx")
    parser.add_argument("--threshold",          type=float, default=0.55)
    parser.add_argument("--iou",                type=float, default=0.65)
    parser.add_argument("--max-detections",     type=int,   default=10)
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction)
    parser.add_argument("--postprocess",        choices=["", "nanodet"], default=None)
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction)
    parser.add_argument("--labels",             type=str)
    parser.add_argument("--print-intrinsics",   action="store_true")
    return parser.parse_args()

# =============================================================
#  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    args = get_args()

    # ── IMX500 / Picamera2 init ──────────────────────────────
    imx500     = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()

    if not imx500.network_intrinsics:
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value) as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open("assets/coco_labels.txt") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics); sys.exit(0)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.pre_callback = draw_detections
    picam2.start(config, show_preview=False)   # no desktop preview needed

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    frame_w, frame_h = get_frame_dimensions()

    # ── Flask server in a background daemon thread ───────────
    flask_thread = threading.Thread(
        target=lambda: app.run(host=HOST, port=PORT, threaded=True),
        daemon=True
    )
    flask_thread.start()

    print("=" * 50)
    print("  NERF TURRET SERVER STARTED")
    print(f"  Open on your phone: http://<pi-ip>:{PORT}")
    print("  Find your IP with:  hostname -I")
    print("  Ctrl+C to stop")
    print("=" * 50)

    # ── Main loop: detections + MJPEG frame updates ──────────
    while True:
        last_results = parse_detections(picam2.capture_metadata())

        # Push latest annotated frame to the MJPEG grabber
        frame = picam2.capture_array("main")
        grabber.update(frame)

        # AI tracking — only active when ai_mode_enabled is True
        if ai_mode_enabled:
            with motor_lock:
                target = find_target_person(last_results)
                if target is not None:
                    cx, cy = get_person_center(target)
                    decide_motor_direction(cx, cy, frame_w, frame_h)
                else:
                    print("[AI]    No person detected — stopping motors")
                    stop_all_motors()

