#!/usr/bin/env python3
"""
=============================================================
  NERF TURRET CONTROLLER
  Flask + MJPEG single-file server
  - Streams camera feed to phone browser
  - Receives button commands to control servos / fire
=============================================================

SETUP:
  pip install flask opencv-python RPi.GPIO

RUN:
  python3 turret_server.py

Then on your phone, connect to the same WiFi and go to:
  http://<your-pi-ip>:5000

Find your Pi's IP with: hostname -I
"""

import cv2
import threading
import time
from flask import Flask, Response, request, jsonify

# ── Optional: uncomment if using GPIO for servos / firing ──
# import RPi.GPIO as GPIO

# =============================================================
#  CONFIGURATION — edit these
# =============================================================

CAMERA_INDEX   = 0       # 0 = default camera / Pi cam via v4l2
STREAM_WIDTH   = 640
STREAM_HEIGHT  = 480
STREAM_FPS     = 24
JPEG_QUALITY   = 70      # 1–100, lower = faster stream
HOST           = "0.0.0.0"
PORT           = 5000

# GPIO pin numbers (BCM mode) — set to None to disable
PIN_PAN_LEFT   = 17
PIN_PAN_RIGHT  = 27
PIN_TILT_UP    = 22
PIN_TILT_DOWN  = 23
PIN_FIRE       = 24

# =============================================================
#  GPIO SETUP (comment out this whole section if not using GPIO)
# =============================================================

def gpio_setup():
    pass  # Replace with real GPIO init when ready
    # GPIO.setmode(GPIO.BCM)
    # for pin in [PIN_PAN_LEFT, PIN_PAN_RIGHT, PIN_TILT_UP, PIN_TILT_DOWN, PIN_FIRE]:
    #     if pin is not None:
    #         GPIO.setup(pin, GPIO.OUT)
    #         GPIO.output(pin, GPIO.LOW)

def gpio_pulse(pin, duration=0.2):
    """Briefly activate a GPIO pin."""
    pass  # Replace with real GPIO pulse when ready
    # if pin is None:
    #     return
    # GPIO.output(pin, GPIO.HIGH)
    # time.sleep(duration)
    # GPIO.output(pin, GPIO.LOW)

# =============================================================
#  CAMERA — grabs frames in a background thread
# =============================================================

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

        self.frame = None
        self.lock  = threading.Lock()
        self.running = True

        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ok, buf = cv2.imencode(
                ".jpg", self.frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            return buf.tobytes() if ok else None

    def release(self):
        self.running = False
        self.cap.release()


# =============================================================
#  FLASK APP
# =============================================================

app = Flask(__name__)
camera = Camera()
gpio_setup()

# ── MJPEG stream generator ──────────────────────────────────

def generate_frames():
    while True:
        jpeg = camera.get_jpeg()
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg +
                b"\r\n"
            )
        time.sleep(1.0 / STREAM_FPS)

# ── Routes ───────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/command", methods=["POST"])
def command():
    """
    Receives a JSON command from the phone UI.
    Expected body: { "action": "pan_left" }
    """
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "")

    actions = {
        "pan_left":  lambda: gpio_pulse(PIN_PAN_LEFT),
        "pan_right": lambda: gpio_pulse(PIN_PAN_RIGHT),
        "tilt_up":   lambda: gpio_pulse(PIN_TILT_UP),
        "tilt_down": lambda: gpio_pulse(PIN_TILT_DOWN),
        "fire":      lambda: gpio_pulse(PIN_FIRE, duration=0.5),
    }

    if action in actions:
        print(f"[CMD] {action}")
        threading.Thread(target=actions[action], daemon=True).start()
        return jsonify({"status": "ok", "action": action})

    print(f"[WARN] Unknown action: '{action}'")
    return jsonify({"status": "error", "message": "unknown action"}), 400


@app.route("/")
def index():
    """Serves the phone control UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
  <title>Turret Control</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0a0c0f;
      --panel:     #10141a;
      --border:    #1e2a38;
      --accent:    #00e5ff;
      --red:       #ff2d55;
      --green:     #39ff14;
      --text:      #c8d8e8;
      --dim:       #4a5a6a;
      --glow:      0 0 12px rgba(0,229,255,0.4);
      --red-glow:  0 0 16px rgba(255,45,85,0.6);
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: 'Rajdhani', sans-serif;
      overflow: hidden;
    }

    /* ── Layout ── */
    .app {
      display: flex;
      flex-direction: column;
      height: 100dvh;
      padding: 10px;
      gap: 10px;
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 12px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 6px;
    }
    header h1 {
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--accent);
      text-shadow: var(--glow);
    }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%,100% { opacity: 1; }
      50%      { opacity: 0.3; }
    }

    /* ── Video feed ── */
    .video-wrapper {
      flex: 1;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      background: #000;
      position: relative;
      min-height: 0;
    }
    .video-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .crosshair {
      position: absolute;
      inset: 0;
      pointer-events: none;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .crosshair::before,
    .crosshair::after {
      content: '';
      position: absolute;
      background: rgba(0,229,255,0.5);
    }
    .crosshair::before { width: 1px; height: 30px; }
    .crosshair::after  { width: 30px; height: 1px; }
    .crosshair-ring {
      width: 40px; height: 40px;
      border: 1px solid rgba(0,229,255,0.4);
      border-radius: 50%;
    }
    .corner {
      position: absolute;
      width: 14px; height: 14px;
      border-color: var(--accent);
      border-style: solid;
      opacity: 0.5;
    }
    .corner.tl { top:8px;    left:8px;   border-width: 2px 0 0 2px; }
    .corner.tr { top:8px;    right:8px;  border-width: 2px 2px 0 0; }
    .corner.bl { bottom:8px; left:8px;   border-width: 0 0 2px 2px; }
    .corner.br { bottom:8px; right:8px;  border-width: 0 2px 2px 0; }

    /* ── Controls ── */
    .controls {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      grid-template-rows: auto auto auto;
      gap: 8px;
      align-items: center;
      justify-items: center;
      padding: 8px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 6px;
    }

    /* D-pad buttons */
    .btn {
      -webkit-tap-highlight-color: transparent;
      user-select: none;
      cursor: pointer;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #0d1117;
      color: var(--text);
      font-family: 'Rajdhani', sans-serif;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 10px 14px;
      min-width: 64px;
      text-align: center;
      transition: background 0.1s, border-color 0.1s, box-shadow 0.1s;
      touch-action: manipulation;
    }
    .btn:active, .btn.active {
      background: rgba(0,229,255,0.08);
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: var(--glow);
    }

    .btn-up    { grid-column: 2; grid-row: 1; }
    .btn-left  { grid-column: 1; grid-row: 2; }
    .btn-right { grid-column: 3; grid-row: 2; }
    .btn-down  { grid-column: 2; grid-row: 3; }

    /* Fire button */
    .btn-fire {
      grid-column: 1 / -1;
      width: 100%;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid rgba(255,45,85,0.4);
      background: rgba(255,45,85,0.08);
      color: var(--red);
      font-size: 1rem;
      letter-spacing: 0.2em;
      transition: background 0.1s, box-shadow 0.1s;
    }
    .btn-fire:active, .btn-fire.active {
      background: rgba(255,45,85,0.2);
      box-shadow: var(--red-glow);
    }

    /* Feedback toast */
    #toast {
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%) translateY(20px);
      background: rgba(0,229,255,0.1);
      border: 1px solid var(--accent);
      color: var(--accent);
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.75rem;
      padding: 6px 16px;
      border-radius: 4px;
      opacity: 0;
      transition: opacity 0.2s, transform 0.2s;
      pointer-events: none;
      z-index: 99;
    }
    #toast.show {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  </style>
</head>
<body>
<div class="app">

  <header>
    <h1>&#9678; Turret Control</h1>
    <div class="status-dot" id="statusDot"></div>
  </header>

  <div class="video-wrapper">
    <img src="/video_feed" alt="Camera feed" />
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <div class="crosshair">
      <div class="crosshair-ring"></div>
    </div>
  </div>

  <div class="controls">
    <button class="btn btn-up"    data-action="tilt_up">    ▲ UP    </button>
    <button class="btn btn-left"  data-action="pan_left">   ◀ LEFT  </button>
    <button class="btn btn-right" data-action="pan_right">  RIGHT ▶ </button>
    <button class="btn btn-down"  data-action="tilt_down">  ▼ DOWN  </button>
    <button class="btn btn-fire"  data-action="fire">       ◆ FIRE  </button>
  </div>

</div>

<div id="toast"></div>

<script>
  const toast = document.getElementById('toast');
  let toastTimer;

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1200);
  }

  async function sendCommand(action, btn) {
    btn.classList.add('active');
    setTimeout(() => btn.classList.remove('active'), 200);

    try {
      const res = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      const data = await res.json();
      showToast(data.action ? `→ ${data.action}` : data.message);
    } catch (err) {
      showToast('⚠ No connection');
      console.error(err);
    }
  }

  // Attach to all control buttons
  document.querySelectorAll('.btn[data-action]').forEach(btn => {
    btn.addEventListener('pointerdown', e => {
      e.preventDefault();
      sendCommand(btn.dataset.action, btn);
    });
  });

  // Keyboard support (handy for testing on desktop)
  const keyMap = {
    ArrowUp:    'tilt_up',
    ArrowDown:  'tilt_down',
    ArrowLeft:  'pan_left',
    ArrowRight: 'pan_right',
    ' ':        'fire',
  };
  document.addEventListener('keydown', e => {
    if (keyMap[e.key]) {
      e.preventDefault();
      const btn = document.querySelector(`[data-action="${keyMap[e.key]}"]`);
      if (btn) sendCommand(keyMap[e.key], btn);
    }
  });
</script>
</body>
</html>"""


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  NERF TURRET SERVER STARTING")
    print(f"  Open on your phone: http://<pi-ip>:{PORT}")
    print("  Find your IP with:  hostname -I")
    print("=" * 50)
    app.run(host=HOST, port=PORT, threaded=True)
