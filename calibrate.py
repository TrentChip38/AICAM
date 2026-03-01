#!/usr/bin/env python3
"""
LED HSV Calibrator - click on your LEDs in the window to see their HSV values
Run: python3 calibrate_leds.py
"""
import cv2
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]
        if frame is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[y, x]
            bgr = frame[y, x]
            print(f"Clicked ({x},{y}) -> HSV: H={h} S={s} V={v}  |  BGR: {bgr}")
            print(f"  Suggested lower: [{max(0,h-10)}, {max(0,s-40)}, {max(0,v-40)}]")
            print(f"  Suggested upper: [{min(180,h+10)}, 255, 255]")

frame_holder = {"frame": None}
cv2.namedWindow("Calibrate - Click on your LED")
cv2.setMouseCallback("Calibrate - Click on your LED", mouse_click, frame_holder)

print("Click directly on your LEDs in the window.")
print("Press Q to quit.")

while True:
    frame = picam2.capture_array("main")
    frame_holder["frame"] = frame.copy()

    # Also show HSV channel H as color map to help visualize
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:,:,0]
    h_colored = cv2.applyColorMap(h_channel * 2, cv2.COLORMAP_HSV)

    combined = np.hstack([frame, h_colored])
    combined = cv2.resize(combined, (1280, 480))
    cv2.putText(combined, "Original | Hue Map (click original side on LED)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imshow("Calibrate - Click on your LED", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

#Run it, then **click directly on your red LED** in the left window. It will print the exact HSV values like:
#```
#Clicked (320, 240) -> HSV: H=172 S=230 V=200
  #Suggested lower: [162, 190, 160]
  #Suggested upper: [180, 255, 255]