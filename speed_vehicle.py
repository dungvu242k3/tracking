import math
import time

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

model = YOLO('best.pt')
tracker = DeepSort(max_age=30)
video_path = "highway1.mp4"
cap = cv2.VideoCapture(video_path)
PIXELS_PER_METER = 5

frame_count = 0
start_time = time.time()

while frame_count < 150:
    ret, _ = cap.read()
    if not ret:
        break
    frame_count += 1

end_time = time.time()
fps = 50
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

object_history = {}

def calculate_speed(start_pos, current_pos, fps):
    dx = current_pos[0] - start_pos[0]
    dy = current_pos[1] - start_pos[1]
    distance_pixels = math.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels / PIXELS_PER_METER
    speed_mps = distance_meters * fps
    return speed_mps * 3.6

frame_index = 0
FRAME_INTERVAL = 5
last_speed = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) in [2, 3, 5, 7]:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, conf, "vehicle"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        if track_id in object_history:
            prev_cx, prev_cy, prev_frame = object_history[track_id]
            if frame_index - prev_frame >= FRAME_INTERVAL:
                speed = calculate_speed((prev_cx, prev_cy), (cx, cy), fps / FRAME_INTERVAL)
                object_history[track_id] = (cx, cy, frame_index)
                last_speed[track_id] = speed
            else:
                speed = last_speed.get(track_id, 0)
        else:
            object_history[track_id] = (cx, cy, frame_index)
            speed = 0
            last_speed[track_id] = 0

        label = f"ID {track_id} | {speed:.1f} km/h"
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    frame_index += 1
    cv2.imshow("Theo dõi tốc độ xe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
