import math
import time

import cv2
import dlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

video = cv2.VideoCapture('highway3.mp4')
model = YOLO("best.pt")

f_width = 1280
f_height = 720

pixels_per_meter = 1

frame_idx = 0
car_number = 0
fps = 0

carTracker = {}
carNumbers = {}
carStartPosition = {}
carCurrentPosition = {}
speed = [None] * 1000

def remove_bad_tracker():
	global carTracker, carStartPosition, carCurrentPosition

	delete_id_list = []

	for car_id in carTracker.keys():
		if carTracker[car_id].update(image) < 10:
			delete_id_list.append(car_id)

	for car_id in delete_id_list:
		carTracker.pop(car_id, None)
		carStartPosition.pop(car_id, None)
		carCurrentPosition.pop(car_id, None)

	return

def calculate_speed(startPosition, currentPosition, fps):

	distance_in_pixels = math.sqrt(math.pow(currentPosition[0] - startPosition[0], 2) + math.pow(currentPosition[1] - startPosition[1], 2))

	distance_in_meters = distance_in_pixels / pixels_per_meter

	speed_in_meter_per_second = distance_in_meters * fps
	speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

	return speed_in_kilometer_per_hour



while True:
	start_time = time.time()
	_, image = video.read()

	if image is None:
		break

	image = cv2.resize(image, (f_width, f_height))
	output_image = image.copy()

	frame_idx += 1
	remove_bad_tracker()

	if not (frame_idx % 10):
		result = model(image)[0]
		for box in result.boxes:
			cls_id = int(box.cls[0])
			if cls_id == 2 :
				x1,y1,x2,y2 = map(int,box.xyxy[0])
				x = x1
				y = y1
				w = x2 - x1
				h = y2 - y1
			x_center = x + 0.5 * w
			y_center = y + 0.5 * h


			matchCarID = None
			
			for carID in carTracker.keys():
				trackedPosition = carTracker[carID].get_position()
				t_x = int(trackedPosition.left())
				t_y = int(trackedPosition.top())
				t_w = int(trackedPosition.width())
				t_h = int(trackedPosition.height())
				t_x_center = t_x + 0.5 * t_w
				t_y_center = t_y + 0.5 * t_h

				if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
					matchCarID = carID

			if matchCarID is None:

				tracker = dlib.correlation_tracker()
				tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

				carTracker[car_number] = tracker
				carStartPosition[car_number] = [x, y, w, h]

				car_number +=1

	for carID in carTracker.keys():
		trackedPosition = carTracker[carID].get_position()

		t_x = int(trackedPosition.left())
		t_y = int(trackedPosition.top())
		t_w = int(trackedPosition.width())
		t_h = int(trackedPosition.height())

		cv2.rectangle(output_image, (t_x, t_y), (t_x + t_w, t_y + t_h), (255,0,0), 4)
		carCurrentPosition[carID] = [t_x, t_y, t_w, t_h]

	end_time = time.time()
	if not (end_time == start_time):
		fps = 1.0/(end_time - start_time)

	for i in carStartPosition.keys():
			[x1, y1, w1, h1] = carStartPosition[i]
			[x2, y2, w2, h2] = carCurrentPosition[i]

			carStartPosition[i] = [x2, y2, w2, h2]

			if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
				if (speed[i] is None or speed[i] == 0) and y2<100:
					speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2],fps)

				if speed[i] is not None and y2 >= 100:
					cv2.putText(output_image, str(int(speed[i])) + " km/h",
								(x2,  y2),cv2.FONT_HERSHEY_SIMPLEX, 1,
								(0, 255, 255), 2)

	cv2.imshow("video",output_image)
	if cv2.waitKey(1) == ord("q") :
		break

cv2.destroyAllWindows()
