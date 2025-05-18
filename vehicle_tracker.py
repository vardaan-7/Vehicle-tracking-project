import cv2
import os
from ultralytics import YOLO
import datetime
import mysql.connector
import logging
import numpy as np
from config import DB_CONFIG  # Import database configuration

# used to create log file each day
today = datetime.datetime.now().strftime('%Y-%m-%d')
os.makedirs("logs_with_callback", exist_ok=True)
info_log_path = f"logs_with_callback/info_{today}.log"
error_log_path = f"logs_with_callback/error_{today}.log"

# Configure info logger
logging.basicConfig(level=logging.INFO)
info_logger = logging.getLogger('info_logger')
info_handler = logging.FileHandler(info_log_path)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
info_logger.addHandler(info_handler)
info_logger.propagate = False

# Configure error logger
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(error_log_path)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
error_logger.addHandler(error_handler)
error_logger.propagate = False

#taking value of line y1 and y as none in the starting 
selected_lines = [None, None] 
click_index = 0
#line setting mode kept as false, so that line cant be changed anytime. 
line_setting_mode = False
press_s = "press s to change line position" 
press_r = "press r to make restricted parking"
quad_points = []
drawing_quadrilateral = False

#function for mouse call back 
def mouse_callback(event, x, y, flags, param):
    global selected_lines, click_index, line_setting_mode, drawing_quadrilateral, quad_points
    if line_setting_mode and event == cv2.EVENT_LBUTTONDOWN:
        if click_index < 2:
            selected_lines[click_index] = y
            print(f"Line {click_index + 1} set at Y = {y}")
            click_index += 1
            if click_index == 2:
                line_setting_mode = False
    if drawing_quadrilateral and event == cv2.EVENT_LBUTTONDOWN:
        if len(quad_points) < 4:
            quad_points.append((x, y))
            print(f"Point {len(quad_points)} set at ({x}, {y})")
        if len(quad_points) == 4:
            drawing_quadrilateral = False
            print("Quadrilateral selection complete.")            

class VehicleTracker:
    def __init__(self, video_path, model_path="yolov8n.pt", line_y1=200, line_y2=400, window_name="Video"):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.class_names = self.model.model.names
        self.line_y = line_y2
        self.line_y1 = line_y1
        self.window_name = window_name
        self.distance_m = 10
        self.stationary_start_time = {}
        self.stationary_vehicles = set()

        # setting up sql 
        try:
            self.mydb = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.mydb.cursor()
        except Exception as e:
            error_logger.error(f"Database connection failed: {e}")
            raise

        self.cross_times = {}
        self.crossed_ids = set()
        self.vehicle_counts = {}
        self.prev_position = {}
        self.speed_data = {}

        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_folder = os.path.join("captured_frames", self.video_name)
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        global quad_points, drawing_quadrilateral
        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Fallback resolution if video doesn't load correctly
        if frame_width <= 0 or frame_height <= 0:
            frame_width, frame_height = 1280, 720

        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        cv2.resizeWindow(self.window_name, min(1280, frame_width), min(720, frame_height))

        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            if selected_lines[0] is not None:
                self.line_y1 = selected_lines[0]
            if selected_lines[1] is not None:
                self.line_y = selected_lines[1]

            if self.line_y1 is not None:
                cv2.line(frame, (0, self.line_y1), (frame_width, self.line_y1), (223, 237, 69), 3)
            if self.line_y is not None:
                cv2.line(frame, (0, self.line_y), (frame_width, self.line_y), (223, 237, 69), 3)

            results = self.model.track(frame, persist=True, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.int().tolist()
                xyxy = boxes.xyxy.tolist()
                classes = boxes.cls.int().tolist()

                for i, box in enumerate(xyxy):
                    obj_id = ids[i]
                    cls_id = classes[i]
                    class_name = self.class_names[cls_id]
                    x1, y1, x2, y2 = box
                    rear_y = int(y1 - 5)
                    prev_y = self.prev_position.get(obj_id, -1)

                    movement = abs(rear_y - prev_y)

                    if movement < 2:  # pixel threshold for movement
                        if obj_id not in self.stationary_start_time:
                            self.stationary_start_time[obj_id] = datetime.datetime.now()
                        else:
                            elapsed = (datetime.datetime.now() - self.stationary_start_time[obj_id]).total_seconds()
                            if elapsed > 3 and obj_id not in self.stationary_vehicles:
                                self.stationary_vehicles.add(obj_id)
                    else:
                        # Vehicle moved, reset timer and remove from stationary list
                        if obj_id in self.stationary_start_time:
                            del self.stationary_start_time[obj_id]
                        if obj_id in self.stationary_vehicles:
                            self.stationary_vehicles.remove(obj_id)

                    # Handle entry line crossing
                    if prev_y < self.line_y1 <= rear_y:
                        if obj_id not in self.cross_times:
                            self.cross_times[obj_id] = {}
                        if 'entry' not in self.cross_times[obj_id]:
                            self.cross_times[obj_id]['entry'] = datetime.datetime.now()

                    # Handle exit line crossing and speed calculation
                    if prev_y < self.line_y <= rear_y:
                        if obj_id not in self.cross_times:
                            self.cross_times[obj_id] = {}
                        if 'entry' in self.cross_times[obj_id] and 'exit' not in self.cross_times[obj_id]:
                            self.cross_times[obj_id]['exit'] = datetime.datetime.now()
                            time_diff = (self.cross_times[obj_id]['exit'] - self.cross_times[obj_id]['entry']).total_seconds()

                            if time_diff > 0:
                                speed = round((self.distance_m / time_diff) * 3.6, 2)

                                if obj_id not in self.speed_data:
                                    # Store the speed once calculated
                                    self.speed_data[obj_id] = speed

                                    if obj_id not in self.crossed_ids:
                                        self.crossed_ids.add(obj_id)
                                        self.vehicle_counts[class_name] = self.vehicle_counts.get(class_name, 0) + 1
                                        filename = f"frame{frame_number}_id{obj_id}.jpg"
                                        filepath = os.path.join(self.output_folder, filename)

                                        try:
                                            cv2.imwrite(filepath, frame)
                                            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            self.cursor.execute("INSERT INTO vehicle_data (image_path, saved_time, speed) VALUES (%s, %s, %s)", (filepath, now, speed))
                                            self.mydb.commit()
                                            info_logger.info(f"Image saved and DB updated: {filename}, Speed: {speed} km/h")
                                        except Exception as e:
                                            error_logger.error(f"Error saving image or updating DB for {filename}: {e}")
                    vehicle_point = (int((x1 + x2) / 2), int(y2))
                    is_stationary = obj_id in self.stationary_vehicles
                    parked_wrongly = False

                    if len(quad_points) == 4 and is_stationary:
                        contour = cv2.convexHull(np.array(quad_points, dtype=np.int32))
                        contour = contour.reshape((-1, 1, 2))
                        dist = cv2.pointPolygonTest(contour, vehicle_point, False)
                        if dist >= 0:
                            parked_wrongly = True

                    # Set color and bounding box
                    if obj_id in self.cross_times and 'exit' in self.cross_times[obj_id]:
                        color = (0, 0, 255)  # Red if crossed both lines
                    else:
                        color = (0, 255, 0)  # Green if not yet crossed both lines
                    
                    speed_text = f"{self.speed_data.get(obj_id, 'Calculating...')} km/h" if obj_id in self.speed_data else "Calculating..."

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame, f"{class_name} {speed_text}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if obj_id in self.stationary_vehicles:
                        cv2.putText(frame, "STATIONARY", (int(x1), int(y2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if parked_wrongly:
                        cv2.putText(frame, "PARKED WRONGLY", (int(x1), int(y2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3) 

                    self.prev_position[obj_id] = rear_y
        
            # Display vehicle counts and total count
            y_offset = 30
            for cls_name, count in self.vehicle_counts.items():
                text = f"{cls_name.upper()}: {count}"
                cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (237, 9, 9), 2)
                y_offset += 40

            total = sum(self.vehicle_counts.values())
            cv2.putText(frame, f"Total: {total}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if len(quad_points) == 4:
                for i in range(4):
                    cv2.line(frame, quad_points[i], quad_points[(i + 1) % 4], (255, 0, 255), 2)
            cv2.putText(frame, press_s, (frame_width - 550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, press_r, (frame_width - 590, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Line setting mode enabled. Click on the video to set entry and exit lines.")
                global click_index, line_setting_mode
                click_index = 0
                line_setting_mode = True
                selected_lines[0] = None
                selected_lines[1] = None
            elif key == ord('r'):
                print("Quadrilateral selection mode enabled. Click 4 points.")
                quad_points = []
                drawing_quadrilateral = True

        cap.release()
        cv2.destroyWindow(self.window_name)