import os
import cv2
import numpy as np
import csv
from yolo.pose import pose_detector

def draw_tennis_court_rect(frame, corners, color=(0, 255, 0), thickness=2):
    cv2.polylines(frame, [np.array(corners, np.int32)], isClosed=True, color=color, thickness=thickness)

def box_center_in_court(box, court_polygon):
    center_x = int((box[0] + box[2]) / 2)
    center_y = int((box[1] + box[3]) / 2)
    return cv2.pointPolygonTest(court_polygon, (center_x, center_y), False) >= 0

def save_detections_to_csv(detections, filepath):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'x1', 'y1', 'x2', 'y2'])
        for detection in detections:
            writer.writerow(detection)

# Set up the data paths
dirname = os.path.dirname(__file__)
output_dir = os.path.join(dirname, 'player_data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_dictionary = {
    'game1': [f'Clip{i+1}' for i in range(13)],
    'game2': [f'Clip{i+1}' for i in range(8)],
    'game3': [f'Clip{i+1}' for i in range(9)],
    'game4': [f'Clip{i+1}' for i in range(7)],
    'game5': [f'Clip{i+1}' for i in range(15)],
    'game6': [f'Clip{i+1}' for i in range(4)],
    'game7': [f'Clip{i+1}' for i in range(9)],
}

tennis_court_corners = [(400, 100), (900, 100), (1200, 700), (100, 700)]
court_polygon = np.array(tennis_court_corners, np.int32)

# Process each game and clip directory
for game, clips in data_dictionary.items():
    for clip in clips:
        clip_dir = os.path.join(dirname, 'Dataset', game, clip)
        csv_filename = f'{game}_{clip}_player_positions.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        all_detections = []

        for filename in os.listdir(clip_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(clip_dir, filename)
                frame = cv2.imread(image_path)
                if frame is None:
                    continue

                results = pose_detector.predict(image_path, conf=0.2, imgsz=(736, 1280), max_det=2, save=False)
                draw_tennis_court_rect(frame, tennis_court_corners)
                
                for r in results:
                    detections = r.boxes.xyxy.numpy()
                    for box in detections:
                        if box_center_in_court(box, court_polygon):
                            all_detections.append([filename] + list(map(int, box[:4])))

        save_detections_to_csv(all_detections, csv_path)
