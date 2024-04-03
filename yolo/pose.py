"""YOLO pose detection model."""
from ultralytics import YOLO


pose_detector = YOLO('yolov8n-pose.pt')