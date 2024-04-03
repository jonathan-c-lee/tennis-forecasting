import os
from yolo.pose import pose_detector


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    video_source = os.path.join(dirname, './videos/test.mp4')
    pose_detector.predict(video_source, imgsz=(720, 1280), max_det=2, save=True)