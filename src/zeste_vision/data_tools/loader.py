import os
import cv2

def _get_test_video_path():
    try:
        data_dir = "/data/aist_dance"
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".mp4")]
        files.sort()
        return os.path.join(data_dir, files[0])
    except Exception as e:
        print(f"Error getting test video path: {e}")
        return None

def get_test_video_frames(video_file: str = None):
    if video_file is None:
        video_file = _get_test_video_path()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def get_test_video_frame(video_file: str = None, frame_i: int = 0):
    if video_file is None:
        video_file = _get_test_video_path()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    for _ in range(frame_i):
        ret, frame = cap.read()
        if not ret:
            break
    return frame
