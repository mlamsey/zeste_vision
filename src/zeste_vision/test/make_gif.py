import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

# front matter
DRAWING = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

MEDIAPIPE_OPTIONS = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "enable_segmentation": True,
    "smooth_segmentation": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# video_path = "/home/lamsey/hrl/zeste_vision/data/aist_dance/gBR_sBM_c01_d04_mBR0_ch01.mp4"
video_path = "/home/lamsey/hrl/zeste_vision/data/aist_dance/gHO_sBM_c01_d19_mHO0_ch07.mp4"

# make output video
out_path = "/home/lamsey/hrl/zeste_vision/data/aist_dance/_test2_med.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, 60, (1920 // 2, 1080))

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file.")
    exit()

pose_estimator = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_estimator.process(frame)
    if results.pose_landmarks is not None:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_annots = frame.copy()
        frame.flags.writeable = True
        DRAWING.draw_landmarks(frame_annots, results.pose_landmarks, POSE_CONNECTIONS)
        
        stacked_frame = np.vstack((frame, frame_annots))
        stacked_frame = cv2.resize(stacked_frame, (1920 // 2, 1080))
        out.write(stacked_frame)

cap.release()
out.release()
