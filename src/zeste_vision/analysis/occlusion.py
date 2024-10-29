import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from tqdm import tqdm

from zeste_vision.data_tools.loader import get_test_video_frames

DRAWING = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

MEDIAPIPE_OPTIONS = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "smooth_segmentation": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

def test(frame_i: int = 10, use_cv: bool = True):
    print("occlusion_analysis::test: loading frames")
    video_path = "/home/lamsey/hrl/zeste_vision/data/aist_dance/"
    video_file = video_path + "gBR_sBM_c01_d04_mBR0_ch01.mp4"
    frames = get_test_video_frames(video_file)
    print("occlusion_analysis::test: frames loaded")

    # get test frame
    test_frame = frames[frame_i]
    if use_cv:
        cv2.imshow("Test frame", test_frame)
        cv2.waitKey(0)
    else:
        rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_frame)
        plt.show()

    # instantiate pose estimator
    pose = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

    # run pose estimator on test frame
    results = pose.process(test_frame)
    print("occlusion_analysis::test: pose estimation complete")

    render_image = test_frame.copy()
    DRAWING.draw_landmarks(render_image, results.pose_landmarks, POSE_CONNECTIONS)

    if use_cv:
        cv2.imshow("MediaPipe Pose", render_image)
        cv2.waitKey(0)
    else:
        rgb_frame = cv2.cvtColor(render_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_frame)
        plt.show()

def main(args):
    if args.test:
        test()
        return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Occlusion analysis")
    parser.add_argument("--test", action="store_true", help="Run tests")

    args = parser.parse_args()
    main(args)
