import cv2
import mediapipe as mp
import numpy as np

from tqdm import tqdm

from zeste_vision.data_tools.loader import get_test_video_frames

DRAWING = mp.solutions.drawing_utils

MEDIAPIPE_OPTIONS = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "smooth_segmentation": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

def test(frame_i: int = 10):
    print("occlusion_analysis::test: loading frames")
    frames = get_test_video_frames()
    print("occlusion_analysis::test: frames loaded")

    # get test frame
    test_frame = frames[frame_i]
    cv2.imshow("Test frame", test_frame)
    cv2.waitKey(0)

    # instantiate pose estimator
    pose = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

    # run pose estimator on test frame
    results = pose.process(test_frame)
    print("occlusion_analysis::test: pose estimation complete")

    render_image = test_frame.copy()
    DRAWING.draw_landmarks(render_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    cv2.imshow("MediaPipe Pose", render_image)
    cv2.waitKey(0)

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
