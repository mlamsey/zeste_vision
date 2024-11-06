import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

import json
from tqdm import tqdm

from zeste_vision.data_tools.loader import get_test_video_frames
from zeste_vision.analysis.pose_tools import add_occlusion, crop_image, compute_bounding_box
from zeste_vision.utils import mpjpe

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

def add_occlusion_to_frame(frame, results, occlusion_pct: float=0.5):
    if results.segmentation_mask is None:
        return None
    mask = results.segmentation_mask.astype(np.uint8)
    bbox = cv2.boundingRect(mask)
    return add_occlusion(frame, bbox, occlusion_pct)

def get_pose_np(pose_landmarks):
    if pose_landmarks is None:
        return None
    
    return np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])

def evaluate_occlusion():
    test_frames = get_test_video_frames()
    pose_estimator = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

    n_occlusion_bands = 10
    occlusion_percents = np.linspace(0., 0.9, n_occlusion_bands)

    error_dict = {o: [] for o in occlusion_percents}
    dropped_dict = {o: 0 for o in occlusion_percents}
    n_rand = 5

    for occlusion_pct in occlusion_percents:
        error = []
        dropped = 0

        print("Testing for Occlusion Percent: ", occlusion_pct)
        for frame in tqdm(test_frames):
            frame_error = []
            for _ in range(n_rand):
                base_results = pose_estimator.process(frame)
                occluded_frame = add_occlusion_to_frame(frame, base_results, occlusion_pct)
                if occluded_frame is None:
                    dropped += 1
                    continue
                occluded_results = pose_estimator.process(occluded_frame)
                base_pose = get_pose_np(base_results.pose_landmarks)
                occluded_pose = get_pose_np(occluded_results.pose_landmarks)
                mpjpe_error = mpjpe(base_pose, occluded_pose)
                
                if mpjpe_error is None:
                    # plt.imshow(occluded_frame)
                    # plt.show()
                    dropped += 1
                    continue

                frame_error.append(mpjpe_error)

            if len(frame_error) == 0:
                continue
            
            error.append(np.mean(frame_error))

        # print(error)
        error = [e for e in error if e is not None]
        error_dict[occlusion_pct] = error
        dropped_dict[occlusion_pct] = dropped
        print(f"Mean MPJPE for {occlusion_pct}: {np.mean(error)}")

    file_prefix = f"occlusion_{n_rand}_rand_{n_occlusion_bands}_bands_"
    json.dump(error_dict, open(file_prefix + "error.json", "w"))
    json.dump(dropped_dict, open(file_prefix + "dropped.json", "w"))

def evaluate_cropping():
    test_frames = get_test_video_frames()
    pose_estimator = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

    n_crops = 10
    crop_percents = np.linspace(0., 0.9, n_crops)

    error_dict = {c: [] for c in crop_percents}
    dropped_dict = {c: 0 for c in crop_percents}
    # n_rand = 5

    for i, crop_pct in enumerate(crop_percents):
        error = []
        dropped = 0

        print("Testing for Crop Percent: ", crop_pct)
        for frame in tqdm(test_frames):
            base_results = pose_estimator.process(frame)
            bbox = compute_bounding_box(base_results)
            cropped_frame = crop_image(frame, bbox, crop_pct)
            if cropped_frame is None:
                dropped += 1
                continue
            cropped_results = pose_estimator.process(cropped_frame)
            base_pose = get_pose_np(base_results.pose_landmarks)
            cropped_pose = get_pose_np(cropped_results.pose_landmarks)
            mpjpe_error = mpjpe(base_pose, cropped_pose)
            
            if mpjpe_error is None:
                # plt.imshow(cropped_frame)
                # plt.show()
                dropped += 1
                continue
            
            error.append(np.mean(mpjpe_error))

        # print(error)
        error = [e for e in error if e is not None]
        error_dict[crop_pct] = error
        dropped_dict[crop_pct] = dropped
        print(f"Mean MPJPE for {crop_pct}: {np.mean(error)}")

    file_prefix = f"crop_{n_crops}_"
    json.dump(error_dict, open(file_prefix + "error.json", "w"))
    json.dump(dropped_dict, open(file_prefix + "dropped.json", "w"))

def test(frame_i: int = 10, use_cv: bool = True):
    print("occlusion_analysis::test: loading frames")
    video_path = "/data/aist_dance/test/"
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
    
    if args.eval:
        evaluate_occlusion()
        return
    
    if args.crop:
        evaluate_cropping()
        return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Occlusion analysis")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--eval", action="store_true", help="Evaluate occlusion")
    parser.add_argument("--crop", action="store_true", help="Evaluate cropping")

    args = parser.parse_args()
    main(args)
