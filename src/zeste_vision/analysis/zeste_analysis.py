import cv2
import mediapipe as mp
import numpy as np

import json
from tqdm import tqdm
import pickle as pkl

from zeste_vision.data_tools.zeste_loader import *
from zeste_vision.utils import mpjpe, get_pose_np

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

def estimate_pose_frame(frame, pose_estimator) -> np.ndarray:
    result = pose_estimator.process(frame)
    return result, get_pose_np(result.pose_landmarks)

def _get_users(arm):
    if arm == ARMS.ZST1XX:
        return USER_RANGES.ZST1XX
    elif arm == ARMS.ZST2XX:
        return USER_RANGES.ZST2XX
    elif arm == ARMS.ZST3XX:
        return USER_RANGES.ZST3XX
    else:
        return []
    
def analyze(arm: ARMS, save_json: bool=True, save_pickle: bool=False):
    pose_estimator = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)
    users = _get_users(arm)

    # testing
    # users = [users[4]]

    for user in users:
        print(f"User {user}")
        pose_memory = {exercise: {} for exercise in EXERCISES}
        for exercise in EXERCISES:
            print(f"Exercise {exercise.name}")
            frames = load_exercise(arm=arm, exercise=exercise, user=user)

            exercise_memory = {f"set{i}": [] for i in range(4)}

            for i, frame_set in enumerate(frames):
                print(f"\tSet {i}")
                poses = []
                for frame in tqdm(frame_set):
                    result, pose = estimate_pose_frame(frame, pose_estimator=pose_estimator)
                    if pose is not None:
                        pose = pose.tolist()
                        
                    poses.append(pose)
                exercise_memory[f"set{i}"] = poses
            
            pose_memory[exercise] = exercise_memory

        file_name = f"zst{arm.value}{user:02d}.json"
        pose_memory_dump = {key.name: value for key, value in pose_memory.items()}
        if save_json:
            json.dump(pose_memory_dump, open(file_name, "w"))
        if save_pickle:
            pkl.dump(result, open(file_name.replace(".json", ".pkl"), "wb"))

def main(args):
    if args.all:
        for arm in ARMS:
            analyze(arm, save_json=False, save_pickle=True)
        return
    
    elif args.arm:
        analyze(ARMS(args.arm), save_json=False, save_pickle=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=int)
    parser.add_argument("--all", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
                