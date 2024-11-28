import cv2
import json
import mediapipe as mp
import os
from tqdm import tqdm

from zeste_vision.analysis.zeste_analysis import estimate_pose_frame

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

def bulk_pose_estimates(video_dir: str, pose_dir: str):
    """
    Args:
        video_dir: str
        pose_dir: str
    """

    pose_estimator = mp.solutions.pose.Pose(**MEDIAPIPE_OPTIONS)

    train_dir = os.path.join(video_dir, "train")
    eval_dir = os.path.join(video_dir, "eval")
    pose_train_dir = os.path.join(pose_dir, "train")
    pose_eval_dir = os.path.join(pose_dir, "eval")

    for in_dir, out_dir in zip([train_dir, eval_dir], [pose_train_dir, pose_eval_dir]):
        input_dirs = os.listdir(in_dir)
        input_dirs = [d for d in input_dirs if "zst" in d]
        input_dirs.sort()

        for user in input_dirs:
            print(f"Processing {user}")
            user_dir = os.path.join(in_dir, user)
            files = os.listdir(user_dir)

            for file in tqdm(files):
                if file.endswith(".mp4"):
                    # print(f"Processing {file}")
                    # process video
                    cap = cv2.VideoCapture(os.path.join(user_dir, file))
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)

                    # process frames
                    poses = []
                    for frame in frames:
                        norm_coords, world_coords = estimate_pose_frame(frame, pose_estimator)
                        if world_coords is not None:
                            world_coords = world_coords.tolist()
                        poses.append(world_coords)
        
                # print(poses)
                # print(file)
                # print(len(poses))
                file_json = file.replace(".mp4", ".json")
                user_out_dir = os.path.join(out_dir, user)
                if not os.path.exists(user_out_dir):
                    os.makedirs(user_out_dir)

                with open(os.path.join(out_dir, user, file_json), "w") as f:
                    json.dump(poses, f)

if __name__ == '__main__':
    # hardcoded bc lazy
    video_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_videos"
    pose_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses"

    bulk_pose_estimates(video_dir, pose_dir)
