import cv2
from enum import Enum
import os

class ARMS(Enum):
    __order__ = "ZST1XX ZST2XX ZST3XX"
    ZST1XX = 1
    ZST2XX = 2
    ZST3XX = 3

class EXERCISES(Enum):
    __order__ = "SEATED_REACH_FORWARD SEATED_FORWARD_KICK SEATED_CALF_RAISE STANDING_REACH_ACROSS STANDING_WINDMILL STANDING_HIGH_KNEE"
    SEATED_REACH_FORWARD = 1
    SEATED_FORWARD_KICK = 2
    SEATED_CALF_RAISE = 3
    STANDING_REACH_ACROSS = 4
    STANDING_WINDMILL = 5
    STANDING_HIGH_KNEE = 6

class USER_RANGES:
    ZST1XX = [i for i in range(1, 12)]
    ZST2XX = [i for i in range(1, 11)]
    ZST3XX = [i for i in range(1, 22)]

VIDEO_NAME_MAP = {
    EXERCISES.SEATED_REACH_FORWARD: "seated_reach_forward_low",
    EXERCISES.SEATED_FORWARD_KICK: "seated_forward_kick",
    EXERCISES.SEATED_CALF_RAISE: "seated_calf_raise",
    EXERCISES.STANDING_REACH_ACROSS: "standing_reach_across",
    EXERCISES.STANDING_WINDMILL: "standing_windmills",
    EXERCISES.STANDING_HIGH_KNEE: "standing_high_knees",
}

ROOT = os.path.expanduser("~/Documents/data/zeste_studies/")
ROOT_1XX = os.path.join(ROOT, "zst1xx")
ROOT_2XX = os.path.join(ROOT, "zst2xx")
ROOT_3XX = os.path.join(ROOT, "zst3xx")

try:
    _files_1xx = os.listdir(ROOT_1XX)
    _files_2xx = os.listdir(ROOT_2XX)
    _files_3xx = os.listdir(ROOT_3XX)

    _files_1xx.sort()
    _files_2xx.sort()
    _files_3xx.sort()

    ZST1XX = [os.path.join(ROOT_1XX, user) for user in _files_1xx if user.startswith("zst1") and len(user) == 6]
    ZST2XX = [os.path.join(ROOT_2XX, user) for user in _files_2xx if user.startswith("zst2") and len(user) == 6]
    ZST3XX = [os.path.join(ROOT_3XX, user) for user in _files_3xx if user.startswith("zst3") and len(user) == 6]

    ZSTALL = ZST1XX + ZST2XX + ZST3XX
except Exception as e:
    pass

def check_videos_present():
    for user in ZSTALL:
        if not os.path.exists(user):
            print(f"User {user} does not exist.")
            
        if not os.path.isdir(user):
            print(f"User {user} is not a directory.")

        video_dir = os.path.join(user, "sws_videos")
        if not os.path.exists(video_dir):
            other_possibility = os.path.join(user, "sws_video")
            if not os.path.exists(other_possibility):
                print(f"User {user[-6:]} does not have a video directory.")
            else:
                print(f"User {user[-6:]} has a mislabeled video directory.")

def get_arm_dirs(arm: ARMS):
    if arm == ARMS.ZST1XX:
        return ZST1XX
    elif arm == ARMS.ZST2XX:
        return ZST2XX
    elif arm == ARMS.ZST3XX:
        return ZST3XX
    else:
        return ZSTALL

def load_exercise(arm: ARMS, exercise: EXERCISES, user: int):
    user_str = f"zst{arm.value}{user:02d}"
    user_dir = os.path.join(ROOT, f"zst{arm.value}xx", user_str)
    if not os.path.exists(user_dir):
        print(f"User {user_str} does not exist.")
        return None
    
    # edge case
    if arm == ARMS.ZST1XX:
        if user == 2 or user == 3:
            user_dir = os.path.join(ROOT, f"zst{arm.value}xx", user_str + "_old_err")
    
    video_dir = os.path.join(user_dir, "sws_videos")
    
    if not os.path.exists(video_dir):
        video_dir = os.path.join(user_dir, "sws_video")

    if not os.path.exists(video_dir):
        print(f"User {user_str} does not have a video directory.")
        return None
    
    # filter videos
    video_files = os.listdir(video_dir)
    video_files.sort()
    exercise_video_files = [f for f in video_files if VIDEO_NAME_MAP[exercise] in f]
    rgb_exercise_video_files = [f for f in exercise_video_files if "rgb" in f]

    if len(rgb_exercise_video_files) == 0:
        print(f"User {user_str} does not have an {exercise} video.")
        return None
    
    frames_per_set = []
    for video in rgb_exercise_video_files:
        video_path = os.path.join(video_dir, video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}.")
            continue
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        frames_per_set.append(frames)
    
    # remove calibration and practice sets
    frames_per_set = frames_per_set[-4:]
    return frames_per_set

def main(args):
    if args.check:
        check_videos_present()
        return
    
    if args.test:
        # frames_per_set = load_exercise(ARMS.ZST1XX, EXERCISES.SEATED_REACH_FORWARD, 11)
        # print(f"Loaded {len(frames_per_set)} sets of frames.")
        # for i, frames in enumerate(frames_per_set):
        #     print(f"Set {i}: {len(frames)} frames.")

        test_users = [i + 1 for i in range(12)]
        for user in test_users:
            frames_per_set = load_exercise(ARMS.ZST1XX, EXERCISES.STANDING_HIGH_KNEE, user)
            if frames_per_set is not None:
                print(f"Loaded {len(frames_per_set)} sets of frames for user {user}.")
                frame_lens = [len(frames) for frames in frames_per_set]
                print(f"Frame lengths: {frame_lens}")
        return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zeste data loader")
    parser.add_argument("--check", action="store_true", help="Check if all videos are present")
    parser.add_argument("--test", action="store_true", help="Test loading an exercise")
    args = parser.parse_args()

    main(args)
