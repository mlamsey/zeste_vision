import torch
from torch.utils.data import Dataset
import pandas as pd
import json

def clean_poses(poses):
    for i in range(len(poses)):
        pose = poses[i]
        if pose is None:
            pose = torch.zeros(33, 5).tolist()
            poses[i] = pose

    return poses

class PoseDateset(Dataset):
    def __init__(self, label_df: pd.DataFrame):
        """
        Args:
            label_df (pd.DataFrame): DataFrame with columns 'File Path' and 'Error'.
        """

        self.data_table = label_df

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        row = self.data_table.iloc[idx]
        
        # hacky
        pose_file = row['File Path']
        pose_file = pose_file.replace("split_videos", "split_poses")
        pose_file = pose_file.replace(".mp4", ".json")

        poses = json.load(open(pose_file))
        poses = clean_poses(poses)
        poses = torch.tensor(poses)
        poses = torch.transpose(poses, 0, 2)

        label = row["Error"]
        label = 1. if label == "Error" else 0.
        label = torch.tensor(label)

        # to device
        poses = poses.to("cuda")
        label = label.to("cuda")
    
        return poses, label


class VideoDateset(Dataset):
    def __init__(self, csv_file: str):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            video_dir (string): Directory with all the videos.
        """

        self.data_table = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        row = self.data_table.iloc[idx]
        
        # hacky
        pose_file = row['File Path']
        pose_file = pose_file.replace("split_videos", "split_poses")
        pose_file = pose_file.replace(".mp4", ".json")

        poses = json.load(open(pose_file))
        poses = torch.tensor(poses)

        label = row["Error"]
    
        return poses, label
    
    # def load_video(self, participant_id: str, exercise: str, set_i: int):
    #     participant_id = participant_id.lower().replace(' ', '_')
    #     participant_path = os.path.join(self.video_dir, participant_id)

    #     video_files = os.listdir(participant_path)
    #     exercise_video_files = [f for f in video_files if exercise in f]
    #     exercise_video_files.sort()

    #     if set_i >= len(exercise_video_files):
    #         return None
            
    #     set_video_file = exercise_video_files[set_i - 1]

    #     video_path = os.path.join(participant_path, set_video_file)

    #     cap = cv2.VideoCapture(video_path)

    #     frames = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frames.append(frame)

    #     return frames