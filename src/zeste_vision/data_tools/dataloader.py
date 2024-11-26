import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import cv2  # or any other library for video processing

class ZesteVisionDataset(Dataset):
    def __init__(self, csv_file: str, video_dir: str):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            video_dir (string): Directory with all the videos.
        """

        self.annotations = pd.read_csv(csv_file)
        self.video_dir = video_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        participant_id, exercise, set_i = row['Participant ID'], row['Exercise'], row['Set']
        error_label = row['Error']

        # Load video file
        video = self.load_video(participant_id, exercise, set_i)

        return video, error_label

    def load_video(self, participant_id: str, exercise: str, set_i: int):
        participant_id = participant_id.lower().replace(' ', '_')
        participant_path = os.path.join(self.video_dir, participant_id)

        video_files = os.listdir(participant_path)
        exercise_video_files = [f for f in video_files if exercise in f]
        exercise_video_files.sort()

        if set_i >= len(exercise_video_files):
            return None
            
        set_video_file = exercise_video_files[set_i - 1]

        video_path = os.path.join(participant_path, set_video_file)

        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        return frames