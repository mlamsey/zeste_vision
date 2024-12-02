import os
import pandas as pd

from torch.utils.data import DataLoader

from zeste_vision.data_tools.dataloader import PoseDateset, clean_poses

def test_pose_dataset():
    data_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses/train"
    csv_file = os.path.join(data_dir, "train_labels.csv")
    df = pd.read_csv(csv_file)
    dataset = PoseDateset(df)
    
    # get test entry
    poses, label = dataset[123]
    print(poses.shape, label)

def test_clean_poses():
    data_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses/train"
    csv_file = os.path.join(data_dir, "train_labels.csv")
    df = pd.read_csv(csv_file)
    dataset = PoseDateset(df)
    
    # get test entry
    poses, label = dataset[123]
    print(poses.shape, label)

    # clean poses
    cleaned_poses = clean_poses(poses)
    print(cleaned_poses.shape)

if __name__ == "__main__":
    test_pose_dataset()
    test_clean_poses()