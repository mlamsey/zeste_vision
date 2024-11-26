import os
import numpy
import torch
import mediapipe
from torch.utils.data import DataLoader
from zeste_vision.data_tools.dataloader import ZesteVisionDataset

def _test_numpy():
    print("Testing numpy:")
    print(numpy.__version__)

def _test_torch():
    print("Testing torch:")
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    print("CUDA device:", torch.cuda.get_device_name(0))

def _test_mediapipe():
    print("Testing mediapipe:")
    print(mediapipe.__version__)
    # human pose estimator
    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose()
    print("Pose model loaded.")

def _test_storage():
    data_dir = "/data"
    print("Testing storage...")
    print(os.listdir(data_dir))

def _test_dataloader():
    print("Testing dataloader...")
    csv_file = "/nethome/mlamsey3/HRL/zeste_vision/labels2.csv"
    video_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/train"
    dataset = ZesteVisionDataset(csv_file, video_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for data in dataloader:
        print(data)

def _test_all():
    print("Testing all...")
    _test_numpy()
    _test_torch()
    _test_mediapipe()
    _test_storage()
    _test_dataloader()

def main(args):
    test_all  = args.test_all
    test_torch = args.test_torch
    test_dataloader = args.test_dataloader

    if test_all:
        _test_all()
        return

    if test_torch:
        _test_torch()
        return

    if test_dataloader:
        _test_dataloader()
        return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_all', action='store_true', default=False)
    parser.add_argument('--test_torch', action='store_true', default=False)
    parser.add_argument('--test_dataloader', action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)