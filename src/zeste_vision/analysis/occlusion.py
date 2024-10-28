import cv2
import mediapipe as mp
import numpy as np

from tqdm import tqdm

from zeste_vision.data_tools.loader import get_test_video_frames

def test():
    frames = get_test_video_frames()

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
