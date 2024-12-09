import cv2
import os
import pandas as pd
from tqdm import tqdm

def split_video(file_path: str, output_file: str, frames_per_video: int):
    # print(f'Splitting {file_path} into {frames_per_video} frames per video')
    # print(f'Writing to {output_file}')
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f'Error: Cannot open file {file_path}')
        return

    if output_file.endswith('.mp4'):
        output_file = output_file[:-4]

    video_i = 0
    frame_i = 0
    out = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % frames_per_video == 0:
            if video_i > 0:
                if out:
                    out.release()
            out = cv2.VideoWriter(f'{output_file}_{video_i:02d}.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (frame.shape[1], frame.shape[0])
                )
            video_i += 1

        out.write(frame)
        frame_i += 1

    cap.release()

def get_video_frame_lengths(dir_path: str):
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.mp4')]
    files.sort()

    frame_lengths = []
    for file in files:
        cap = cv2.VideoCapture(os.path.join(dir_path, file))
        frame_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    return frame_lengths

def bulk_split(top_dir: str, frames_per_video: int, output_top_dir: str = None, bool_ask: bool = True):
    if output_top_dir is None:
        output_top_dir = top_dir

    if bool_ask:
        print(f"Splitting videos in {top_dir} into {frames_per_video} frames per video")
        print(f"Writing to {output_top_dir}")
        ui = input('Continue? (y/n): ').lower()
        if ui != 'y':
            return

    for root, dirs, _ in os.walk(top_dir):
        for d in tqdm(dirs):
            files = os.listdir(os.path.join(root, d))
            files = [f for f in files if f.endswith('.mp4')]
            for file in files:
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, d, file)
                    # print(file_path)
                    output_local_dir = os.path.join(output_top_dir, d)
                    if not os.path.exists(output_local_dir):
                        os.makedirs(output_local_dir)
                    output_file = os.path.join(output_local_dir, file[:-4])
                    # print(output_file)
                    split_video(file_path=file_path, output_file=output_file, frames_per_video=frames_per_video)

def _test(top_dir: str):
    dir_103 = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/videos/train/zst103"
    files = os.listdir(dir_103)
    files = [f for f in files if f.endswith('.mp4')]
    files.sort()

    test_file = files[0]
    file_path = os.path.join(dir_103, test_file)

    split_video(file_path=file_path, output_file='test/test_split', frames_per_video=30)
    frame_lengths = get_video_frame_lengths('test')
    print(frame_lengths)

def do_split(frames_per_video: int = 30, output_top_dir: str = None):
    top_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/videos"
    train_dir = os.path.join(top_dir, 'train')
    eval_dir = os.path.join(top_dir, 'eval')
    for d in [train_dir, eval_dir]:
        if d == train_dir:
            output_top_dir_split = os.path.join(output_top_dir, 'train')
        else:
            output_top_dir_split = os.path.join(output_top_dir, 'eval')

        bulk_split(d, frames_per_video, output_top_dir=output_top_dir_split)

def split_by_exercise(in_file: str):
    # read csv
    df = pd.read_csv(in_file)
    exercises = [
        "seated_reach_forward_low",
        "seated_forward_kick",
        "seated_calf_raise",
        "standing_reach_across",
        "standing_high_knee",
        "standing_windmill",
    ]

    file_root = in_file.split('.')[0]

    for exercise in exercises:
        exercise_df = df[df["File Path"].str.contains(exercise)]
        print(exercise_df.shape)
        exercise_df.to_csv(f'{file_root}_{exercise}.csv', index=False)

def cleanup(bool_ask: bool = False):
    print('Cleaning up...')
    top_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/videos"
    train_dir = os.path.join(top_dir, 'train')
    eval_dir = os.path.join(top_dir, 'eval')
    n_erased = 0
    for d in [train_dir, eval_dir]:
        subjects = os.listdir(d)
        for subject in subjects:
            subject_dir = os.path.join(d, subject)
            files = os.listdir(subject_dir)
            files = [f for f in files if f.endswith('.mp4')]
            for file in files:
                if not file.endswith('rgb.mp4'):
                    if bool_ask:
                        ui = input(f'Delete {file}? (y/n): ').lower()
                        if ui == 'y':
                            os.remove(os.path.join(subject_dir, file))
                            n_erased += 1
                    else:
                        os.remove(os.path.join(subject_dir, file))
                        n_erased += 1

    print(f'Erased {n_erased} files')

def main(args):
    if args.test:
        _test(args.top_dir)
    elif args.split:
        do_split(args.frames_per_video, args.output_top_dir)
    elif args.cleanup:
        cleanup()
    elif args.split_by_exercise:
        split_by_exercise(args.file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--frames_per_video', type=int, default=30)
    parser.add_argument('--output_top_dir', type=str)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--split_by_exercise', action='store_true')
    parser.add_argument('--file', type=str)

    args = parser.parse_args()
    main(args)
