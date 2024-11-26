import cv2
import os

def split_video(file_path: str, output_file: str, frames_per_video: int):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f'Error: Cannot open file {file_path}')
        return

    if output_file.endswith('.jpg'):
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

def main(args):
    if args.test:
        _test(args.top_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', type=str)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    main(args)
