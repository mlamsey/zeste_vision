import os
import mediapipe
import cv2

def _get_test_video_path():
    data_dir = "/data/aist_dance"
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".mp4")]
    files.sort()
    return os.path.join(data_dir, files[0])

def _get_test_video_frames():
    video_file = _get_test_video_path()
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def test_load_video():
    print("Testing video loading...")
    video_file = _get_test_video_path()
    print("Video file:", video_file)

    # open video file for viewing
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    print("Number of frames:", len(frames))

def test_mediapipe():
    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose()
    print("Pose model loaded.")
    frames = _get_test_video_frames()
    print("Number of frames:", len(frames))

    # run mediapipe on frames
    missing = [0] * 33
    for frame in frames:
        results = pose.process(frame)
        # print(results.pose_landmarks)

        # overlay
        # image = frame.copy()
        # mp_drawing = mediapipe.solutions.drawing_utils
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow("MediaPipe Pose", image)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

        # get missing keypoints
        missing_keypoints = get_missing_keypoints(results)
        for i in missing_keypoints:
            missing[i] += 1

    print("Missing keypoints:")
    for i in range(33):
        print(i, missing[i])

    pose.close()

def get_missing_keypoints(results):
    n_landmarks = 33
    landmarks = results.pose_landmarks.landmark
    is_visible = [landmarks[i].visibility for i in range(n_landmarks)]
    missing = [i for i in range(n_landmarks) if is_visible[i] < 0.5]
    return missing

def main(args):
    test_mediapipe()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
