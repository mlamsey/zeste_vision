import os
import numpy as np
import mediapipe
import cv2
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

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

#####
def run_pose_estimator_on_frames(frames):
    """
    returns overlaid image
    """
    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    results = []

    for frame in frames:
        # Convert the BGR frame to RGB (MediaPipe expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with the pose estimator
        result = pose.process(rgb_frame)

        # overlay
        image = frame.copy()
        mp_drawing = mediapipe.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        results.append(image)
    
    pose.close()
    return results

def load_frames_to_shared_memory(frames):
    # Convert the frames list to a contiguous array
    frame_array = np.array(frames)
    shm = shared_memory.SharedMemory(create=True, size=frame_array.nbytes)
    shared_frames = np.ndarray(frame_array.shape, dtype=frame_array.dtype, buffer=shm.buf)
    shared_frames[:] = frame_array[:]
    return shm, shared_frames

# Function to divide the frames into chunks
def split_frames(frames, n_chunks):
    # Divide frames into roughly equal chunks for each worker
    return np.array_split(frames, n_chunks)

def split_frame_indices(n_frames, n_chunks):
    return np.array_split(np.arange(n_frames), n_chunks)
    
def get_missing_keypoints(results):
    n_landmarks = 33
    landmarks = results.pose_landmarks.landmark
    is_visible = [landmarks[i].visibility for i in range(n_landmarks)]
    missing = [i for i in range(n_landmarks) if is_visible[i] < 0.5]
    return missing

#####
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

def test_mediapipe(n_workers: int = 1, render: bool = False):
    # mp_pose = mediapipe.solutions.pose
    # pose = mp_pose.Pose(model_complexity=1, num_threads=8)
    # print("Pose model loaded.")
    frames = _get_test_video_frames()
    print("Number of frames:", len(frames))

    # Load frames into shared memory
    frame_chunks = split_frames(frames, n_workers)

    # # Split the frames into chunks for each worker
    # frame_chunks = split_frames(frames, n_workers)
    
    if n_workers == 1:
        flat_results = run_pose_estimator_on_frames(frames)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Execute pose estimation in parallel on different chunks of frames
            all_results = list(executor.map(run_pose_estimator_on_frames, frame_chunks))
        
        # Flatten the list of results (since it will return a list of lists)
        flat_results = [item for sublist in all_results for item in sublist]
    
    if render:
        for frame in flat_results:
            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    return flat_results

    # run mediapipe on frames
    # missing = [0] * 33
    # for frame in tqdm(frames):
    #     results = pose.process(frame)
    #     # print(results.pose_landmarks)

    #     # overlay
    #     # image = frame.copy()
    #     # mp_drawing = mediapipe.solutions.drawing_utils
    #     # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    #     # cv2.imshow("MediaPipe Pose", image)
    #     # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     #     break

    #     # get missing keypoints
    #     missing_keypoints = get_missing_keypoints(results)
    #     for i in missing_keypoints:
    #         missing[i] += 1

    print("Missing keypoints:")
    for i in range(33):
        print(i, missing[i])

    pose.close()

def test_mediapipe_in_batches(batch_size: int = 10, total_frames: int = 100, render: bool = False):
    start_frame = 0
    flat_results = []

    while start_frame < total_frames:
        # Load a batch of frames
        frames, start_frame = _get_test_video_frames(batch_size, start_frame, total_frames)
        print(f"Processing frames {start_frame - batch_size} to {start_frame}")

        # Process the current batch of frames
        batch_results = run_pose_estimator_on_frames(frames)
        flat_results.extend(batch_results)  # Store the results

        # Optional: Render the frames with the results in the main process

    if render:
        cv2.destroyAllWindows()

    return flat_results

def test_mediapipe_threading():
    for n_workers in range(1, 5):
        print("Number of workers:", n_workers)
        start_time = time.time()
        test_mediapipe(n_workers, render=False)
        end_time = time.time()
        print("Elapsed time:", end_time - start_time)

def main(args):
    test_mediapipe(n_workers=1)
    # test_mediapipe_threading()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
