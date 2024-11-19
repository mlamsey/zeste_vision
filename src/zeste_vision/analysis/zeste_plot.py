import json
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt

from zeste_vision.data_tools.zeste_loader import EXERCISES, ARMS, USER_RANGES

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def rotation_matrix_y(theta):
    """
    Returns a 3x3 rotation matrix for a rotation around the y-axis by angle theta (in radians).
    
    Parameters:
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: A 3x3 rotation matrix.
    """
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def load_poses(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        data = json.load(f)
        return data

def test_animate_poses(full_data):
    test_exercise = EXERCISES.SEATED_FORWARD_KICK
    exercise_data = full_data[test_exercise.name]
    
    sets = ["set0", "set1", "set2", "set3"]

    for set_name in sets:
        f = plt.figure()
        ax = f.add_subplot(121, projection='3d')
        ax_2d = f.add_subplot(122)
        scatter = ax.scatter([], [], [])
        # Store line plots for each connection
        lines = [ax.plot([], [], [], 'k-')[0] for _ in POSE_CONNECTIONS]
        lines_2d = [ax_2d.plot([], [], 'k-')[0] for _ in POSE_CONNECTIONS]

        ax.set_title(f"{test_exercise.name} - {set_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        ax.set_zlabel("Z")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax_2d.set_title(f"{test_exercise.name} - {set_name}")
        ax_2d.set_xlabel("X")
        ax_2d.set_ylabel("Y")

        ax_2d.set_xlim(-1, 1)
        ax_2d.set_ylim(-1, 1)
        
        set_data = exercise_data[set_name]
        poses = np.array(set_data)
        
        for pose in poses:
            pose_rotated = pose @ rotation_matrix_y(np.pi)
            x = pose_rotated[:, 0]
            y = pose_rotated[:, 1]
            z = pose_rotated[:, 2]

            x_2d = -pose[:, 0]
            y_2d = -pose[:, 1]

            # Update scatter plot points
            scatter._offsets3d = (x, y, z)

            # Update existing line plots
            for i, connection in enumerate(POSE_CONNECTIONS):
                start, end = connection
                lines[i].set_data([x[start], x[end]], [y[start], y[end]])
                lines[i].set_3d_properties([z[start], z[end]])

                lines_2d[i].set_data([x_2d[start], x_2d[end]], [y_2d[start], y_2d[end]])

            plt.draw()
            plt.pause(0.05)
    
def plot_pose_xy(full_data: dict):
    test_exercise = EXERCISES.STANDING_HIGH_KNEE
    exercise_data = full_data[test_exercise.name]
    
    set_name = "set2"
    set_poses = np.array(exercise_data[set_name])

    x = set_poses[:, :, 0]
    y = set_poses[:, :, 1]
    z = set_poses[:, :, 2]

    n_joints = x.shape[1]
    n_t = x.shape[0]
    t = np.arange(n_t)
    
    # f = plt.figure()
    # axes = f.subplots(n_joints, 1, sharex=True)
    # for i in range(n_joints):
    #     axes[i].plot(t, x[:, i], label='x')
    #     axes[i].plot(t, y[:, i], label='y')

    # plt.show()

    joints_of_interest = [
        25, # left knee
        26, # right knee
        27, # left ankle
        28, # right ankle
    ]

    joint_names = [
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
    ]

    f, axes = plt.subplots(len(joints_of_interest), 1, sharex=True, figsize=(6, 10))

    for i, joint in enumerate(joints_of_interest):
        x_plot = x[:, joint] - x[0, joint]
        y_plot = y[:, joint] - y[0, joint]
        z_plot = z[:, joint] - z[0, joint]

        hi = max(x_plot.max(), y_plot.max(), z_plot.max())
        lo = min(x_plot.min(), y_plot.min(), z_plot.min())

        # round ylim to nearest 0.1
        hi = np.ceil(hi * 10) / 10 + 0.1
        lo = np.floor(lo * 10) / 10 - 0.1

        axes[i].plot(t, x_plot, label='x')
        axes[i].plot(t, y_plot, label='y')
        axes[i].plot(t, z_plot, label='z')
        axes[i].set_title(f"Joint {joint_names[i]}")
        axes[i].legend(loc='upper right')
        axes[i].set_ylim(lo, hi)

        # grid
        axes[i].grid(True)

        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Position")

    plt.tight_layout()
    plt.show()

def _get_subshape(arr: list):
    dim1 = len(arr)
    dim2 = [len(a) if a is not None else -1 for a in arr]
    n_missing = dim2.count(-1)
    dim2_unique = list(set(dim2))

    if len(dim2_unique) == 0:
        dim2_unique = [-1]

    if len(dim2_unique) > 1:
        print(f"Array has multiple unique dimensions: {dim1} x {dim2_unique} with {n_missing} missing values.")
        for i in range(dim1):
            a = arr[i]
            if a is None:
                print(i, a)
            #     print("None")
            # else:
            #     shape = len(a)
            #     print(shape)
        input()
    else:
        print(f"Array has dimensions: {dim1} x {dim2_unique[0]}")


def test():
    # trying to see which exercises have missing joints
    path = "/home/lamsey/hrl/zeste_vision/data/zeste"
    json_files = os.listdir(path)
    json_files = [f for f in json_files if f.endswith(".json")]
    json_files.sort()

    for file in json_files:
        print(f"Loading file: {file}")
        data = load_poses(os.path.join(path, file))
        for exercise in EXERCISES:
            for set_i in ["set0", "set1", "set2", "set3"]:
                print(f"Exercise: {exercise.name}, Set: {set_i}", end=" ")
                _get_subshape(data[exercise.name][set_i])
                
                # ex_set_data = data[exercise.name][set_i]
                # try:
                #     data_np = np.array(ex_set_data)
                # except ValueError as e:
                #     print(f"Error loading data for {exercise.name}, {set_i}: {e}")
                #     continue
                # print(f"Exercise: {exercise.name}, Set: {set_i}, Data shape: {data_np.shape}")

def test2():
    path = "/home/lamsey/hrl/zeste_vision/data/zeste"
    file = os.path.join(path, "zst101.json")
    data = load_poses(file)
    ex_s_data = data[EXERCISES.STANDING_REACH_ACROSS.name]["set0"][136:139]
    print(json.dumps(ex_s_data, indent=2))

def main(args):
    if args.plot:
        data = load_poses(args.file_path)
        plot_pose_xy(data)
    if args.animate:
        data = load_poses(args.file_path)
        test_animate_poses(data)
    if args.test:
        test2()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load poses from a file.')
    parser.add_argument('--file_path', type=str, help='Path to the file containing poses.')
    parser.add_argument('--animate', action='store_true', help='Animate poses.')
    parser.add_argument('--plot', action='store_true', help='Plot poses.')
    parser.add_argument('--test', action='store_true', help='Test loading poses.')
    args = parser.parse_args()

    main(args)
