import json
import mediapipe as mp
import numpy as np
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
    


def main(args):
    data = load_poses(args.file_path)
    test_animate_poses(data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load poses from a file.')
    parser.add_argument('--file_path', type=str, help='Path to the file containing poses.')
    args = parser.parse_args()

    main(args)
