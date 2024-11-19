import numpy as np

def get_pose_np(pose_landmarks):
    if pose_landmarks is None:
        return None
    
    return np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in pose_landmarks.landmark])

def mpjpe(pose1, pose2):
    """
    Compute the mean per-joint position error (MPJPE) between two poses.
    The poses should be numpy arrays with shape (n_joints, 3).
    """
    if pose1 is None or pose2 is None:
        return None
    
    return np.mean(np.linalg.norm(pose1 - pose2, axis=1))
