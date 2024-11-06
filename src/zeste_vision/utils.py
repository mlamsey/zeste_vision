import numpy as np

def mpjpe(pose1, pose2):
    """
    Compute the mean per-joint position error (MPJPE) between two poses.
    The poses should be numpy arrays with shape (n_joints, 3).
    """
    if pose1 is None or pose2 is None:
        return None
    
    return np.mean(np.linalg.norm(pose1 - pose2, axis=1))
