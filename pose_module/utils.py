# pose_module/utils.py

import numpy as np

def normalize_keypoints(keypoints):
    """
    Normalize 3D keypoints to have zero mean and unit variance for each axis.
    Input: List of [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]
    Output: Normalized numpy array of shape (99,)
    """
    keypoints = np.array(keypoints).reshape(-1, 3)  # shape: (33, 3)
    mean = np.mean(keypoints, axis=0)
    std = np.std(keypoints, axis=0) + 1e-6  # avoid divide by zero
    normalized = (keypoints - mean) / std
    return normalized.flatten().tolist()
