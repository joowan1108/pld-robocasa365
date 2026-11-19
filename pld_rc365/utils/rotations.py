import numpy as np


def quat_xyzw_to_axis_angle(quaternion) -> np.ndarray:
    """Identical to Xiaomi-Robotics-1 eval_robocasa365/entry.py (state construction for XR1)."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[3] < 0:
        q = -q
    xyz = q[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)
