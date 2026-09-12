"""
Utility functions for file operations and data handling.
"""
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
import numpy as np
import csv

def ensure_dir(p: Path) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:p: Path to directory
    Returns:Path: The directory path
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


def imwrite(path: Path, img) -> None:
    """
    Save image to file, creating parent directories if needed.
    
    Args:
        path: Output file path
        img: Image array to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), img)


def pick_first_video(data_dir: Path) -> Path:
    exts = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"]
    cands: List[Path] = []
    for e in exts:
        cands += list(data_dir.glob(e))
    if not cands:
        raise FileNotFoundError(
            f"No video found under {data_dir}. Put a video in ./data/ (mp4/mov/avi/mkv)."
        )
    return cands[0]


def _get_cam_rt(cam):
    def arr(x): return np.asarray(x, dtype=np.float64)

    # (R,t) tuple/list
    if isinstance(cam, (tuple, list)) and len(cam) == 2:
        return arr(cam[0]), arr(cam[1]).ravel()

    # numpy 3x4/4x4
    if isinstance(cam, np.ndarray):
        if cam.shape == (3, 4):
            return arr(cam[:, :3]), arr(cam[:, 3]).ravel()
        if cam.shape == (4, 4):
            return arr(cam[:3, :3]), arr(cam[:3, 3]).ravel()

    # dict/object
    def get(a, *names):
        if isinstance(a, dict):
            for n in names:
                if n in a: return a[n]
        else:
            for n in names:
                if hasattr(a, n): return getattr(a, n)
        return None

    # P (3x4)
    P = get(cam, 'P')
    if P is not None:
        P = arr(P)
        if P.shape == (3, 4):
            return arr(P[:, :3]), arr(P[:, 3]).ravel()

    # T (4x4)
    T = get(cam, 'T', 'Pose', 'pose')
    if T is not None:
        T = arr(T)
        if T.shape == (4, 4):
            return arr(T[:3, :3]), arr(T[:3, 3]).ravel()

    # R,t
    R = get(cam, 'R', 'Rcw')
    t = get(cam, 't', 'tcw')
    if R is not None and t is not None:
        return arr(R), arr(t).ravel()

    # Rwc, twc (camera->world)
    Rwc = get(cam, 'Rwc')
    twc = get(cam, 'twc')
    if Rwc is not None and twc is not None:
        Rwc = arr(Rwc); twc = arr(twc).ravel()
        Rcw = Rwc.T
        tcw = -Rcw @ twc
        return Rcw, tcw

    # R, C(camera center in world) -> t = -R @ C
    R_only = get(cam, 'R')
    C = get(cam, 'C', 'center', 'camera_center')
    if R_only is not None and C is not None:
        R_only = arr(R_only); C = arr(C).ravel()
        return R_only, (-R_only @ C)

    raise ValueError("Unsupported camera representation for _get_cam_rt().")


def _project_point(K, R, t, X):
    
    X = np.asarray(X, dtype=np.float64).ravel()
    Xc = R @ X + t
    z = Xc[2] if Xc[2] != 0 else 1e-9
    x = Xc[:2] / z
    uvw = K @ np.array([x[0], x[1], 1.0], dtype=np.float64)
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


def compute_observation_errors(cameras, points_3d, obs4, K):
    errs = np.zeros(len(obs4), dtype=np.float64)
    for i, (v, p, u, w) in enumerate(obs4):
        R, t = _get_cam_rt(cameras[v])
        uh, vh = _project_point(K, R, t, points_3d[p])
        errs[i] = np.hypot(uh - u, vh - w)
    return errs


def aggregate_colmap_point_errors(n_points: int, obs4, obs_errs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    per_point = [[] for _ in range(n_points)]
    for (v, p, u, w), e in zip(obs4, obs_errs):
        if 0 <= p < n_points and np.isfinite(e):
            per_point[p].append(float(e))

    per_point_mean   = np.full(n_points, np.nan, dtype=np.float64)
    per_point_median = np.full(n_points, np.nan, dtype=np.float64)
    per_point_count  = np.zeros(n_points, dtype=np.int32)
    for i, lst in enumerate(per_point):
        if lst:
            per_point_count[i]  = len(lst)
            per_point_mean[i]   = float(np.mean(lst))
            per_point_median[i] = float(np.median(lst))

    valid = np.isfinite(per_point_mean)
    arr = per_point_mean[valid] if valid.any() else np.array([], dtype=np.float64)
    if len(arr) == 0:
        summary = {'count': 0, 'mean': np.nan, 'median': np.nan, 'rmse': np.nan, 'p90': np.nan, 'max': np.nan}
    else:
        summary = {
            'count' : int(len(arr)),
            'mean'  : float(np.mean(arr)),
            'median': float(np.median(arr)),
            'rmse'  : float(np.sqrt(np.mean(arr**2))),
            'p90'   : float(np.percentile(arr, 90)),
            'max'   : float(np.max(arr)),
        }
    return per_point_mean, per_point_median, per_point_count, summary


def save_colmap_point_errors_csv(csv_path, per_point_mean, per_point_median, per_point_count):
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["point_id", "mean_error_px", "median_error_px", "n_obs"])
        for pid, (m, md, c) in enumerate(zip(per_point_mean, per_point_median, per_point_count)):
            wr.writerow([pid, m, md, int(c)])
