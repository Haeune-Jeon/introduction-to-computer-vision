from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------------------- Math helpers -------------------------

def _proj_KRT(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D points X (N,3) with intrinsics K and pose (R,t). Returns (N,2)."""
    X = np.asarray(X, dtype=np.float32)
    RXt = (R @ X.T) + t.reshape(3, 1)
    x = RXt[:2] / RXt[2:3]
    u = (K[:2, :2] @ x) + K[:2, 2:3]
    return u.T


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = q.astype(np.float64)
    n = qw*qw + qx*qx + qy*qy + qz*qz
    s = 2.0 / n if n > 0 else 0.0
    x, y, z = qx, qy, qz
    w = qw
    R = np.array([
        [1 - s*(y*y + z*z),     s*(x*y - z*w),     s*(x*z + y*w)],
        [    s*(x*y + z*w), 1 - s*(x*x + z*z),     s*(y*z - x*w)],
        [    s*(x*z - y*w),     s*(y*z + x*w), 1 - s*(x*x + y*y)]
    ], dtype=np.float64)
    return R.astype(np.float32)


# ------------------------- Evaluation core -------------------------

def compute_reprojection_report(
    cameras: List[Tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    observations: List[Tuple[int, int, float, float]],
    K: Optional[np.ndarray] = None,
    K_map: Optional[Dict[int, np.ndarray]] = None,
):
    """
    Compute per-observation reprojection errors.

    Args:
        cameras: list indexed by internal view id -> (R(3x3), t(3x1))
        points_3d: (P,3) float32 array
        observations: list of (view_id, point_id, x, y)
        K: shared intrinsics (3x3) for all views (optional if K_map is given)
        K_map: dict view_id -> K (3x3); overrides K when present

    Returns dict with:
        - 'errors': np.ndarray (N,)
        - 'per_cam': {cam_id: {'count','mean','median','rmse'}}
        - 'summary': {'count','mean','median','rmse','p90','max'}
        - 'records': list[(view, pid, x, y, uhat, vhat, err)]
    """
    if K is None and not K_map:
        raise ValueError("Provide either a shared K or a K_map per view.")

    errs: List[float] = []
    records: List[Tuple[int,int,float,float,float,float,float]] = []
    per_cam: Dict[int, Dict[str, List[float]]] = {}

    for (v, p, x, y) in observations:
        if v >= len(cameras) or p >= len(points_3d):
            continue
        R, t = cameras[v]
        K_use = K_map[v] if (K_map and v in K_map) else K
        if K_use is None:
            continue
        pred = _proj_KRT(K_use, R, t, points_3d[p:p+1])[0]
        if not np.all(np.isfinite(pred)):
            continue
        err = float(np.linalg.norm(pred - np.array([x, y], dtype=np.float32)))
        errs.append(err)
        records.append((int(v), int(p), float(x), float(y), float(pred[0]), float(pred[1]), err))
        per_cam.setdefault(int(v), {"vals": []})["vals"].append(err)

    if len(errs) == 0:
        return {
            'errors': np.array([]),
            'per_cam': {},
            'summary': {'count':0, 'mean':np.nan, 'median':np.nan, 'rmse':np.nan, 'p90':np.nan, 'max':np.nan},
            'records': records,
        }

    errs_np = np.asarray(errs, dtype=np.float32)
    summary = {
        'count': int(len(errs_np)),
        'mean':  float(np.mean(errs_np)),
        'median':float(np.median(errs_np)),
        'rmse':  float(np.sqrt(np.mean(errs_np**2))),
        'p90':   float(np.percentile(errs_np, 90.0)),
        'max':   float(np.max(errs_np)),
    }

    per_cam_out: Dict[int, Dict[str, float]] = {}
    for cid, slot in per_cam.items():
        vals = np.asarray(slot['vals'], dtype=np.float32)
        per_cam_out[cid] = {
            'count': int(len(vals)),
            'mean':  float(np.mean(vals)) if len(vals) else np.nan,
            'median':float(np.median(vals)) if len(vals) else np.nan,
            'rmse':  float(np.sqrt(np.mean(vals**2))) if len(vals) else np.nan,
        }

    return {
        'errors': errs_np,
        'per_cam': per_cam_out,
        'summary': summary,
        'records': records,
    }


def save_reprojection_csv(report, out_csv: str | Path):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("view_id,point_id,x,y,uhat,vhat,error_px\n")
        for (v, p, x, y, uhat, vhat, err) in report['records']:
            f.write(f"{v},{p},{x:.6f},{y:.6f},{uhat:.6f},{vhat:.6f},{err:.6f}\n")

# -------------------------- For COLMAP --------------------------
def _summarize(values: np.ndarray, weights: np.ndarray | None = None) -> Dict[str, float]:
    if values.size == 0:
        return {'count': 0, 'mean': np.nan, 'median': np.nan,
                'rmse': np.nan, 'p90': np.nan, 'max': np.nan}
    if weights is None:
        w = None
        mean = float(values.mean())
        rmse = float(np.sqrt(np.mean(values**2)))
    else:
        w = weights.astype(np.float64)
        w /= (w.sum() + 1e-12)
        mean = float(np.sum(w * values))
        rmse = float(np.sqrt(np.sum(w * (values**2))))
    return {
        'count': int(values.size),
        'mean':  mean,
        'median': float(np.median(values)),
        'rmse':  rmse,
        'p90':   float(np.percentile(values, 90.0)),
        'max':   float(np.max(values)),
    }

def load_colmap_point_errors(model_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    read reprojection error and length of track per point from points3D.txt
    Returns:
        errs:  (P,) float64  per-point mean reprojection error (px)
        lens:  (P,) int      per-point track length (#observations)
    """
    model_dir = Path(model_dir)
    f = model_dir / "points3D.txt"
    if not f.exists():
        raise FileNotFoundError(f"points3D.txt not found: {f}")
    errs, lens = [], []
    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if not ln.strip() or ln.startswith("#"):
                continue
            tok = ln.split()
            err = float(tok[7])
            n_pairs = (len(tok) - 8) // 2
            errs.append(err)
            lens.append(n_pairs)
    return np.asarray(errs, dtype=np.float64), np.asarray(lens, dtype=np.int32)

def colmap_error_report(model_dir: str | Path) -> Dict[str, Dict[str, float]]:
    errs, lens = load_colmap_point_errors(model_dir)
    rep_point = _summarize(errs)                       # points
    rep_obs   = _summarize(errs, weights=lens)         # wegithed by number of obs
    return {"point_avg": rep_point, "obs_weighted": rep_obs}
