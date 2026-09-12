# ----- add: COLMAP TXT parsers & pose conversion -----
from typing import List, Dict
import numpy as np
from pathlib import Path
from typing import Tuple

def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)],
    ], dtype=float)
    return R

def load_colmap_intrinsics(model_dir: Path) -> Dict[int, dict]:
    """
    cameras.txt -> {camera_id: {'model':str, 'w':int, 'h':int, 'params':list, 'K':3x3}}
    (Ignore distortion coefficients and construct K only)
    """
    cams = {}
    path = Path(model_dir) / "cameras.txt"
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"): 
                continue
            # id, MODEL, width, height, params...
            toks = ln.strip().split()
            cam_id = int(toks[0]); model = toks[1]
            w, h = int(toks[2]), int(toks[3])
            params = list(map(float, toks[4:]))

            # Handle only some representative models (add others as needed)
            if model in ("PINHOLE", "OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
            elif model in ("SIMPLE_RADIAL", "RADIAL"):
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
            else:
                # fallback: assume center and f≈w roughly
                fx = fy = max(w, h)
                cx, cy = w/2.0, h/2.0

            K = np.array([[fx, 0,  cx],
                          [0,  fy, cy],
                          [0,  0,  1.0]], dtype=float)
            cams[cam_id] = dict(model=model, w=w, h=h, params=params, K=K)
    return cams

def load_colmap_poses_cam2world(model_dir: Path) -> List[Tuple[np.ndarray, np.ndarray, str, int]]:
    """
    images.txt -> cam2world (Rcw, tcw, name, camera_id) list (sorted by filename)
    COLMAP's q,t are world2cam:  X_c = R X_w + t
      → cam2world: Rcw = R^T, C = -R^T t, tcw = C
    """
    path = Path(model_dir) / "images.txt"
    recs = []
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            toks = ln.strip().split()
            if len(toks) < 9:
                continue
            image_id = int(toks[0])
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8])
            name = toks[9]
            # Skip 2D observations in next line
            # (file has 2D keypoint/track line following each image line)

            R_wc = _quat_to_rot(qw, qx, qy, qz)    # world->cam
            t_wc = np.array([tx, ty, tz], dtype=float)
            Rcw = R_wc.T
            C   = -R_wc.T @ t_wc
            tcw = C
            recs.append((Rcw, tcw, name, cam_id))

            # skip the next line (2D observations)
            _ = next(f, None)

    # Sort by filename (generally matches video frame order)
    recs.sort(key=lambda x: x[2])
    return recs
