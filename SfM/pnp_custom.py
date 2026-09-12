import numpy as np

def _pnp_dlt(pts3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray):
    """Simple PnP via DLT on normalized camera coords, then Procrustes (EPnP-like).
    Not numerically optimal but adequate with RANSAC.
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (pts2d[:,0]-cx)/fx
    y = (pts2d[:,1]-cy)/fy
    # Solve for pose using Umeyama on 3D-3D (bearing * depth). Approximate depths with z from linear LS.
    # Build A z ≈ 1 to estimate scale per correspondence; fallback: center and align using SVD.
    X = pts3d.astype(np.float64)
    # Center
    Xc = X - X.mean(axis=0)
    bc = np.stack([x, y, np.ones_like(x)], axis=1)
    bc /= np.linalg.norm(bc, axis=1, keepdims=True)
    bc = bc - bc.mean(axis=0)
    H = Xc.T @ bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = (S.sum()) / (np.sum(Xc**2))
    t = bc.mean(axis=0) - s * (R @ X.mean(axis=0))
    # return in world2cam convention: X_c = R X_w + t
    return R.astype(np.float64), t.reshape(3,1).astype(np.float64)

def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X3: np.ndarray):
    Xc = (R @ X3.T + t).T
    uv = (Xc[:, :2] / (Xc[:, 2:3] + 1e-12))
    return (K[:2,:2] @ uv.T + K[:2,2:3]).T

def pnp_ransac_solve(pts3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray,
                     reproj_thresh: float = 3.0, max_iters: int = 3000, confidence: float = 0.999):
    """Minimal custom PnP RANSAC wrapper using DLT+SVD pose and reprojection scoring.
    Returns: ok, R(3x3), t(3x1), inlier_mask(bool N)
    """
    n = pts3d.shape[0]
    if n < 6:
        return False, None, None, None
    rng = np.random.default_rng()
    best_inliers = None
    best_count = -1
    for _ in range(max_iters):
        # use minimal, stable subset
        idx = rng.choice(n, size=min(6, n), replace=False)
        try:
            R, t = _pnp_dlt(pts3d[idx], pts2d[idx], K)
        except Exception:
            continue
        proj = _project(K, R, t, pts3d)
        errs = np.linalg.norm(proj - pts2d, axis=1)
        Z = (R @ pts3d.T + t).T[:, 2]
        inliers = (errs < reproj_thresh) & (Z > 0)
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            # no premature early-stop; keep best across iterations
    if best_inliers is None or best_count < 6:
        return False, None, None, None
    # Refit on inliers
    R, t = _pnp_dlt(pts3d[best_inliers], pts2d[best_inliers], K)
    # final inlier recompute
    proj = _project(K, R, t, pts3d)
    errs = np.linalg.norm(proj - pts2d, axis=1)
    Z = (R @ pts3d.T + t).T[:, 2]
    inliers = (errs < reproj_thresh) & (Z > 0)
    return True, R, t, inliers


