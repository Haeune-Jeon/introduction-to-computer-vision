import numpy as np
import cv2 as cv

try:
    from scipy.optimize import least_squares
    from scipy import sparse
except Exception:
    least_squares = None
    sparse = None

def _rod_from_R(R):
    r, _ = cv.Rodrigues(R.astype(np.float64)); return r.reshape(3)

def _R_from_rod(r):
    R, _ = cv.Rodrigues(r.reshape(3,1)); return R

def _project(K, R, t, X):  # X: (N,3)
    Xc = (R @ X.T + t.reshape(3,1)).T
    uv = (K[:2,:2] @ (Xc[:,:2] / (Xc[:,2:3] + 1e-12)).T + K[:2,2:3]).T
    return uv, Xc[:,2]

def _K_from_params(kparams):
    fx, fy, cx, cy = kparams
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def global_bundle_adjustment(
    pipeline, K_init, *,
    huber_delta=3.0,
    max_nfev=80,
    fix_anchor=True,    
    fix_point=True,      
    gate_px=None,         
    max_residuals=None,   
    optimize_K = True,
    K_prior_weight=1e-3
):
    
    if least_squares is None:
        print("[BA] SciPy: pip install scipy")
        return 0, 0

    if not hasattr(pipeline, 'points_3d') or len(pipeline.points_3d) == 0:
        print("[BA] no 3D points.")
        return 0, 0
    
    # K initialize, parameteraize
    if optimize_K:
        kparams0 = np.array([K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2]], dtype=np.float64)
    else:
        kparams0 = None
    K0 = K_init.astype(np.float64)

    # 1) registered camera
    win_views = [i for i, cam in enumerate(pipeline.cameras) if cam is not None and cam[0] is not None]
    win_views = sorted(set(win_views))
    if len(win_views) < 2:
        print("[BA] not enough cameras.")
        return 0, 0

    # 2) collect observation
    win_points = set()
    obs = []  # (vid, pid, u, v)
    active_set = set(win_views)
    for v, pid, u, vpix in pipeline.observations:
        if v in active_set:
            win_points.add(pid)
            obs.append((v, pid, float(u), float(vpix)))
    if len(win_points) == 0 or len(obs) < 8: 
        print("[BA] not enough observations in window.")
        return 0, 0

    # 3) indexation
    cam_order = sorted(win_views)
    pt_order  = sorted(win_points)
    vid2i = {v: i for i, v in enumerate(cam_order)}
    pid2i = {p: i for i, p in enumerate(pt_order)}

    K0 = K_init.astype(np.float64)

    # 4) initial parameter
    cam_r = np.zeros((len(cam_order), 3), dtype=np.float64)
    cam_t = np.zeros((len(cam_order), 3), dtype=np.float64)
    for i, v in enumerate(cam_order):
        R, t = pipeline.cameras[v]
        cam_r[i, :] = _rod_from_R(R)
        cam_t[i, :] = t.reshape(3)

    P_all = pipeline.points_3d[np.array(pt_order, dtype=int)].astype(np.float64)  # (P,3)

    # 5) make obs array
    o_iv, o_ip, o_u, o_v = [], [], [], []
    for v, p, u, vpix in obs:
        iv = vid2i.get(v); ip = pid2i.get(p)
        if iv is not None and ip is not None:
            o_iv.append(iv); o_ip.append(ip); o_u.append(u); o_v.append(vpix)
    if len(o_iv) < 8:
        print("[BA] not enough valid obs after indexing.")
        return 0, 0

    o_iv = np.asarray(o_iv, dtype=np.int32)
    o_ip = np.asarray(o_ip, dtype=np.int32)
    o_u  = np.asarray(o_u,  dtype=np.float64)
    o_v  = np.asarray(o_v,  dtype=np.float64)
    
    pts_in_obs = sorted(set(o_ip.tolist()))
    old2new = {old:i for i,old in enumerate(pts_in_obs)}
    P_all = P_all[pts_in_obs]                 
    pt_order = [pt_order[i] for i in pts_in_obs]
    o_ip = np.array([old2new[i] for i in o_ip], dtype=np.int32)

    # 6) gating
    if gate_px is not None:
        mask_keep = np.zeros(o_iv.shape[0], dtype=bool)
        for iv in np.unique(o_iv):
            sel = (o_iv == iv)
            pts = P_all[o_ip[sel]]
            R = _R_from_rod(cam_r[iv]); t = cam_t[iv]
            uv_hat, z = _project(K0, R, t, pts)
            du = uv_hat[:, 0] - o_u[sel]
            dv = uv_hat[:, 1] - o_v[sel]
            err = np.sqrt(du*du + dv*dv)
            keep = (z > 0) & (err <= gate_px)
            mask_keep[sel] = keep
        if mask_keep.sum() < 8:
            print(f"[BA] after gating, only {mask_keep.sum()} residuals. Skipping BA.")
            return 0, 0
        if max_residuals is not None and mask_keep.sum() > max_residuals:
            rng = np.random.default_rng(0)
            kept_idx = np.where(mask_keep)[0]
            pick = rng.choice(kept_idx, size=max_residuals, replace=False)
            mask_keep[:] = False; mask_keep[pick] = True
        o_iv = o_iv[mask_keep]; o_ip = o_ip[mask_keep]
        o_u  = o_u [mask_keep]; o_v  = o_v [mask_keep]

    # 7) cam0 + 1 point
    fixed = set()
    if fix_anchor and 0 in cam_order:
        fixed.add(cam_order.index(0))  # fix cam0

    fixed_point_i = None
    if fix_point:
        fixed_point_i = 0  

    offset = 0

    # K 
    if optimize_K:
        k_block_offset = 0
        offset += 4          # [fx, fy, cx, cy]
    else:
        k_block_offset = None

    # camera
    cam_block_index = {}     
    for i_cam in range(len(cam_order)):
        if i_cam in fixed:
            cam_block_index[i_cam] = None
        else:
            cam_block_index[i_cam] = offset
            offset += 6       # rvec(3) + tvec(3)

    n_cam_vars = offset - (4 if optimize_K else 0)

    # Point
    pt_var_index = np.full(P_all.shape[0], -1, dtype=np.int32)  
    var_cnt = 0
    for ip_local in range(P_all.shape[0]):
        if fixed_point_i is not None and ip_local == fixed_point_i:
            continue
        pt_var_index[ip_local] = var_cnt
        var_cnt += 1
    pt_block_offset = offset
    n_pt_vars = 3 * var_cnt
    offset += n_pt_vars

    n_vars = offset

    def pack_vars(kparams, cr, ct, P):
        chunks = []
        if optimize_K:
            chunks.append(kparams.reshape(-1))  # [fx, fy, cx, cy]
        for i in range(len(cam_order)):
            if i in fixed:
                continue
            chunks.append(cr[i]); chunks.append(ct[i])
        mask = np.ones(P.shape[0], dtype=bool)
        if fixed_point_i is not None:
            mask[fixed_point_i] = False
        chunks.append(P[mask].reshape(-1))
        return np.concatenate([c.reshape(-1) for c in chunks], axis=0)

    def unpack_vars(x, kparams_base, cr_base, ct_base, P_base):
        cr = cr_base.copy(); ct = ct_base.copy(); P = P_base.copy()
        i = 0
        if optimize_K:
            kparams = x[i:i+4]; i += 4
        else:
            kparams = kparams_base
        for k in range(len(cam_order)):
            if k in fixed:
                continue
            cr[k, :] = x[i:i+3]; i += 3
            ct[k, :] = x[i:i+3]; i += 3
        mask = np.ones(P.shape[0], dtype=bool)
        if fixed_point_i is not None:
            mask[fixed_point_i] = False
        P[mask] = x[i:].reshape((-1, 3))
        return kparams, cr, ct, P

    x0 = pack_vars(kparams0, cam_r, cam_t, P_all)
    
    # K optimize
    if optimize_K:
        lb = [1e-3, 1e-3, -1e4, -1e4]   # fx, fy, cx, cy
        ub = [1e6,  1e6,   1e4,  1e4]
    else:
        lb = []; ub = []
    # camera
    for i_cam in range(len(cam_order)):
        if i_cam in fixed:
            continue
        lb += [-np.inf]*3 + [-np.inf]*3   # rvec, tvec
        ub += [ np.inf]*3 + [ np.inf]*3
    
    lb += [-np.inf] * n_pt_vars
    ub += [ np.inf] * n_pt_vars

    bounds = (np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64))
    assert bounds[0].shape[0] == n_vars and bounds[1].shape[0] == n_vars

    # 8) residual
    M = o_iv.shape[0]
    do_kprior = (optimize_K and K_prior_weight > 0.0)
    def residual(x):
        kparams, cr, ct, P = unpack_vars(x, kparams0, cam_r, cam_t, P_all)
        K_use = _K_from_params(kparams) if optimize_K else K_init
        res = np.empty(2*M + (4 if do_kprior else 0), dtype=np.float64)
        for iv in np.unique(o_iv):
            idxs = np.where(o_iv == iv)[0]
            pts  = P[o_ip[idxs]]
            R = _R_from_rod(cr[iv]); t = ct[iv]
            uv_hat, _ = _project(K_use, R, t, pts)
            res[2*idxs]   = uv_hat[:, 0] - o_u[idxs]
            res[2*idxs+1] = uv_hat[:, 1] - o_v[idxs]
        
        if do_kprior:
            kvec0 = np.array([K0[0,0], K0[1,1], K0[0,2], K0[1,2]], dtype=np.float64)
            res[2*M:2*M+4] = np.sqrt(K_prior_weight) * (kparams - kvec0)
        return res

    rows, cols = [], []
    for r in range(M):
        iv = o_iv[r]; ip = o_ip[r]
        ru = 2*r; rv = 2*r+1
        
        # (a) K 
        if optimize_K:
            for j in range(4):
                cols += [k_block_offset + j, k_block_offset + j]
                rows += [ru, rv]
        # (b) camera
        cb = cam_block_index[iv]
        if cb is not None:
            for j in range(6):
                cols += [cb+j, cb+j]; rows += [ru, rv]
        # (c) point block
        pvi = pt_var_index[ip]
        if pvi >= 0:  
            base = pt_block_offset + 3*pvi
            cols += [base+0, base+1, base+2, base+0, base+1, base+2]
            rows += [ru, ru, ru, rv, rv, rv]
    
    if do_kprior:
        base_row = 2*M
        for j in range(4):
            rows.append(base_row + j)
            cols.append(k_block_offset + j)  # fx, fy, cx, cy

    jac_sparsity = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(2*M + (4 if do_kprior else 0), n_vars)
    )
    
    # 9) optimization
    sol = least_squares(
        residual, x0,
        method='trf',
        loss='huber', f_scale=huber_delta,
        max_nfev=max_nfev,
        jac_sparsity=jac_sparsity,
        bounds=bounds
    )

    # 10) apply
    kparams_opt, cr_opt, ct_opt, P_opt = unpack_vars(sol.x, kparams0, cam_r, cam_t, P_all)
    K_opt = _K_from_params(kparams_opt) if optimize_K else K_init
    
    for i, v in enumerate(cam_order):
        R = _R_from_rod(cr_opt[i]); t = ct_opt[i].reshape(3, 1)
        pipeline.cameras[v] = (R.astype(np.float32), t.astype(np.float32))
    pipeline.points_3d[np.array(pt_order, dtype=int)] = P_opt.astype(np.float32)

    print("[BA] Global BA complete. success:", sol.success, " cost:", sol.cost)
    if optimize_K:
        print(f"[BA] K updated → fx={K_opt[0,0]:.3f}, fy={K_opt[1,1]:.3f}, cx={K_opt[0,2]:.3f}, cy={K_opt[1,2]:.3f}")
    return len(cam_order), len(pt_order), (K_opt if optimize_K else K_init)
