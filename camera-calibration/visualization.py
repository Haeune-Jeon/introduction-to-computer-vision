import os, cv2
import numpy as np

class Visualizer:
    @staticmethod
    def project_points_plane_distort(p_plane, K, R, t, k1, k2, p1=0.0, p2=0.0):
        """
        p_plane: (N,3) [X,Y,1] on Z=0 plane (homogeneous 2D on board)
        K: (3,3) intrinsics with skew=K[0,1] allowed
        R: (3,3), t: (3,) or (3,1)
        k1,k2,p1,p2: distortion (Zhang의 3.3: r^2, r^4 + tangential)
        return: (N,2) pixel coordinates
        """
        XY = p_plane[:, :2]                        # (N,2)
        N  = XY.shape[0]
        objp = np.hstack([XY, np.zeros((N,1))])    # (N,3): [X,Y,0]

        t = np.asarray(t).reshape(3, 1)
        Pc = (R @ objp.T + t).T                    # (N,3)
        Xc, Yc, Zc = Pc[:,0], Pc[:,1], Pc[:,2]

        x = Xc / Zc
        y = Yc / Zc
        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2
        x_tan = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tan = p1*(r2 + 2*y*y) + 2*p2*x*y
        xd = x*radial + x_tan
        yd = y*radial + y_tan

        alpha, gamma, u0 = K[0,0], K[0,1], K[0,2]
        beta,  v0        = K[1,1], K[1,2]
        u = u0 + alpha*xd + gamma*yd
        v = v0 + beta *yd
        return np.stack([u, v], axis=1)
    
    @staticmethod
    def visualized_reprojection_zhang_all(image_paths, m_list, p_plane,
                                         K, R_list, t_list, k1, k2,
                                         p1=0.0, p2=0.0,
                                         out_dir="calibration_results/reproj_zhang"):
        os.makedirs(out_dir, exist_ok=True)

        for idx, (img_path, m_obs, R, t) in enumerate(zip(image_paths, m_list, R_list, t_list)):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[skip] cannot read: {img_path}")
                continue

            # observed corner
            obs = (m_obs[:, :2] / m_obs[:, 2:3]).astype(np.float64)  # (N,2)

            # reprojected corner
            rep = Visualizer.project_points_plane_distort(p_plane, K, R, t, k1, k2, p1, p2)

            # draw
            vis = img.copy()
            for o in obs:
                cv2.circle(vis, (int(round(o[0])), int(round(o[1]))), 10, (0,0,255), -1)   # red
            for r in rep:
                cv2.circle(vis, (int(round(r[0])), int(round(r[1]))), 8, (0,255,0), -1)   # green

            # No essential
            err = np.linalg.norm(obs - rep, axis=1)
            mean_err = float(np.mean(err))
            rms_err  = float(np.sqrt(np.mean((obs - rep)**2)))
            stamp = f"Zhang reprojection | mean={mean_err:.3f}px, rms={rms_err:.3f}px"
            # cv2.putText(vis, stamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (32,32,32), 3, cv2.LINE_AA)
            # cv2.putText(vis, stamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)

            # Save
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{idx:03d}_{base}_zhang_reproj.jpg")
            cv2.imwrite(out_path, vis)
            # print(f"[saved] {out_path}  (mean={mean_err:.3f}, rms={rms_err:.3f})")
    
    @staticmethod
    def _draw_obs_vs_reproj(img, obs_xy, rep_xy,
                            radius_obs=10, radius_rep=8,
                            color_obs=(0,0,255), color_rep=(0,255,0)):
        vis = img.copy()
        for o in obs_xy:
            cv2.circle(vis, (int(round(o[0])), int(round(o[1]))),
                       radius_obs, color_obs, -1, lineType=cv2.LINE_AA)
        for r in rep_xy:
            cv2.circle(vis, (int(round(r[0])), int(round(r[1]))),
                       radius_rep, color_rep, -1, lineType=cv2.LINE_AA)
        return vis
    
    @staticmethod
    def visualize_all_reprojection_opencv(
        image_paths,
        objp_3d,                  # (N,3) [X,Y,Z] = [X,Y,0]
        rvecs, tvecs,             # lists from cv2.calibrateCamera
        K, dist_coeffs,           # OpenCV intrinsics & distortion
        observed_list,            # list of (N,3) homogeneous [u,v,1] (same 순서)
        out_dir="calibration_results/opencv_reproj",
        prefix="OpenCV",
        radius_obs=10, radius_rep=8
    ):
        os.makedirs(out_dir, exist_ok=True)

        assert len(image_paths) == len(rvecs) == len(tvecs) == len(observed_list), \
            "Must the same length"

        per_img_stats = []
        all_errs = []

        for i, (img_path, rvec, tvec, m_obs_h) in enumerate(
            zip(image_paths, rvecs, tvecs, observed_list)
        ):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Cannot read {img_path}, skip.")
                continue

            # Reprojection
            proj, _ = cv2.projectPoints(objp_3d.astype(np.float32),
                                        rvec, tvec, K, dist_coeffs)
            proj = proj.reshape(-1, 2)

            # observation -> 2D
            m_obs = (m_obs_h[:, :2] / m_obs_h[:, 2:3]).astype(np.float32)

            # Visualization
            vis = Visualizer._draw_obs_vs_reproj(
                img, m_obs, proj, radius_obs=radius_obs, radius_rep=radius_rep
            )
            out_path = os.path.join(out_dir, f"{prefix}_reproj_{i:02d}.jpg")
            cv2.imwrite(out_path, vis)

    @staticmethod
    def _distort_forward(x, y, k1, k2, p1, p2, k3=0.0):
        """normalized (x,y) -> after distortion (xd,yd) : radial + tangential"""
        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        x_tan = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
        y_tan = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
        xd = x*radial + x_tan
        yd = y*radial + y_tan
        return xd, yd
    
    @staticmethod
    def _invert_distortion_newton(xd, yd, k1, k2, p1, p2, k3=0.0, iters=5):
        x = xd.copy()
        y = yd.copy()
        for _ in range(iters):
            r2 = x*x + y*y
            radial = 1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            x_tan = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
            y_tan = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y

            fx = x*radial + x_tan - xd
            fy = y*radial + y_tan - yd

            # Jacobian (∂fx/∂x, ∂fx/∂y, ∂fy/∂x, ∂fy/∂y)
            dr_dx = 2.0*x
            dr_dy = 2.0*y
            d_rad_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r2*r2

            d_rad_dx = d_rad_dr2 * dr_dx
            d_rad_dy = d_rad_dr2 * dr_dy

            # ∂(x*radial)/∂x = radial + x*d_rad_dx
            d_xr_dx = radial + x * d_rad_dx
            d_xr_dy = x * d_rad_dy
            # ∂x_tan/∂x, ∂x_tan/∂y
            d_xt_dx = 2.0*p1*y + p2*(dr_dx + 4.0*x)
            d_xt_dy = 2.0*p1*x + p2*(dr_dy)

            dfx_dx = d_xr_dx + d_xt_dx
            dfx_dy = d_xr_dy + d_xt_dy

            # ∂(y*radial)/∂x = y*d_rad_dx
            d_yr_dx = y * d_rad_dx
            d_yr_dy = radial + y * d_rad_dy
            # ∂y_tan/∂x, ∂y_tan/∂y
            d_yt_dx = 2.0*p2*y + p1*(dr_dx)
            d_yt_dy = 2.0*p2*x + p1*(dr_dy + 4.0*y)

            dfy_dx = d_yr_dx + d_yt_dx
            dfy_dy = d_yr_dy + d_yt_dy

            # solve 2x2 linear system
            det = dfx_dx*dfy_dy - dfx_dy*dfy_dx
            # stability : update until det = 0
            mask = np.abs(det) > 1e-12
            dx = np.zeros_like(x)
            dy = np.zeros_like(y)
            dx[mask] = ( -fx[mask]*dfy_dy[mask] + fy[mask]*dfx_dy[mask]) / det[mask]
            dy[mask] = ( -fy[mask]*dfx_dx[mask] + fx[mask]*dfy_dx[mask]) / det[mask]

            x += dx
            y += dy
        return x, y
    
    @staticmethod
    def _bilinear_sample(img, u, v):
        h, w = img.shape[:2]
        u0 = np.floor(u).astype(np.int32)
        v0 = np.floor(v).astype(np.int32)
        u1 = u0 + 1
        v1 = v0 + 1

        u0 = np.clip(u0, 0, w-1); u1 = np.clip(u1, 0, w-1)
        v0 = np.clip(v0, 0, h-1); v1 = np.clip(v1, 0, h-1)

        wa = (u1 - u) * (v1 - v)
        wb = (u1 - u) * (v - v0)
        wc = (u - u0) * (v1 - v)
        wd = (u - u0) * (v - v0)

        Ia = img[v0, u0].astype(np.float32)
        Ib = img[v1, u0].astype(np.float32)
        Ic = img[v0, u1].astype(np.float32)
        Id = img[v1, u1].astype(np.float32)

        out = (Ia*wa[...,None] + Ib*wb[...,None] + Ic*wc[...,None] + Id*wd[...,None])
        return np.clip(out, 0, 255).astype(np.uint8)
    
    @staticmethod
    def undistort_nocv(img, K, dist):
        h, w = img.shape[:2]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        
        d = np.asarray(dist, dtype=np.float64).ravel()
        k1 = float(d[0]) if d.size > 0 else 0.0
        k2 = float(d[1]) if d.size > 1 else 0.0
        p1 = float(d[2]) if d.size > 2 else 0.0
        p2 = float(d[3]) if d.size > 3 else 0.0
        k3 = float(d[4]) if d.size > 4 else 0.0

        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))

        # normalized coordinate (x_d, y_d)  [distorted-normalized]
        xd = (uu - cx) / fx
        yd = (vv - cy) / fy

        x, y = Visualizer._invert_distortion_newton(xd, yd, k1, k2, p1, p2, k3, iters=5)

        # coordinate of source image
        us = fx * x + cx
        vs = fy * y + cy

        und = Visualizer._bilinear_sample(img, us, vs)
        return und

    @staticmethod
    def visualize_undistortion_all_nocv(
        image_paths,
        K,
        dist_coeffs,
        method_name="OpenCV",
        out_dir="calibration_results/undistort_custom",
        draw_side_by_side=True
    ):
        os.makedirs(out_dir, exist_ok=True)
        saved = []

        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"[skip] cannot read: {path}")
                continue

            und = Visualizer.undistort_nocv(img, K, dist_coeffs)

            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{base}_undist.jpg")
            cv2.imwrite(out_path, und)
            saved.append(out_path)
            
        return saved            