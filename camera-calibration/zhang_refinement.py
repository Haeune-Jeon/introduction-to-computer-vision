import numpy as np
from scipy.optimize import least_squares

class ZhangRefinement:
    def __init__(self, p_plane, m_list, A, R_list, t_list, k1=0.0, k2=0.0):
        """
        Args:
            p_plane: (N,3) model plane points [X,Y,1]
            m_list: list of (N,3) image points per view [u,v,1]
            A: (3,3) initial intrinsic matrix
            R_list: list of (3,3) rotation matrices
            t_list: list of (3,1) translation vectors
            k1, k2: initial distortion coefficients
        """
        self.p_plane = p_plane.astype(np.float64)
        self.m_list = [m.astype(np.float64) for m in m_list]
        self.num_views = len(self.m_list)
        self.A = A.astype(np.float64)
        self.R_list = [R.astype(np.float64) for R in R_list]
        self.t_list = [t.astype(np.float64) for t in t_list]
        self.k1 = float(k1)
        self.k2 = float(k2)
    
    # --------------- Rodrigues (R <-> r) -------------------
    @staticmethod
    def rodrigues_to_matrix(r):
        """
        Rodrigues vector r (3,) -> Rotation matrix R (3,3)
        """
        theta = np.linalg.norm(r)
        if theta < 1e-10:
            return np.eye(3, dtype=np.float64)
        
        k = r / theta
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=np.float64)
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    
    @staticmethod
    def matrix_to_rodrigues(R):
        """
        Rotation matrix R (3,3) -> Rodrigues vector r (3,)
        """
        theta = np.arccos((np.trace(R) - 1) / 2)
        if theta < 1e-10:
            return np.zeros(3, dtype=np.float64)
        
        K = (R - R.T) / (2 * np.sin(theta))
        k = np.array([K[2,1], K[0,2], K[1,0]], dtype=np.float64)
        r = theta * k
        return r
    
    # --------------- Projection with distortion -------------------
    def _plane_to_obj3d(self):
        """p_plane(N,3:[X,Y,1]) -> objp(N,3:[X,Y,0])"""
        XY = self.p_plane[:, :2]
        Z0 = np.zeros((XY.shape[0], 1), dtype=np.float64)
        return np.hstack([XY, Z0])
    
    def project_points_with_distortion(self, A, R, t, k1, k2, p1=0.0, p2=0.0):
        """
        Zhang 2.1 + 3.3:
        1) camera coordinate:   Pc = R*[X,Y,0]^T + t
        2) normalize:   x = Xc/Zc,  y = Yc/Zc
        3) radial distortion:      x' = x*(1 + k1 r^2 + k2 r^4),  y' = y*(1 + k1 r^2 + k2 r^4)
        4) pixel coordinate:     u = u0 + α x' + γ y',  v = v0 + β y'
        """
        objp = self._plane_to_obj3d()                    # (N,3) with Z=0
        t = np.asarray(t).reshape(3,1)   # (3,1)
        Pc = (R @ objp.T + t).T                          # (N,3): [Xc,Yc,Zc]
        Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

        # 2) normalized image coords
        x = Xc / Zc
        y = Yc / Zc

        # 3) radial distortion (k1,k2)
        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2
        x_tan = 2*p1*x*y + p2*(r2 + 2*x*x)
        y_tan = p1*(r2 + 2*y*y) + 2*p2*x*y
        xd = x*radial + x_tan
        yd = y*radial + y_tan

        # 4) pixel coords using intrinsics (allow skew γ)
        alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
        beta,  v0        = A[1, 1], A[1, 2]

        u = u0 + alpha * xd + gamma * yd
        v = v0 + beta  * yd

        return np.stack([u, v], axis=1)  
    
    # --------------- Non-linear Refinement -------------------
    def _pack_params(self, A, r_list, t_list, k1, k2):
        """
        Pack all parameters into a single vector
        Order: [alpha, gamma, u0, beta, v0, k1, k2, r1, t1, r2, t2, ...]
        """
        intrinsics = [A[0,0], A[0,2], A[1,1], A[1,2]]
        distortion = [k1, k2]
        extrinsics = []
        for r, t in zip(r_list, t_list):
            extrinsics.extend(np.asarray(r).reshape(3).tolist())
            extrinsics.extend(np.asarray(t).reshape(3).tolist())
        return np.array(intrinsics + distortion + extrinsics, dtype=np.float64)
    
    def _unpack_params(self, params):
        """
        Unpack parameter vector into A, R_list, t_list, k1, k2
        """
        alpha, u0, beta, v0 = params[0:4]
        k1, k2 = params[4:6]
        
        A = np.array([[alpha, 0.0, u0],
                      [0.0,   beta,  v0],
                      [0.0,   0.0,   1.0]], dtype=np.float64)
        
        # Extract extrinsics
        r_list = []
        t_list = []
        R_list = []
        idx = 6
        for _ in range(self.num_views):
            r = params[idx:idx+3]
            t = params[idx+3:idx+6]
            r_list.append(r)
            t_list.append(t)
            R_list.append(self.rodrigues_to_matrix(r))
            idx += 6
        
        return A, R_list, t_list, r_list, k1, k2
    
    def _residual_function(self, params):
        """
        Compute residuals for all points in all views
        Equation (14): ||m_ij - m_hat(A, k1, k2, R_i, t_i, M_j)||^2
        """
        A, R_list, t_list, _, k1, k2 = self._unpack_params(params)
        
        residuals = []
        for i, (R, t, m_obs) in enumerate(zip(R_list, t_list, self.m_list)):
            # Project with distortion
            m_proj = self.project_points_with_distortion(A, R, t, k1, k2)
            
            # Observed points
            m_obs_2d = m_obs[:, :2] / m_obs[:, 2:3]
            
            # Residual
            residuals.append((m_proj - m_obs_2d).ravel())
        return np.concatenate(residuals)
    
    def refine_all_parameters(self, max_iter=100, verbose=True):
        """
        Non-linear refinement using Levenberg-Marquardt
        Refines intrinsics (A), distortion (k1, k2), and extrinsics (R, t)
        """
        # Convert R to rodrigues vectors
        r_list = [self.matrix_to_rodrigues(R) for R in self.R_list]
        
        # Pack initial parameters
        params0 = self._pack_params(self.A, r_list, self.t_list, self.k1, self.k2)
        
        if verbose:
            print("Starting non-linear refinement...")
            print(f"Initial parameters: {len(params0)} parameters")
        
        # Run Levenberg-Marquardt optimization
        result = least_squares(
            self._residual_function,
            params0,
            method='lm',
            max_nfev=max_iter,
            verbose=2 if verbose else 0
        )
        
        # Unpack optimized parameters
        A_opt, R_opt, t_opt, _, k1_opt, k2_opt = self._unpack_params(result.x)
        
        if verbose:
            print(f"\nOptimization finished!")
            # print(f"Final RMS error: {np.sqrt(np.mean(result.fun**2)):.4f} pixels")
            print(f"Zhang's distortion : [{k1_opt}  {k2_opt}]")
        
        return {
            "K": A_opt,
            "k1": k1_opt,
            "k2": k2_opt,
            "R_list": R_opt,
            "t_list": t_opt,
            "rms_error": np.sqrt(np.mean(result.fun**2)),
            "success": result.success,
            "optim_result": result
        }
    
    # --------------- Reprojection Error (RMS) -------------------
    def reproj_error_per_view(self, A, R_list, t_list, k1, k2):
        """
        Compute reprojection error for given parameters
        """
        all_res = []
        total_N = 0
        
        for i, (R, t, m) in enumerate(zip(R_list, t_list, self.m_list)):
            proj = self.project_points_with_distortion(A, R, t, k1, k2)
            obs = m[:, :2] / m[:, 2:3]
            res = proj - obs
            all_res.append(res)
            total_N += res.shape[0]
        
        if total_N == 0:
            return float("nan"), float("nan")
            
        R = np.vstack(all_res)                    # (ΣN, 2)
        d_all = np.linalg.norm(R, axis=1)         # (ΣN,)
        total_mean_l2   = float(d_all.mean())
        total_rms_error = float(np.sqrt(np.mean(np.sum(R**2, axis=1))))
        return total_rms_error, total_mean_l2
