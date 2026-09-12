import numpy as np
from zhang_refinement import ZhangRefinement

class ZhangCalibrator:
    """
    (1) Inputs
        - p_plane : (N,3) float 32  # [X, Y, 1] on model plane (Z=0)
        - m_list : list of (N,3) float 32  # per-view image points [u,v,1]
    
    (2) Notes
        - Homography H_j 
        - Closed-form intrinsics : V b = 0 (equation (9))
    """
    def __init__(self, p_plane, m_list):
        assert p_plane.shape[1] == 3  # at least 3 views (for stability)
        self.p_plane = p_plane.astype(np.float64)
        self.m_list = [m.astype(np.float64) for m in m_list]
        self. num_views = len(self.m_list)
        self.H_list = []
        self.A = None  # intrinsics
        self.R_list = []  # extrinsics - rotation matrix
        self.t_list = []  # extrinsics - translation
        self.k1 = 0.0  # initial distortion (paper 3.3 : 0.... but I implemented 0:0)
        self.k2 = 0.0  # initial distortion
    
    # --------------- 1) Estimate H by normalized DLT -------------------
    @staticmethod
    def _normalize_points_2d(x3):  # (N,3), homogeneous
        x3 = x3.astype(np.float64)
        x = x3[:, :2] / x3[:, 2:3]  # (N,2) -> inhomogeneous
        mean = x.mean(axis=0)
        d = np.sqrt(((x-mean) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / (d + 1e-12)
        
        T = np.array([
            [s, 0.0, -s*mean[0]],
            [0.0, s, -s*mean[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        xn = (T @ x3.T).T  # normalized homogeneous (N,3)
        return xn, T
    
    @staticmethod
    def _dlt_homography(Mp, m):
        """ 
        Mp : (N, 3) model-plane homogeneous [X,Y,1]
        m : (N,3) image homogeneous [u,v,1]
        Returns H (3,3) s.t m~ H Mp
        """
        Mp_n, Tm = ZhangCalibrator._normalize_points_2d(Mp)
        m_n, Tm_img = ZhangCalibrator._normalize_points_2d(m)
        
        # M_i = [X_i, Y_i, 1] ^ T
        X = Mp_n[:, 0]
        Y = Mp_n[:, 1]
        
        # m_i = [u_i, v_i, 1] ^ T
        u = m_n[:, 0] / m_n[:, 2]
        v = m_n[:, 1] / m_n[:, 2]
        
        N = Mp.shape[0]
        A = np.zeros((2*N, 9), dtype=np.float64)
        for i in range(N):
            Xi, Yi, ui, vi = X[i], Y[i], u[i], v[i]
            A[2*i] = [0,0,0, Xi, Yi, 1, -vi*Xi, -vi*Yi, -vi]
            A[2*i+1] = [Xi, Yi, 1,  0,  0,  0, -ui*Xi,-ui*Yi,-ui]
        
        # Solve A h = 0 via SVD (last column of V)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]  # X : Lx = 0 solution
        Hn = h.reshape(3,3)  # normlalized M
        
        # denormalize : m ~ T_img^(-1) * Hm * T_model * Mp
        H = np.linalg.inv(Tm_img) @ Hn @ Tm
        
        # scale normalize (h33 = 1)
        if abs(H[2,2]) > 1e-12:
            H = H / H[2,2]
        return H
    
    def estimate_homographies(self):
        self.H_list = [self._dlt_homography(self.p_plane, m) for m in self.m_list]
        return self.H_list
    
    # --------------- 2) Construct v b = 0 (paper 3.1 B, 2.3) -------------------
    @staticmethod
    def _v_ij_from_H(H, i, j):
        h = H
        hi = h[:, i-1]
        hj = h[:, j-1]
        return np.array([
            hi[0]*hj[0],                              # h_i1 h_j1
            hi[0]*hj[1] + hi[1]*hj[0],                # h_i1 h_j2 + h_i2 h_j1
            hi[1]*hj[1],                              # h_i2 h_j2
            hi[2]*hj[0] + hi[0]*hj[2],                # h_i3 h_j1 + h_i1 h_j3
            hi[2]*hj[1] + hi[1]*hj[2],                # h_i3 h_j2 + h_i2 h_j3
            hi[2]*hj[2]                               # h_i3 h_j3
        ], dtype=np.float64)  # euqation (7) (in paper)
    
    def _stack_V(self):
        """ 
        equation (8)
        v12^T = b
        stack to V (2n * 6)
        """
        rows = []
        for H in self.H_list:
            v12 = self._v_ij_from_H(H, 1, 2)
            v11 = self._v_ij_from_H(H, 1, 1)
            v22 = self._v_ij_from_H(H, 2, 2)
            rows.append(v12)
            rows.append(v11 - v22)
        V = np.vstack(rows)
        return V
    
    @staticmethod
    def _solbe_b(V):
        # Solve V b = 0 by SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]
        return b
    
    # --------------- 3) b -> A (intrinsics) -------------------
    @staticmethod
    def _A_from_b(b):
        """ 
        b = [B11, B12, B22, B13, B23, B33]^T (B symmetric)
        Use closed-form in Zhang's paper (3.1)
        Appendix B : Extraction of the Intrinsic Parameters from Matrix B
        """
        B11, B12, B22, B13, B23, B33 = b
        v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
        lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
        alpha = np.sqrt(lam / B11)
        beta  = np.sqrt(lam * B11 / (B11*B22 - B12**2))
        gamma = -B12 * alpha**2 * beta / lam
        u0    = (gamma * v0 / beta) - (B13 * alpha**2 / lam)
        
        # equation in 2.1
        A = np.array([[alpha, gamma, u0],
                      [0.0,   beta,  v0],
                      [0.0,   0.0,   1.0]], dtype=np.float64)
        
        return A
    
    # --------------- 4) (R,t) for each view -------------------
    @staticmethod
    def _extrinsic_from_H(A,H):
        # Eq from paper 7p
        Ainv = np.linalg.inv(A)
        h1 = H[:,0]; h2 = H[:,1]; h3 = H[:,2]
        # scale λ = 1/||A^{-1} h1|| = 1/||A^{-1} h2|| (from 6p)
        lam = 1.0 / np.linalg.norm(Ainv @ h1)
        r1 = lam * (Ainv @ h1)
        r2 = lam * (Ainv @ h2)
        r3 = np.cross(r1, r2)
        t  = lam * (Ainv @ h3)
        
        if t[2] < 0:
            r1, r2, t = -r1, -r2, -t
        
        # Orthonormalized R via SVD
        R_hat = np.column_stack([r1, r2, r3])  # R_hat = [r1, r2, r3] : with noise
        U, _, Vt = np.linalg.svd(R_hat)        # # \hat R = U S V^T
        R = U @ Vt
        # Ensure det(R)=+1
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R, t
    
    def _fix_skew_in_A(self):
        # Fix skew = 0
        if self.A is not None:
            self.A[0,1] = 0.0
    
    # --------------- 5) Non-linear Refinement (delegated to ZhangRefinement) -------------------
    def refine_all_parameters(self, max_iter=100, verbose=True):
        """
        Non-linear refinement using ZhangRefinement class
        """
        if self.A is None:
            raise ValueError("Run calibrate() first to get initial estimates")
        
        # Create refinement instance
        refiner = ZhangRefinement(
            self.p_plane, self.m_list, self.A, 
            self.R_list, self.t_list, self.k1, self.k2
        )
        
        # Run refinement
        result = refiner.refine_all_parameters(max_iter, verbose)
        
        # Update stored parameters
        self.A = result["K"]
        self.R_list = result["R_list"]
        self.t_list = result["t_list"]
        self.k1 = result["k1"]
        self.k2 = result["k2"]
        
        return result
    
    # --------------- 8) end-to-end -------------------
    def calibrate(self):
        if not self.H_list:
            self.estimate_homographies()
        V = self._stack_V()  # stack linear constraint from 2.3
        b = self._solbe_b(V)  # Solution for Vb=0 (SVD)
        self.A = self._A_from_b(b)  # Eq. 3.1 (from paper)
        
        # Fix skew = 0
        self._fix_skew_in_A()
        
        # extrinsics per view
        self.R_list, self.t_list = [], []
        for H in self.H_list:
            R,t = self._extrinsic_from_H(self.A, H)
            self.R_list.append(R)
            self.t_list.append(t)
            
        return {
            "K": self.A,               # (alpha,gamma,u0; 0,beta,v0; 0,0,1)
            "H_list": self.H_list,
            "R_list": self.R_list,
            "t_list": self.t_list,
            "V": V
        }
    
    # --------------- 6) Reprojection Error (RMS) -------------------
    def reproj_error_per_view(self, use_distortion=True):
        """
        Compute reprojection error using ZhangRefinement
        """
        if use_distortion:
            # Use ZhangRefinement for distortion-aware projection
            refiner = ZhangRefinement(
                self.p_plane, self.m_list, self.A, 
                self.R_list, self.t_list, self.k1, self.k2
            )
            return refiner.reproj_error_per_view(self.A, self.R_list, self.t_list, self.k1, self.k2)
        else:
            # Simple homography-based projection (no distortion)
            all_res = []
            total_N = 0
            
            for i, (H, m) in enumerate(zip(self.H_list, self.m_list)):
                proj_h = (H @ self.p_plane.T).T
                proj = proj_h[:, :2] / proj_h[:, 2:3]
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
    