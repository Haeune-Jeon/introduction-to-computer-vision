import numpy as np
import cv2
import glob

class OpenCVCalibrator:
    @staticmethod
    def calibrate_camera(corners_list, board_size=(13, 9), square_size=20.0,
                         image_paths=None, image_size=None):
        """ 
        - corners_list : homogeneous points [u,v,1]
        - board_size : (cols, rows)
        - square_size : float, mm 
        - image_path : list[str]
        - image_size : (W,H)
        """
        cols, rows = board_size
        N_corners = cols * rows
        
        # 1) 3D world coordinate (Z=0)
        objp = np.zeros((N_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(square_size)
        
        # 2D
        objpoints = []  # 3D world coordinate
        imgpoints = []  # 2D image coordinate
        
        for corners in corners_list:
            uv = corners[:, :2].astype(np.float32)
            assert uv.shape[0] == N_corners, f"expected {N_corners}, got {uv.shape[0]}"
            objpoints.append(objp)
            imgpoints.append(uv)
        
        # 2) image size (from first image)
        if image_paths is not None and len(image_paths) > 0:
            img = cv2.imread(image_paths[0])
            if img is None:
                raise RuntimeError(f"Failed to read image: {image_paths[0]}")
            img_size_cv = (img.shape[1], img.shape[0])
        elif image_size is not None:
            img_size_cv = tuple(image_size)  # (W,H)
        else:
            raise ValueError("Provide image_paths or image_size to determine image size.")
        
        print(f"Calibrating with {len(objpoints)} images, image size: {img_size_cv}")
                
        # 3) OpenCV camera calibration
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size_cv, None, None
        )
        
        # 4) Extrinsic ->  (R,t)
        extrinsic_RT = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)  # (3,3)
            t = tvec.reshape(3,1)  # (3,1)
            extrinsic_RT.append((R,t))
        
        # 5) reprojection error (RSM)
        all_res = []
        total_pts = 0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2).astype(np.float32)
            gt = imgpoints[i].astype(np.float32)
            
            res = proj - gt
            all_res.append(res)
            total_pts += res.shape[0]
        
        if total_pts > 0:
            R_all = np.vstack(all_res)
            d_all = np.linalg.norm(R_all, axis=1)
            global_mean_l2 = float(d_all.mean())
            global_rmse_point = float(np.sqrt(np.mean(np.sum(R_all**2, axis=1))))
        else:
            global_mean_l2 = global_rmse_point = float("nan")
        
        return {
            "rms": rms,                           # OpenCV global RMS
            "camera_matrix": K,
            "distortion_coeffs": dist,
            "rotation_vectors": rvecs,
            "translation_vectors": tvecs,
            "extrinsics_RT": extrinsic_RT,        # list of (R,t)
            "image_size": img_size_cv,
            "num_images": len(objpoints),
            "global_rmse_point": global_rmse_point,   
            "global_mean_l2": global_mean_l2,         
        }
    
    # decompose R and t
    @staticmethod
    def parse_RT_to_lists(RT):
        if isinstance(RT, tuple) and len(RT) == 2:
            # RT is (R_list, t_list)
            R_list, t_list = RT
            # Ensure proper format
            R_list = [np.array(R) for R in R_list]
            t_list = [np.array(t) if isinstance(t, np.ndarray) else np.array(t)
                      for t in t_list]
            t_list = [t.reshape(3,1) if t.ndim == 1 else t for t in t_list]
        elif isinstance(RT, np.ndarray):
            if RT.ndim == 3:  # (n_views, 3, 4)
                n_views = RT.shape[0]
                R_list = [RT[i, :, :3] for i in range(n_views)]
                t_list = [RT[i, :, 3:4] for i in range(n_views)]
            elif isinstance(RT, np.ndarray):
                if RT.ndim == 3:  # (n_views, 3, 4)
                    n_views = RT.shape[0]
                    R_list = [RT[i, :, :3] for i in range(n_views)]
                    t_list = [RT[i, :, 3:4] for i in range(n_views)]
            elif RT.ndim == 2:  # (3, 4)
                R_list = [RT[:, :3]]
                t_list = [RT[:, 3:4]]
            else:
                raise ValueError(f"Unsupported numpy array shape: {RT.shape}.")
        elif isinstance(RT, list):
            if len(RT) == 0:
                raise ValueError("Empty RT list provided")
            # List of (3, 4) matrices
            R_list = []
            t_list = []
            for i, rt in enumerate(RT):
                rt = np.array(rt)
                if rt.shape != (3, 4):
                    raise ValueError(f"RT[{i}] has shape {rt.shape}, expected (3, 4)")
                R_list.append(rt[:, :3])
                t_list.append(rt[:, 3:4])
        else:
            raise ValueError(f"Unsupported RT format: {type(RT)}. Supported formats: tuple, numpy.ndarray, list")
        
        return R_list, t_list