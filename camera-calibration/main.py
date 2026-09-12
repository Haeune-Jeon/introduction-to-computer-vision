import glob
import os
import numpy as np
import cv2

# Module
from board_spec import BoardSpec
from corner_detection import CornerDetector
from zhang_calibrator import ZhangCalibrator
from opencv_calibration import OpenCVCalibrator
from visualization import Visualizer

from result import print_calibration_summary  # for report

def main():
    print("="*60)
    print("CAMERA CALIBRATION: Zhang's Method vs OpenCV")
    print("="*60)
    
    # --------------------------------------------------
    # 1) Board (13x9 corners, square_size = 20mm)
    # --------------------------------------------------
    board = BoardSpec(rows=13, cols=9, square_size=20.0)
    print(f"Board specification: {board.rows}x{board.cols} corners, {board.square_size}mm squares")
    
    # homogeneous coodrdinate
    M_homo = board.get_M_homo()  # (N,4): [X, Y, 0, 1]
    p_plane = board.get_p_plane()  # (N,3): [u, v, 1]


    # --------------------------------------------------
    # 2) Corner Detection
    # --------------------------------------------------
    print("\n" + "="*40)
    print("STEP 1: CORNER DETECTION")
    print("="*40)
    m_list, successful_images = CornerDetector.load_corners_for_all_views(
        "checker_board6", (9, 13), max_images=45, use_sb=True, save_debug=False)  # 모든 이미지 사용
    
    print(M_homo.shape, p_plane.shape)   # (N,4), (N,3)
    print(len(m_list), len(successful_images))
    print(m_list[0].shape)
    
    
    # --------------------------------------------------
    # 3) Zhang's Method
    # --------------------------------------------------
    print("\n" + "="*40)
    print("STEP 2: ZHANG'S METHOD CALIBRATION")
    print("="*40)
    calib = ZhangCalibrator(p_plane, m_list)
    zhang_result = calib.calibrate()
    
    refined = calib.refine_all_parameters(max_iter=200, verbose=True)  # If you want to Zhang's method without distortion, erase this line
    
    rms_after, mean_after = calib.reproj_error_per_view(use_distortion=True)
    
    # st()
    print("Total RMS error =", rms_after)
    print("Total Mean L2 error =", mean_after)
    
    K       = calib.A
    R_list  = calib.R_list
    t_list  = calib.t_list
    
    # print("K=\n", K)
    
    print_calibration_summary(K, R_list, t_list, title="Zhang's Method — Report-Ready Console Summary")  # print results for results (You don't need to do this code only for debug)
    
    # --------------------------------------------------
    # 4) OpenCV
    # --------------------------------------------------
    print("\n" + "="*40)
    print("STEP 3: OPENCV CALIBRATION")
    print("="*40)
    
    opencv_result = OpenCVCalibrator.calibrate_camera(
        corners_list = m_list,
        board_size=(9, 13), 
        square_size=20.0,
        image_paths=successful_images)
    
    K_cv   = opencv_result["camera_matrix"]
    dist   = opencv_result["distortion_coeffs"].ravel()
    rms_cv = opencv_result["rms"]
    
    # print("OpenCV K=\n", opencv_result["camera_matrix"])
    print("OpenCV dist=\n", opencv_result["distortion_coeffs"].ravel())
    print(f"OpenCV RMS (cv2.calibrateCamera): {opencv_result['rms']:.6f} px")
    print(f"OpenCV MeanL2 (cv2.calibrateCamera): {opencv_result['global_mean_l2']:.6f} px")
    
    # extrinsic vector -> extrinsic matrix
    R_list_cv, t_list_cv = map(list, zip(*opencv_result["extrinsics_RT"]))
    t_list_cv = [t.reshape(3,) for t in t_list_cv]
    
    print_calibration_summary(K_cv, R_list_cv, t_list_cv, title="OpenCV's Method — Report-Ready Console Summary")  # print results for results (You don't need to do this code only for debug)
    
    
    # 5) Visualization
    print("\n" + "="*40)
    print("STEP 5: VISUALIZATION")
    print("="*40)
    
    # Zhang's parameter
    K_zhang = calib.A.copy()
    dist_zhang = np.array([calib.k1, calib.k2, getattr(calib, 'p1', 0.0), getattr(calib, 'p2', 0.0), 0.0], dtype=np.float64)
    
    # visualize Zhang's results
    Visualizer.visualized_reprojection_zhang_all(
        successful_images,
        m_list, p_plane, K_zhang, calib.R_list, calib.t_list, calib.k1, calib.k2,
        getattr(calib, "p1", 0.0), getattr(calib, "p2", 0.0),
        out_dir="calibration_results/reproj_zhang"
    )
    
    # OpenCV's parameter
    K_cv = opencv_result['camera_matrix']
    dist_cv = opencv_result['distortion_coeffs']
    
    # 3D point (Z=0)
    objp_3d = np.hstack([
        p_plane[:, :2],
        np.zeros((p_plane.shape[0], 1), dtype=np.float32)
    ]).astype(np.float32)
    
    # visualize OpenCV's results
    Visualizer.visualize_all_reprojection_opencv(
        image_paths=successful_images,
        objp_3d=objp_3d,                                   # (N,3) [X,Y,0]
        rvecs=opencv_result["rotation_vectors"],           # list of (3,1)
        tvecs=opencv_result["translation_vectors"],        # list of (3,1)
        K=K_cv,                                            # (3,3)
        dist_coeffs=dist_cv,                               # (5,) or (1,5)
        observed_list=m_list,                              # list of (N,3) homogeneous [u,v,1]
        out_dir="calibration_results/reproj_opencv",       # path of save file
        prefix="OpenCV",
        radius_obs=10,                                     # Red : Observed corner
        radius_rep=8                                       # Green : Reprpjected corner
    )
    
    # visualize undistored image
    Visualizer.visualize_undistortion_all_nocv(
        successful_images,
        K_cv, dist_cv,
        method_name="OpenCV (custom undistort)",
        out_dir="calibration_results/undistort_opencv_custom"
    )


if __name__ == "__main__":
    main()
