import cv2, os, glob
import numpy as np

class CornerDetector:
    @staticmethod
    def load_corners_for_all_views(checkerboard_folder="checker_board", 
                                   board_size=(13, 9),   # (cols, rows)
                                   max_images=None,
                                   use_sb=True, 
                                   save_debug=False):
        # 1) image files
        image_paths = glob.glob(os.path.join(checkerboard_folder, "*.jpg"))
        image_paths.sort()
        
        # limit max number of images
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images in {checkerboard_folder}")
        
        cols, rows = board_size
        expected = cols * rows
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        
        m_list = []
        successful_images = []
        
        for i, img_path in enumerate(image_paths):
            # print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Failed to load image: {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Corner detection
            ret, corners = False, None
            if use_sb and hasattr(cv2, "findChessboardCornersSB"):
                try:
                    sb = cv2.findChessboardCornersSB(gray, (cols, rows))
                    if isinstance(sb, tuple):
                        ret = bool(sb[0])
                        corners = sb[1]
                    else:
                        corners = sb
                        ret = corners is not None and corners.shape[0] == expected
                except Exception:
                    ret = False
                    corners = None
            
            # 2) fallback (past version)
            if not ret:
                flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                         cv2.CALIB_CB_NORMALIZE_IMAGE |
                         cv2.CALIB_CB_FAST_CHECK)
                ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            
            if not ret or corners is None or corners.shape[0] != expected:
                print(f"  [fail] corners not found ({corners.shape[0] if corners is not None else 0}/{expected}).")
                continue
            
            # 3) refine Subpixel
            corners = corners.astype(np.float32)
            win_size = (11, 11)
            zero_zone = (-1, -1)
            corners2 = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
            
            uv = corners2.reshape(-1, 2).astype(np.float32)
            if uv.shape[0] != expected:
                print(f"  [warn] detected {uv.shape[0]} != expected {expected}")
                continue
            
            # 6) m = (u,v,1) - homogeneous
            ones = np.ones((uv.shape[0], 1), dtype=np.float32)
            uv1 = np.hstack([uv, ones])  # (N,3)
            m_list.append(uv1)
            successful_images.append(img_path)
            # print(f"  [ok] {uv.shape[0]} corners")

            # save image
            if save_debug:
                dbg = img.copy()
                cv2.drawChessboardCorners(dbg, (cols, rows), corners2, True)
                out = os.path.join(checkerboard_folder, f"_dbg_{os.path.basename(img_path)}")
                cv2.imwrite(out, dbg)
        
        print(f"\nSuccessfully processed {len(m_list)} / {len(image_paths)} images")
        return m_list, successful_images
