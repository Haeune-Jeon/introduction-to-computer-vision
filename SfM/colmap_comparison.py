"""
COLMAP comparison script for Structure from Motion.
This script runs COLMAP on the same dataset and visualizes the results.
Fixed for WSL environment without GPU/OpenGL support.
"""
import os
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from typing import Tuple, Optional

# Import our modules
from config import DATA_DIR, OUT_DIR
from utils import pick_first_video, ensure_dir
from frame_extraction import extract_frames
from visualization import SfMVisualizer
from reprojection_error import colmap_error_report    
from colmap_parser import load_colmap_intrinsics, load_colmap_poses_cam2world
from vis_open3d import SfMVisualizerO3D, ViewSpec

def _sh(cmd, env=None):
    res = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if res.returncode != 0:
        print("\n[CMD FAILED]", " ".join(cmd))
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError("Command failed")
    return res


def _check_colmap() -> bool:
    try:
        r = subprocess.run(["colmap", "--help"], text=True, capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def run_colmap_sfm_default():
    if not _check_colmap():
        raise RuntimeError("COLMAP not found. Install with: sudo apt-get install colmap")

    # --- workspace
    work_dir = ensure_dir(OUT_DIR / "colmap")
    images_dir = ensure_dir(work_dir / "images")
    sparse_dir = ensure_dir(work_dir / "sparse")
    db_path = work_dir / "database.db"
    if db_path.exists():
        db_path.unlink()  # 새로 시작

    # 헤드리스/WSL에서도 안전하게
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    # --- pick video & extract frames (only if not already)
    # video_path = pick_first_video(DATA_DIR)
    # video_path = "/mnt/d/대학교/3-2/기초컴퓨터비전이론및응용/assignment_02/data/IMG_1578.mov"
    # video_path = "/mnt/d/대학교/3-2/기초컴퓨터비전이론및응용/assignment_02/data/IMG_3592.mov"
    video_path = r"D:\\대학교\\3-2\\기초컴퓨터비전이론및응용\\assignment_02\\data\\IMG_3592.MOV"
    # video_path = "/mnt/d/대학교/3-2/기초컴퓨터비전이론및응용/assignment_02/data/IMG_3600.mov"
    # video_path = "/mnt/d/대학교/3-2/기초컴퓨터비전이론및응용/assignment_02/data/IMG_3597.mov"

    print(f"[COLMAP] Video: {video_path}")
    # print("[COLMAP] Extracting frames @ 10 FPS ...")
    frames = extract_frames(video_path, images_dir, target_fps=10.0)
    print(f"[COLMAP] Extracted {frames} frames -> {images_dir}")
    

    if not any(images_dir.glob("*.jpg")):
        print("[COLMAP] Extracting frames @ 5 FPS ...")
        # print("[COLMAP] Extracting frames @ 10 FPS ...")
        frames = extract_frames(video_path, images_dir, target_fps=2.0)
        print(f"[COLMAP] Extracted {frames} frames -> {images_dir}")
        
    else:
        frames = len(list(images_dir.glob("*.jpg")))
        print(f"[COLMAP] Using existing {frames} frames -> {images_dir}")

    sample = sorted(images_dir.glob("*.jpg"))[0]
    img0 = cv.imread(str(sample), cv.IMREAD_GRAYSCALE)
    H0, W0 = img0.shape[:2]
    # 네가 쓰는 모드로 선택: (f35만 알고 있으면 f35_mm=26.0 사용)
    # K_native = estimate_camera_intrinsics(width_px=W0, height_px=H0,
    #                                       f35_mm=26.0,  # 필요시 f_mm+sensor_size_mm로 교체
    #                                       principal_at_center=True)

    # ---- COLMAP 설정 ----
    COLMAP_MAX_IMAGE_SIZE = 4000  # 리사이즈 안 하려면 0. (리사이즈 쓰면 아래 K_scaled 사용)
    # cam_model = "OPENCV"      # 언디스토트 안 했고 왜곡 모를 때는 K만 고정해서 PINHOLE 권장
    # K_scaled = _scale_K_for_resize(K_native, (W0, H0), COLMAP_MAX_IMAGE_SIZE)
    # fx, fy, cx, cy = K_scaled[0,0], K_scaled[1,1], K_scaled[0,2], K_scaled[1,2]
    
    # k1 = k2 = p1 = p2 = 0.0
    # cam_params = f"{fx},{fy},{cx},{cy},0,0,0,0"
    
    # --- feature extraction (CPU)
    print("[COLMAP] Feature extraction (SIFT, CPU)")
    _sh([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),

        "--ImageReader.single_camera", "1",
        # "--ImageReader.camera_model", "PINHOLE",
        # "--ImageReader.camera_params", cam_params,

        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
        "--SiftExtraction.max_num_features", "12000",
    ], env=env)

    # --- sequential matching (video-friendly)
    print("[COLMAP] Sequential matching (CPU)")
    _sh([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "5",
        "--SequentialMatching.loop_detection", "0",
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
    ], env=env)

    # --- sparse mapper (SfM)
    print("[COLMAP] Sparse reconstruction (mapper)")
    _sh([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", str(os.cpu_count() or 4),
        
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ], env=env)

    # --- convert model to TXT
    model0 = sparse_dir / "0"
    if not model0.exists():
        # 컬맵이 여러 모델(0,1,2,...)을 만들 수 있음 → 첫 폴더 사용
        subs = [p for p in sparse_dir.iterdir() if p.is_dir()]
        model0 = subs[0] if subs else None

    if model0 is None:
        print("[COLMAP] ⚠ No model directory found under sparse/.")
        return

    print("[COLMAP] Converting model to TXT")
    _sh([
        "colmap", "model_converter",
        "--input_path", str(model0),
        "--output_path", str(model0),
        "--output_type", "TXT",
    ], env=env)

    # --- quick stats
    pts_txt = model0 / "points3D.txt"
    npts = sum(1 for ln in pts_txt.open() if ln.strip() and not ln.startswith("#")) if pts_txt.exists() else 0
    print("\n==========================")
    print(" COLMAP SfM (CPU) DONE")
    print("==========================")
    print(f" • Workspace:    {work_dir}")
    print(f" • Database:     {db_path}")
    print(f" • Images dir:   {images_dir}")
    print(f" • Sparse model: {model0} (cameras.txt, images.txt, points3D.txt)")
    print(f" • #Frames:      {frames}")
    print(f" • #3D points:   {npts}")
    
    return work_dir, model0


def load_colmap_points_txt(model_dir: Path):
    """points3D.txt → (N,3) float32, (N,3) uint8 (없으면 None)"""
    model_dir = Path(model_dir)
    pts_file = model_dir / "points3D.txt"
    if not pts_file.exists():
        print(f"[COLMAP] points3D.txt not found: {pts_file}")
        return None, None

    xs, rgb = [], []
    with open(pts_file, "r") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.strip().split()
            # COLMAP TXT: id X Y Z R G B error track_len <(img_id, kp_idx)*>
            if len(p) >= 7:
                x, y, z = map(float, p[1:4])
                r, g, b = map(int, p[4:7])
                xs.append((x, y, z))
                rgb.append((r, g, b))
    if not xs:
        return None, None
    import numpy as np
    X = np.asarray(xs, dtype=np.float32)
    C = np.asarray(rgb, dtype=np.uint8)
    return X, C


def visualize_colmap_sparse(model_dir: Path, out_path: Path,
                            every: int = 1, s: int = 1, alpha: float = 0.9):
    """축/그리드 없는 minimal 뷰로 스파스 포인트만 저장."""
    from visualization import SfMVisualizer  # 너의 미니멀 뷰어

    X, C = load_colmap_points_txt(model_dir)
    if X is None or len(X) == 0:
        print("[COLMAP] No 3D points to visualize.")
        return

    viz = SfMVisualizer()
    viz.setup_minimal(bg="white")
    # RGB가 있으면 색 사용, 없으면 회색
    if C is not None:
        viz.plot_sparse_cloud(X, every=every, s=s, alpha=alpha, color="#666666")
    else:
        viz.plot_sparse_cloud(X, every=every, s=s, alpha=alpha, color="#666666")
    viz.save_plot(out_path)
    print(f"[COLMAP] Saved sparse visualization → {out_path}")


if __name__ == "__main__":
    work_dir, model0 = run_colmap_sfm_default()
    if model0 is None:
        sys.exit(1)
        
    img_path = work_dir / "colmap_sparse.png"
    visualize_colmap_sparse(model0, img_path, every=1, s=1, alpha=0.9)
    # ------------------------- visualize Open3D -------------------------------
    cams_by_id = load_colmap_intrinsics(model0)
    poses = load_colmap_poses_cam2world(model0)  # [(Rcw, tcw, name, cam_id), ...]

    # 2) 카메라 리스트(cam2world) & 이미지 크기(K) 결정
    cams_c2w = [(Rcw, tcw) for (Rcw, tcw, name, cam_id) in poses]
    # 첫 카메라의 intrinsics 사용 (복수이면 동일 모델 가정)
    if len(poses) == 0:
        print("[Open3D] No registered images. Skip visualization.")
    else:
        first_cam_id = poses[0][3]
        if first_cam_id not in cams_by_id:
            # 혹시 mapper가 여러 camera_id를 만들었으면, 실제 등장하는 cam_id 중 첫 것 사용
            for _, _, _, cid in poses:
                if cid in cams_by_id:
                    first_cam_id = cid
                    break
        K = cams_by_id[first_cam_id]["K"]
        W = cams_by_id[first_cam_id]["w"]
        H = cams_by_id[first_cam_id]["h"]

        # 3) 포인트클라우드 로드
        X_col, _rgb = load_colmap_points_txt(model0)
        if X_col is None:
            X_col = np.zeros((0,3), dtype=float)

        # 4) Open3D로 그림
        viz = SfMVisualizerO3D()
        viz.set_background((1,1,1))

        # 포인트 (깊이색상 옵션)
        if len(X_col) > 0:
            viz.add_point_cloud(
                X_col, every=1, sample_max=120_000, clip_quantile=0.995,
                color="#666666", scalars=X_col[:, 2], cmap="turbo"
            )

        # 경로: cam2world 기준 (center = tcw)
        viz.add_camera_path(
            cams_c2w, pose_type="cam2world", color=(1.0, 0.0, 0.0),
            normalize=False, max_len_ratio=10.0
        )

        # 프러스텀: cam2world로 일관
        viz.add_cameras(
            cams_c2w, K=K, img_size=(W, H),
            scale=0.12, color=(1.0, 0.0, 0.0),
            show_axes=False, filled=True
        )

        # 화면 띄우기 (혹은 저장)
        viz.show(view=ViewSpec(20, -60, 1.6), width=1600, height=1200)
    
    col_rep = colmap_error_report(r"D:\\대학교\\3-2\\기초컴퓨터비전이론및응용\\assignment_02\\runs\\run_3592\\colmap\\sparse\\0")
    print("[COLMAP points3D.txt] ->", col_rep)
