import numpy as np
from pathlib import Path
import cv2 as cv

# ---------- 기본 유틸 ----------
def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (R[2,1] - R[1,2]) / S
        q[2] = (R[0,2] - R[2,0]) / S
        q[3] = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[0] = (R[2,1] - R[1,2]) / S
            q[1] = 0.25 * S; q[2] = (R[0,1] + R[1,0]) / S; q[3] = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[0] = (R[0,2] - R[2,0]) / S
            q[1] = (R[0,1] + R[1,0]) / S; q[2] = 0.25 * S; q[3] = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[0] = (R[1,0] - R[0,1]) / S
            q[1] = (R[0,2] + R[2,0]) / S; q[2] = (R[1,2] + R[2,1]) / S; q[3] = 0.25 * S
    return q

def build_extid_to_name(matches_data: list) -> dict:
    ext2name = {}
    for m in matches_data:
        v1, v2 = int(m['view1_id']), int(m['view2_id'])
        ext2name.setdefault(v1, Path(m['f1']).name)
        ext2name.setdefault(v2, Path(m['f2']).name)
    return ext2name

# ---------- (A) 포즈 해석 자동 판별 ----------
def _cheirality_ratio(cams, X, pose_kind):
    """pose_kind:
       0: (Rcw,  t_cw)   x_cam = R x + t
       1: (Rcw,  C_w )   t = -R @ C
       2: (Rwc,  t_wc)   Rcw=Rwc^T, t=-Rwc^T t_wc
       3: (Rwc,  C_w )   Rcw=Rwc^T, t=-Rwc^T C
    """
    ok = 0; tot = 0
    idx = np.arange(min(len(X), 5000))
    Xsub = X[idx]
    for (R_, t_) in cams:
        if R_ is None: continue
        R_ = np.asarray(R_, np.float64)
        t_ = np.asarray(t_, np.float64).reshape(3,)
        if pose_kind == 0:
            Rcw, t = R_, t_
        elif pose_kind == 1:
            Rcw, t = R_, -R_ @ t_
        elif pose_kind == 2:
            Rcw, t = R_.T, -R_.T @ t_
        else:
            Rcw, t = R_.T, -R_.T @ t_
        Z = (Rcw @ Xsub.T + t.reshape(3,1)).T[:,2]
        tot += len(Z); ok += np.count_nonzero(Z > 0)
    return ok / max(1, tot)

def autodetect_pose_convention(cams, X):
    ratios = [ _cheirality_ratio(cams, X, k) for k in range(4) ]
    best = int(np.argmax(ratios))
    return best, ratios[best], ratios

def convert_pose(R_in, t_in, pose_kind):
    R_in = np.asarray(R_in, np.float64)
    t_in = np.asarray(t_in, np.float64).reshape(3,)
    if pose_kind == 0:    # (Rcw, t_cw)
        return R_in, t_in
    elif pose_kind == 1:  # (Rcw, C_w) -> t=-R C
        return R_in, -R_in @ t_in
    elif pose_kind == 2:  # (Rwc, t_wc) -> Rcw=Rwc^T, t=-R^T t_wc
        Rcw = R_in.T; t = -Rcw @ t_in
        return Rcw, t
    else:                  # (Rwc, C_w) -> Rcw=Rwc^T, t=-R^T C
        Rcw = R_in.T; t = -Rcw @ t_in
        return Rcw, t

# ---------- (B) COLMAP 텍스트 출력 ----------
def write_colmap_cameras_txt(path: Path, camera_id: int, width: int, height: int,
                             K: np.ndarray, dist: np.ndarray | None = None):
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    if dist is None: k1=k2=p1=p2=0.0
    else:
        k1 = float(dist[0]) if len(dist)>0 else 0.0
        k2 = float(dist[1]) if len(dist)>1 else 0.0
        p1 = float(dist[2]) if len(dist)>2 else 0.0
        p2 = float(dist[3]) if len(dist)>3 else 0.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{camera_id} OPENCV {width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")

def write_colmap_images_txt(path: Path, sfm, image_id_map: dict, camera_id: int,
                            extid_to_name: dict, pose_kind: int):
    per_img_obs = {img_id: [] for img_id in image_id_map.values()}
    # COLMAP은 POINT3D_ID 1-based가 일반적 → 1부터
    for (cam_idx, pid0, u, v) in sfm.observations:
        if cam_idx in image_id_map:
            per_img_obs[image_id_map[cam_idx]].append((float(u), float(v), int(pid0)+1))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_id_map)}\n")
        for cam_idx, img_id in sorted(image_id_map.items(), key=lambda x: x[1]):
            R_in, t_in = sfm.cameras[cam_idx]
            Rcw, t = convert_pose(R_in, t_in, pose_kind)
            q = rotmat_to_qvec(Rcw)
            ext_id = sfm.inv_view_id_map.get(cam_idx, cam_idx)
            name = extid_to_name.get(ext_id, f"img_{ext_id:06d}.jpg")
            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n")
            pts = per_img_obs.get(img_id, [])
            f.write((" ".join([f"{u} {v} {pid}" for (u,v,pid) in pts]) if pts else "") + "\n")

def _finite_mask(X):
    X = np.asarray(X)
    return np.isfinite(X).all(axis=1)

def write_colmap_points3D_txt(
    path: Path,
    sfm,
    image_id_map: dict,
    point_errors: np.ndarray | None = None,
    *,
    normalize: bool = False
):
    X = np.asarray(sfm.points_3d, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError("points_3d must be (N,3)")

    # 1) 유한성 필터
    fin = _finite_mask(X)
    Xk = X[fin]
    old_ids = np.nonzero(fin)[0]

    # 2) (선택) 보기용 정규화 (중앙=median, 크기=5~95% bbox 길이)
    if normalize and len(Xk) > 0:
        ctr = np.median(Xk, axis=0)
        span = np.percentile(Xk, 95, axis=0) - np.percentile(Xk, 5, axis=0)
        s = float(np.linalg.norm(span))
        if s == 0: s = 1.0
        Xk = (Xk - ctr) / s

    # 3) old pid -> new pid 재매핑 (1-based)
    old2new = {int(o): i+1 for i, o in enumerate(old_ids)}

    # 4) 트랙 구성 (관측/point_tracks에서 new pid로 매핑)
    track_map = {}
    has_kp = hasattr(sfm, "point_tracks") and isinstance(sfm.point_tracks, dict)
    if has_kp:
        for (cam_idx, kp_idx), pid0 in sfm.point_tracks.items():
            pid0 = int(pid0)
            if pid0 not in old2new:  # 드롭된 포인트
                continue
            img_id = image_id_map.get(cam_idx, None)
            if img_id is None: 
                continue
            pid = old2new[pid0]
            track_map.setdefault(pid, []).append((img_id, int(kp_idx)))

    for (cam_idx, pid0, u, v) in sfm.observations:
        pid0 = int(pid0)
        if pid0 not in old2new:
            continue
        img_id = image_id_map.get(cam_idx, None)
        if img_id is None:
            continue
        pid = old2new[pid0]
        lst = track_map.setdefault(pid, [])
        if all(img_id != t[0] for t in lst):
            lst.append((img_id, -1))

    # 5) 파일 쓰기
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(Xk)}\n")
        for i, (pid0, X3) in enumerate(zip(old_ids, Xk)):
            pid = i+1  # 1-based
            x, y, z = map(float, X3[:3])
            err = 0.0
            if point_errors is not None and pid0 < len(point_errors) and np.isfinite(point_errors[pid0]):
                err = float(point_errors[pid0])
            R = G = B = 255
            tr = track_map.get(old2new[int(pid0)], [])
            tr_str = " ".join([f"{img} {kp}" for (img, kp) in tr])
            f.write(f"{pid} {x} {y} {z} {R} {G} {B} {err} {tr_str}\n")

# ---------- (C) 내보내기(자동 포즈탐지 + 선택적 정규화 + PLY 직출) ----------
def export_colmap_text(out_dir: Path, sfm, K: np.ndarray, image_wh: tuple[int,int],
                       dist: np.ndarray | None, matches_data: list,
                       point_errors: np.ndarray | None = None,
                       *, pose_kind: int | None = None,
                       normalize_for_view: bool = False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 0) 포즈 해석 자동 판별
    if pose_kind is None:
        best, score, allr = autodetect_pose_convention(sfm.cameras, sfm.points_3d)
        print(f"[export] autodetect pose_kind={best} (cheirality={score:.3f}, all={np.round(allr,3)})")
        pose_kind = best

    # 1) cameras.txt
    cam_id = 1
    W,H = image_wh
    write_colmap_cameras_txt(out_dir/"cameras.txt", cam_id, W, H, K, dist)

    # 2) images.txt
    int_cam_indices = [i for i,(R,t) in enumerate(sfm.cameras) if R is not None]
    image_id_map = {ci: (k+1) for k,ci in enumerate(sorted(int_cam_indices))}
    ext2name = build_extid_to_name(matches_data)
    write_colmap_images_txt(out_dir/"images.txt", sfm, image_id_map, cam_id, ext2name, pose_kind)

    # 3) points3D.txt (옵션: 보기 좋게 정규화)
    X = sfm.points_3d
    if normalize_for_view and len(X) > 0:
        med = np.median(X, axis=0)
        scale = np.linalg.norm(np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0))
        scale = 1.0 if scale == 0 else scale
        Xn = (X - med) / scale
        class _Tmp: pass
        tmp = _Tmp(); tmp.points_3d = Xn; tmp.point_tracks = getattr(sfm, "point_tracks", {})
        tmp.observations = sfm.observations; tmp.cameras = sfm.cameras; tmp.inv_view_id_map = sfm.inv_view_id_map
        write_colmap_points3D_txt(out_dir/"points3D.txt", tmp, image_id_map, point_errors)
    else:
        write_colmap_points3D_txt(out_dir/"points3D.txt", sfm, image_id_map, point_errors)

def export_ply(path: Path, points_3d: np.ndarray, colors: np.ndarray | None = None):
    """빠르게 Meshlab 확인용 PLY 직출."""
    path = Path(path)
    P = np.asarray(points_3d, np.float64)
    if colors is None or len(colors) != len(P):
        colors = np.full((len(P),3), 200, dtype=np.uint8)
    header = [
        "ply", "format ascii 1.0",
        f"element vertex {len(P)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header)+"\n")
        for (x,y,z), (r,g,b) in zip(P, colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
