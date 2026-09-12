"""
Entry-point for SfM v4 (custom PnP+RANSAC+Triangulation). Logic mirrors main_2.py
but imports SfMPipeline from sfm_4.
"""
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict

from config import *
from utils import ensure_dir
from utils import compute_observation_errors, aggregate_colmap_point_errors
from utils_2 import export_ply
from frame_extraction import extract_frames
from features import FeatureBackbone, match_consecutive, load_features
from sfm_4 import SfMPipeline
from global_ba import global_bundle_adjustment
from vis_open3d import SfMVisualizerO3D, ViewSpec

def estimate_camera_intrinsics(width_px: int, height_px: int, *, principal_at_center: bool = True) -> np.ndarray:
    W, H = float(width_px), float(height_px)
    f = 1.2 * max(W, H)
    cx = W * 0.5; cy = H * 0.5
    return np.array([[f, 0.0, cx],[0.0, f, cy],[0.0, 0.0, 1.0]], dtype=np.float64)

def load_matches_data(matches_dir: Path, features_dir: Path) -> list:
    match_files = sorted(matches_dir.glob("*.npz"))
    matches_data = []
    all_stems = set(); tmp = []
    for f in match_files:
        z = np.load(f)
        f1 = Path(str(z['f1'])).stem; f2 = Path(str(z['f2'])).stem
        all_stems.add(f1); all_stems.add(f2)
        tmp.append((f, f1, f2))
    stem2id = {stem: vid for vid, stem in enumerate(sorted(all_stems))}
    kp_cache = {}
    def get_kp(stem):
        if stem not in kp_cache:
            kps, desc = load_features(features_dir / f"{stem}.npz")
            kp_cache[stem] = (kps[:, :2].astype(np.float32), desc)
        return kp_cache[stem][0], kp_cache[stem][1]
    for match_file, f1_stem, f2_stem in tmp:
        data = np.load(match_file)
        qidx = data['qidx'].astype(np.int32); tidx = data['tidx'].astype(np.int32)
        kps1, _ = get_kp(f1_stem); kps2, _ = get_kp(f2_stem)
        valid = (qidx >= 0) & (qidx < len(kps1)) & (tidx >= 0) & (tidx < len(kps2))
        if not np.any(valid): continue
        qidx = qidx[valid]; tidx = tidx[valid]
        pairs = np.unique(np.c_[qidx, tidx], axis=0).astype(np.int32)
        pts1 = kps1[pairs[:, 0]]; pts2 = kps2[pairs[:, 1]]
        if 'distance' in data:
            dist = np.asarray(data['distance'])[valid]; avg_distance = float(np.mean(dist)) if len(dist) else 0.0
        else:
            avg_distance = 0.0
        matches_data.append({
            'view1_id': stem2id[f1_stem], 'view2_id': stem2id[f2_stem],
            'f1': Path(str(data['f1'])), 'f2': Path(str(data['f2'])),
            'pairs': pairs, 'pts1': pts1.astype(np.float32), 'pts2': pts2.astype(np.float32),
            'kps1_all': kps1.astype(np.float32), 'kps2_all': kps2.astype(np.float32),
            'num_matches': len(pairs), 'avg_distance': avg_distance,
            'quality_score': len(pairs) / (1.0 + avg_distance),
        })
    matches_data.sort(key=lambda x: x['quality_score'], reverse=True)
    return matches_data

def run_sfm_pipeline():
    print("=== SfM v4 (custom PnP/RANSAC/Triangulation) ===")
    frames_dir = OUT_DIR / "frames"; feats_dir = OUT_DIR / "features"; matches_dir = OUT_DIR / "matches"
    viz_dir = OUT_DIR / "viz"; results_dir = OUT_DIR / "results"
    frames_exist = frames_dir.exists() and any(frames_dir.glob("*.jpg"))
    feats_exist = feats_dir.exists() and any(feats_dir.glob("*.npz"))
    matches_exist = matches_dir.exists() and any(matches_dir.glob("*.npz"))
    if not (frames_exist and feats_exist and matches_exist):
        video_path = r"D:\\대학교\\3-2\\기초컴퓨터비전이론및응용\\assignment_02\\data\\IMG_3592.MOV"
        frames_dir = ensure_dir(frames_dir); feats_dir = ensure_dir(feats_dir)
        matches_dir = ensure_dir(matches_dir); viz_dir = ensure_dir(viz_dir); results_dir = ensure_dir(results_dir)
        print(f"Processing video: {video_path}")
        frames = extract_frames(video_path, frames_dir, target_fps=TARGET_FPS)
        backbone = FeatureBackbone(METHOD)
        match_consecutive(backbone, frames, feats_dir, matches_dir, viz_dir, ratio=RATIO, crosscheck=CROSSCHECK)
    else:
        frames = sorted(frames_dir.glob("*.jpg"))

    print("\n3. Estimating camera intrinsics...")
    W, H = 1920, 1080
    K0 = estimate_camera_intrinsics(W, H)
    print(K0)

    print("\n4. Loading matches data...")
    matches_data = load_matches_data(matches_dir, feats_dir)
    print(f"Loaded {len(matches_data)} frame pairs")

    print("\n5. Running Structure from Motion...")
    sfm = SfMPipeline(confidence=CONFIDENCE, min_matches=MIN_MATCHES)
    sfm.set_intrinsics(K0)

    print("Selecting a valid seed pair...")
    ok_init = False; seed_matches = None
    for m in matches_data[:50]:
        if int(m['view1_id']) == int(m['view2_id']): continue
        if sfm.initialize_from_two_view(m):
            if len(sfm.cameras) >= 2 and len(sfm.inv_view_id_map) >= 2:
                ok_init = True; seed_matches = m; print(f"Seeded with {m['f1'].name} -> {m['f2'].name}"); break
    if not ok_init:
        print("Failed to find a valid two-view seed."); return

    sfm.build_index(matches_data)

    print("\n[Incremental SfM] Entering registration loop")
    attempts = defaultdict(int); bad_images = set(); deferred = set()
    remaining = set()
    for m in matches_data:
        remaining.add(m['view1_id']); remaining.add(m['view2_id'])
    remaining -= set(sfm.inv_view_id_map.keys())

    # optional: re-seed helper (copy of main_2 logic, simplified)
    def try_reseed_from_remaining(sfm, matches_data, remaining):
        remain_set = set(remaining)
        for m in matches_data:
            v1, v2 = int(m['view1_id']), int(m['view2_id'])
            if (v1 in remain_set) and (v2 in remain_set) and (v1 != v2):
                if sfm.initialize_from_two_view(m):
                    # build index again to include new seed
                    sfm.build_index(matches_data)
                    if v1 in remaining: remaining.remove(v1)
                    if v2 in remaining: remaining.remove(v2)
                    return True
        return False

    while True:
        if not remaining:
            print("[Loop] No remaining images. End."); break
        cand_ext, scores = sfm.select_next_best_image(
            matches_data,
            min_common_points=6,
            min_spread_px=10.0,
            exclude_ids=bad_images | deferred
        )
        if cand_ext is None:
            print("[Loop] No suitable next image. Trying re-seed on remaining images...")
            if try_reseed_from_remaining(sfm, matches_data, remaining):
                continue
            print("[Loop] Re-seed failed. Ending incremental SfM.")
            break
        print(f"[Loop] Next image: view_id={cand_ext}")
        pkg = sfm.build_pnp_correspondences(sfm, cand_ext, matches_data)
        if (pkg is None) or (len(pkg[0]) < 8):
            attempts[cand_ext] += 1
            if attempts[cand_ext] >= 3: bad_images.add(cand_ext); deferred.discard(cand_ext)
            else: deferred.add(cand_ext)
            continue
        sfm.pending_pnp[cand_ext] = {"pts3d": pkg[0], "pts2d": pkg[1], "pid_list": pkg[2], "kp_list": pkg[3]}
        ok_reg = sfm.register_image_via_pnp(cand_ext, ransac_reproj_err=6.0, ransac_conf=0.9995, ransac_iters=12000, min_inliers=8)
        if not ok_reg:
            attempts[cand_ext] += 1
            if attempts[cand_ext] >= 3: bad_images.add(cand_ext); deferred.discard(cand_ext)
            else: deferred.add(cand_ext)
            continue
        add_pts, attached_obs = sfm.triangulate_new_matches_from_view(cand_ext, matches_data, reproj_thresh=5.0)
        print(f"[Loop]Triangulated: +{add_pts} points, attached: {attached_obs} observations")

        n_cam, n_pts, K_new = global_bundle_adjustment(sfm, sfm.K, huber_delta=3.0, max_nfev=80, fix_anchor=True, fix_point=False)
        sfm.set_intrinsics(K_new)
        if cand_ext in remaining: remaining.discard(cand_ext)
        attempts.pop(cand_ext, None); deferred.discard(cand_ext); bad_images.discard(cand_ext)

    # -------- Evaluation metrics (like main_2.py) --------
    print("\n6. Computing reprojection error report...")
    obs_errs = compute_observation_errors(sfm.cameras, sfm.points_3d, sfm.observations, sfm.K)
    pp_mean, pp_median, pp_count, colmap_like = aggregate_colmap_point_errors(
        n_points=len(sfm.points_3d),
        obs4=sfm.observations,
        obs_errs=obs_errs
    )
    P = np.asarray(sfm.points_3d)
    finite = np.isfinite(P).all(axis=1) if len(P) else np.array([], dtype=bool)
    # build per-point robust stats
    nP = len(P)
    per_point_mean = np.full(nP, np.nan, dtype=np.float64)
    per_point_p90  = np.full(nP, np.nan, dtype=np.float64)
    per_point_max  = np.full(nP, np.nan, dtype=np.float64)
    from collections import defaultdict as _dd
    _errs_by_pid = _dd(list)
    for i, (_, pid, _, _) in enumerate(sfm.observations):
        if 0 <= pid < nP:
            e = float(obs_errs[i])
            if np.isfinite(e):
                _errs_by_pid[pid].append(e)
    for pid, lst in _errs_by_pid.items():
        a = np.asarray(lst, dtype=np.float64)
        if a.size:
            per_point_mean[pid] = np.mean(a)
            per_point_p90[pid]  = np.percentile(a, 90)
            per_point_max[pid]  = np.max(a)

    # geometric filtering
    if nP:
        lo, hi = np.quantile(P[finite], [0.01, 0.99], axis=0) if finite.any() else (np.min(P,0), np.max(P,0))
        in_pct = np.all((P >= lo) & (P <= hi), axis=1) if nP else np.array([], bool)
        med = np.median(P[finite], axis=0) if finite.any() else np.zeros(3)
        mad = np.median(np.abs(P[finite] - med), axis=0) + 1e-9 if finite.any() else np.ones(3)
        in_mad = np.all(np.abs(P - med) <= 6 * 1.4826 * mad, axis=1)
        # error thresholds
        thr_med = 2.5; thr_mean = 3.5; thr_p90 = 5.0; thr_max = 10.0
        per_point_err = pp_median if (isinstance(pp_median, np.ndarray) and len(pp_median)==nP) else np.full(nP, np.nan)
        err_ok = (
            np.isfinite(per_point_err)  & (per_point_err  < thr_med) &
            np.isfinite(per_point_mean) & (per_point_mean < thr_mean) &
            np.isfinite(per_point_p90)  & (per_point_p90  < thr_p90) &
            np.isfinite(per_point_max)  & (per_point_max  < thr_max)
        )
        keep_eval = finite & in_pct & in_mad & err_ok
        if keep_eval.sum() == 0:
            err_ok = np.isfinite(per_point_err) & (per_point_err < 8.0)
            keep_eval = finite & in_pct & in_mad & err_ok
    else:
        keep_eval = np.array([], dtype=bool)

    # COLMAP-like summary on filtered points if available
    if keep_eval.sum() > 0 and isinstance(pp_mean, np.ndarray) and len(pp_mean)==nP:
        vals = pp_mean[keep_eval]
        colmap_like_f = {
            'count': int(keep_eval.sum()),
            'mean': float(np.nanmean(vals)),
            'median': float(np.nanmedian(vals)),
            'rmse': float(np.sqrt(np.nanmean(vals**2))),
            'p90': float(np.nanpercentile(vals, 90)),
            'max': float(np.nanmax(vals)),
        }
        print("[COLMAP-like point_avg FILTERED]", colmap_like_f)

    # -------- Open3D visualization (COLMAP-style) --------
    if P.size > 0:
        finite = np.isfinite(P).all(axis=1)
        if finite.any():
            lo, hi = np.quantile(P[finite], [0.01, 0.99], axis=0)
            in_pct = np.all((P >= lo) & (P <= hi), axis=1)
            keep = finite & in_pct
        else:
            keep = finite
        P_vis = P[keep] if keep.any() else P

        viz = SfMVisualizerO3D()
        viz.set_background((1,1,1))
        viz.add_point_cloud(
            P_vis,
            every=1,
            sample_max=100_000,
            clip_quantile=0.98,
            color="#666666",
            scalars=P_vis[:,2] if len(P_vis)>0 else None,
            cmap="turbo"
        )

        # order cameras by frame order
        frames = sorted((OUT_DIR/"frames").glob('*.jpg'))
        stem2idx = {Path(f).stem: i for i, f in enumerate(frames)}
        cams_with_frame = []
        for i, (R, t) in enumerate(sfm.cameras):
            ext = sfm.inv_view_id_map.get(i, None)
            if ext is None or ext >= len(frames):
                continue
            stem = str(frames[ext].stem)
            order = stem2idx.get(stem, ext)
            cams_with_frame.append((order, (R.copy(), t.reshape(3,))))
        cams_with_frame.sort(key=lambda x: x[0])
        cams_ordered = [rt for _, rt in cams_with_frame]

        # convert world2cam [R|t] to cam2world for both path and frustums
        cams_c2w = []
        for (R, t) in cams_ordered:
            Rcw = R.T
            tcw = (-R.T @ t.reshape(3,)).reshape(3,)
            cams_c2w.append((Rcw, tcw))

        # draw path and frustums
        if cams_c2w:
            viz.add_camera_path(cams_c2w, pose_type="cam2world", color=(1,0,0), normalize=True, max_len_ratio=10.0)
            viz.add_cameras(cams_c2w, K=sfm.K, img_size=(1920,1080), scale=0.02, color=(1.0,0.0,0.0), show_axes=False, filled=True)

        viz.show(view=ViewSpec(20, -60, 1.6), width=1600, height=1200)

    print("\n[Done]")

if __name__ == "__main__":
    run_sfm_pipeline()


