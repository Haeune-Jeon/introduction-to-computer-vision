"""
Feature detection, description, and matching functionality.
"""
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
import numpy as np


class FeatureBackbone:
    """
    Feature detection and description backbone supporting SIFT and ORB.
    """
    
    def __init__(self, method: str = "sift"):
        """
        Initialize feature detector.
        
        Args:
            method: Feature detection method ("sift" or "orb")
            
        Raises:
            RuntimeError: If SIFT is not available in OpenCV build
            ValueError: If method is not supported
        """
        if method == "sift":
            if not hasattr(cv, "SIFT_create"):
                raise RuntimeError("SIFT is not available in your OpenCV build. Switch METHOD='orb'.")
            self.det = cv.SIFT_create()
            self.norm = cv.NORM_L2
            self.name = "sift"
        elif method == "orb":
            self.det = cv.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
            self.norm = cv.NORM_HAMMING
            self.name = "orb"
        else:
            raise ValueError("method must be one of {sift, orb}")

    def detect_and_compute(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect keypoints and compute descriptors for a grayscale image.
        
        Args:
            gray: Grayscale input image
            
        Returns:
            Tuple of (keypoints_array, descriptors)
        """
        kps, desc = self.det.detectAndCompute(gray, None)
        kp_arr = np.array(
            [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps],
            dtype=np.float32,
        )
        return kp_arr, desc


def save_features(out_features_dir: Path, frame_path: Path, kp_arr: np.ndarray, desc: np.ndarray) -> Path:
    """
    Save detected features to disk.
    
    Args:
        out_features_dir: Directory to save features
        frame_path: Path to the original frame
        kp_arr: Keypoints array
        desc: Descriptors array
        
    Returns:
        Path: Path to saved features file
    """
    out = out_features_dir / (frame_path.stem + ".npz")
    np.savez_compressed(out, keypoints=kp_arr, descriptors=desc)
    return out


def load_features(features_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features from disk.
    
    Args:
        features_path: Path to features file
        
    Returns:
        Tuple of (keypoints_array, descriptors)
    """
    data = np.load(features_path)
    return data['keypoints'], data['descriptors']


def draw_and_save_matches(img1: np.ndarray, kps1: np.ndarray, img2: np.ndarray, kps2: np.ndarray, 
                         matches_good, out_path: Path, max_viz: int = 200) -> None:
    """
    Draw and save feature matches visualization.
    
    Args:
        img1: First image
        kps1: Keypoints from first image
        img2: Second image
        kps2: Keypoints from second image
        matches_good: List of good matches
        out_path: Output path for visualization
        max_viz: Maximum number of matches to visualize
    """
    def to_cv2kps(kp_arr: np.ndarray):
        """Convert keypoint array to OpenCV KeyPoint objects."""
        out = []
        for x, y, size, ang, resp, octv, cid in kp_arr:
            out.append(cv.KeyPoint(float(x), float(y), float(size), float(ang), float(resp), int(octv), int(cid)))
        return out

    kps1_cv = to_cv2kps(kps1)
    kps2_cv = to_cv2kps(kps2)
    draw_matches = matches_good[:max_viz]
    img_m = cv.drawMatches(img1, kps1_cv, img2, kps2_cv, draw_matches, None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(str(out_path), img_m)


def match_consecutive(backbone: FeatureBackbone, frames: List[Path], feats_dir: Path, 
                    matches_dir: Path, viz_dir: Path, ratio: float = 0.75, 
                    crosscheck: bool = True) -> None:
    """
    Match features between consecutive frames.
    
    Args:
        backbone: Feature detection backbone
        frames: List of frame paths
        feats_dir: Directory to save/load features
        matches_dir: Directory to save matches
        viz_dir: Directory to save visualizations
        ratio: Lowe ratio for feature matching
        crosscheck: Enable symmetric matching
    """
    bf = cv.BFMatcher(backbone.norm, crossCheck=False)

    cache = {}
    def load_or_compute(frame_path: Path):
        """Load features from cache or compute and save them."""
        if frame_path in cache:
            return cache[frame_path]
        gray = cv.imread(str(frame_path), cv.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read {frame_path}")
        kp_arr, desc = backbone.detect_and_compute(gray)
        save_features(feats_dir, frame_path, kp_arr, desc)
        color = cv.imread(str(frame_path), cv.IMREAD_COLOR)
        cache[frame_path] = (gray, color, kp_arr, desc)
        return cache[frame_path]

    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        img1, color1, kps1, d1 = load_or_compute(f1)
        img2, color2, kps2, d2 = load_or_compute(f2)

        # Check if descriptors are valid
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            print(f"[warn] No descriptors in pair {f1.name} - {f2.name}")
            continue

        # KNN matching with Lowe ratio test
        knn12 = bf.knnMatch(d1, d2, k=2)
        good12 = []
        for match_set in knn12:
            if len(match_set) == 2:
                m, n = match_set
                if m.distance < ratio * n.distance:
                    good12.append(m)
            elif len(match_set) == 1:
                # If only one match, use it directly
                good12.append(match_set[0])

        # Cross-check for symmetric matching
        if crosscheck:
            knn21 = bf.knnMatch(d2, d1, k=2)
            good21 = []
            for match_set in knn21:
                if len(match_set) == 2:
                    m, n = match_set
                    if m.distance < ratio * n.distance:
                        good21.append(m)
                elif len(match_set) == 1:
                    # If only one match, use it directly
                    good21.append(match_set[0])
            
            idx21 = {(m.queryIdx, m.trainIdx) for m in good21}
            good = [m for m in good12 if (m.trainIdx, m.queryIdx) in idx21]
        else:
            good = good12

        # Save matches
        pair_name = f"pair_{f1.stem}_{f2.stem}"
        out_npz = matches_dir / f"{pair_name}.npz"
        qidx = np.array([m.queryIdx for m in good], dtype=np.int32)
        tidx = np.array([m.trainIdx for m in good], dtype=np.int32)
        dist = np.array([m.distance for m in good], dtype=np.float32)
        np.savez_compressed(out_npz, qidx=qidx, tidx=tidx, distance=dist, 
                          f1=str(f1), f2=str(f2), method=backbone.name)

        # Save visualization
        viz_path = viz_dir / f"matches_{f1.stem}_{f2.stem}.jpg"
        draw_and_save_matches(color1, kps1, color2, kps2, good, viz_path)
        print(f"Saved {len(good)} matches → {viz_path}")


def match_all_pairs(
    backbone: FeatureBackbone,
    frames: List[Path],
    feats_dir: Path,
    matches_dir: Path,
    viz_dir: Path,
    *,
    ratio: float = 0.75,
    crosscheck: bool = True,
    use_flann: bool = True,          
    min_gap: int = 1,               
    max_gap: int | None = None,      
    max_pairs: int | None = None,    
    max_matches_per_pair: int = 6000
) -> None:
    
    # --- Matcher ---
    if use_flann and backbone.name == "sift":
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE=1
        search_params = dict(checks=64)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        knn_ok = True
    else:
        matcher = cv.BFMatcher(backbone.norm, crossCheck=False)
        knn_ok = True

    cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    def load_or_compute(frame_path: Path):
        if frame_path in cache:
            return cache[frame_path]
        gray = cv.imread(str(frame_path), cv.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read {frame_path}")
        kp_arr, desc = backbone.detect_and_compute(gray)
        save_features(feats_dir, frame_path, kp_arr, desc)
        color = cv.imread(str(frame_path), cv.IMREAD_COLOR)
        cache[frame_path] = (gray, color, kp_arr, desc)
        return cache[frame_path]

    N = len(frames)
    pair_count = 0
    for i in range(N - 1):
        img1, color1, kps1, d1 = load_or_compute(frames[i])
        if d1 is None or len(d1) == 0:
            continue
        
        for j in range(i + 1, N):
            gap = j - i
            if gap < min_gap:
                continue
            if (max_gap is not None) and (gap > max_gap):
                continue
            if (max_pairs is not None) and (pair_count >= max_pairs):
                break

            img2, color2, kps2, d2 = load_or_compute(frames[j])
            if d2 is None or len(d2) == 0:
                continue

            # --- KNN + ratio test ---
            if not knn_ok:
                continue
            knn12 = matcher.knnMatch(d1, d2, k=2)
            good12 = []
            for ms in knn12:
                if len(ms) == 2:
                    m, n = ms
                    if m.distance < ratio * n.distance:
                        good12.append(m)
                elif len(ms) == 1:
                    good12.append(ms[0])

            # --- Cross-check ---
            if crosscheck:
                knn21 = matcher.knnMatch(d2, d1, k=2)
                good21 = []
                for ms in knn21:
                    if len(ms) == 2:
                        m, n = ms
                        if m.distance < ratio * n.distance:
                            good21.append(m)
                    elif len(ms) == 1:
                        good21.append(ms[0])
                idx21 = {(m.queryIdx, m.trainIdx) for m in good21}
                good = [m for m in good12 if (m.trainIdx, m.queryIdx) in idx21]
            else:
                good = good12

            if len(good) == 0:
                continue

            if len(good) > max_matches_per_pair:
                good.sort(key=lambda g: g.distance)
                good = good[:max_matches_per_pair]

            # --- save ---
            pair_name = f"pair_{frames[i].stem}_{frames[j].stem}"
            out_npz = matches_dir / f"{pair_name}.npz"
            qidx = np.array([m.queryIdx for m in good], dtype=np.int32)
            tidx = np.array([m.trainIdx for m in good], dtype=np.int32)
            dist = np.array([m.distance for m in good], dtype=np.float32)
            np.savez_compressed(
                out_npz, qidx=qidx, tidx=tidx, distance=dist,
                f1=str(frames[i]), f2=str(frames[j]), method=backbone.name
            )

            # --- visualization ---
            viz_path = viz_dir / f"matches_{frames[i].stem}_{frames[j].stem}.jpg"
            draw_and_save_matches(color1, kps1, color2, kps2, good, viz_path, max_viz=200)

            pair_count += 1
            print(f"[all-pairs] {frames[i].name} ↔ {frames[j].name}  matches={len(good)}  (gap={gap})")

        if (max_pairs is not None) and (pair_count >= max_pairs):
            break
