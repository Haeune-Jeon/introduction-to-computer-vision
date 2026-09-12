"""
Frame extraction from video files.
"""
from pathlib import Path
from typing import List
import cv2 as cv
from utils import ensure_dir, imwrite


def extract_frames(video_path: Path, out_frames_dir: Path, target_fps: float = 2.0) -> List[Path]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video
        out_frames_dir: Directory to save extracted frames
        target_fps: Target frames per second for extraction
        
    Returns:
        List of paths to extracted frame files
        
    Raises:
        FileNotFoundError: If video cannot be opened
        RuntimeError: If not enough frames are extracted
    """
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # Get source FPS and calculate step size
    src_fps = cap.get(cv.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0  # Default FPS if unknown
    step = max(int(round(src_fps / max(target_fps, 1e-6))), 1)
    
    frame_paths: List[Path] = []
    idx, saved_idx = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx % step == 0:
            fname = out_frames_dir / f"{saved_idx:06d}.jpg"
            imwrite(fname, frame)
            frame_paths.append(fname)
            saved_idx += 1
        idx += 1
    
    cap.release()
    
    if len(frame_paths) < 2:
        raise RuntimeError("Not enough frames extracted (need ≥ 2). Try a longer video or higher TARGET_FPS.")
    
    return frame_paths
