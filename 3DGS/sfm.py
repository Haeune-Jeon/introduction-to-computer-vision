import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

class ColmapSfM:
    def __init__(self, data_root: str = "data"):
        """
        Initialize COLMAP SfM pipeline
        
        Args:
            data_root: Root directory containing video folders (object_1, object_2, scene)
        """
        self.data_root = Path(data_root)
        self.fps = 7.5 # Extraction frame rate
        self.skip_seconds = 3  # Skip first 3 seconds (student ID verification)
        self.skip_frame_extraction = True  # Set to True to skip frame extraction
        self.skip_feature_extraction = False  # Set to True to skip feature extraction
        self.skip_feature_matching = False  # Set to True to skip feature matching
        self.skip_mapper = False  # Set to True to skip sparse reconstruction (bundle adjustment)
        
    def extract_frames_from_video(self, video_path: str, output_frame_dir: str) -> int:
        """
        Extract frames from video, skipping the first 3 seconds
        
        Args:
            video_path: Path to input video
            output_frame_dir: Directory to save extracted frames
            
        Returns:
            Number of extracted frames
        """
        os.makedirs(output_frame_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = int(self.skip_seconds * fps)
        
        print(f"Video: {video_path}")
        print(f"  FPS: {fps}, Total frames: {frame_count}")
        print(f"  Skipping first {self.skip_seconds}s ({skip_frames} frames)")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        frame_idx = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as JPEG
            frame_name = f"{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_frame_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            
            extracted_count += 1
            frame_idx += 1
            
            if extracted_count % 100 == 0:
                print(f"  Extracted {extracted_count} frames...")
        
        cap.release()
        print(f"  Total extracted: {extracted_count} frames\n")
        
        return extracted_count
    
    def run_colmap_feature_extraction(self, image_dir: str, database_path: str):
        """Run COLMAP feature extraction with FAST settings"""
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.mask_path", "",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.max_image_size", "2000",  # 줄임 (4000->2000)
            "--SiftExtraction.max_num_features", "8000"  # 줄임 (12000->8000)
        ]
        
        print(f"Running COLMAP feature extraction...")
        try:
            subprocess.run(cmd, check=True)
            print("Feature extraction completed.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error during feature extraction: {e}\n")
            raise
    
    def run_colmap_feature_matching(self, database_path: str):
        """Run COLMAP feature matching with sequential matcher"""
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", "1",
            "--SequentialMatching.loop_detection", "0",
            "--SequentialMatching.overlap", "5",
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7"
        ]
        
        print(f"Running COLMAP feature matching...")
        try:
            subprocess.run(cmd, check=True)
            print("Feature matching completed.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error during feature matching: {e}\n")
            raise
    
    def run_colmap_mapper(self, database_path: str, image_dir: str, sparse_dir: str):
        """Run COLMAP mapper with FAST settings"""
        os.makedirs(sparse_dir, exist_ok=True)
        
        cmd = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
            "--Mapper.num_threads", str(os.cpu_count() or 4),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.max_num_models", "1",  # 1개 모델만
            "--Mapper.ba_global_max_num_iterations", "10",  # BA 반복 줄임
            "--Mapper.ba_local_max_num_iterations", "5"   # 로컬 BA 줄임
        ]
        
        print(f"Running COLMAP mapper...")
        print(f"  Note: ba_refine_extra_params=0 to prevent distortion parameters")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Sparse reconstruction completed.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error during mapping: {e}")
            if e.stdout:
                print(f"\nSTDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"\nSTDERR:\n{e.stderr}")
            raise
    
    def extract_camera_params(self, sparse_dir: str, output_json: str):
        """
        Extract camera intrinsic and extrinsic parameters from COLMAP output
        """
        recon_folders = [d for d in os.listdir(sparse_dir) 
                        if os.path.isdir(os.path.join(sparse_dir, d))]
        
        if not recon_folders:
            print("Warning: No reconstruction found\n")
            return None
        
        recon_path = os.path.join(sparse_dir, recon_folders[0])
        
        # First, try to convert binary to text if txt files don't exist
        cameras_file = os.path.join(recon_path, "cameras.txt")
        images_file = os.path.join(recon_path, "images.txt")
        
        if not os.path.exists(cameras_file):
            print("Text files not found. Converting binary to text format...")
            cmd = [
                "colmap", "model_converter",
                "--input_path", recon_path,
                "--output_path", recon_path,
                "--output_type", "TXT"
            ]
            try:
                subprocess.run(cmd, check=False)
                print("Conversion completed.\n")
            except Exception as e:
                print(f"Error converting binary to text: {e}\n")
        
        cameras_data = {}
        images_data = []
        
        # Parse cameras.txt
        try:
            with open(cameras_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    camera_id = parts[0]
                    camera_model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    
                    cameras_data[camera_id] = {
                        "model": camera_model,
                        "width": width,
                        "height": height,
                        "params": params
                    }
            print(f"Extracted {len(cameras_data)} camera(s)")
        except FileNotFoundError:
            print(f"Warning: cameras.txt not found at {cameras_file}")
            return None
        
        # Parse images.txt
        try:
            with open(images_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split()
                    if len(parts) < 10:  # Skip point data
                        image_id = parts[0]
                        qvec = [float(p) for p in parts[1:5]]  # qw, qx, qy, qz
                        tvec = [float(p) for p in parts[5:8]]   # tx, ty, tz
                        camera_id = parts[8]
                        image_name = parts[9]
                        
                        images_data.append({
                            "image_id": image_id,
                            "camera_id": camera_id,
                            "image_name": image_name,
                            "qvec": qvec,  # quaternion (w, x, y, z)
                            "tvec": tvec   # translation vector
                        })
            print(f"Extracted {len(images_data)} image pose(s)\n")
        except FileNotFoundError:
            print(f"Warning: images.txt not found at {images_file}")
            return None
        
        # Save to JSON
        output_data = {
            "cameras": cameras_data,
            "images": images_data
        }
        
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Camera parameters saved to {output_json}")
        return output_data
    
    def run_colmap_image_undistorter(self, image_dir: str, sparse_dir: str, undistorted_dir: str):
        """
        Run COLMAP image_undistorter to convert SIMPLE_RADIAL to PINHOLE/SIMPLE_PINHOLE
        This is required for 3D Gaussian Splatting compatibility
        """
        os.makedirs(undistorted_dir, exist_ok=True)
        
        # Find reconstruction folder (usually 0)
        recon_folders = [d for d in os.listdir(sparse_dir) 
                        if os.path.isdir(os.path.join(sparse_dir, d))]
        if not recon_folders:
            print("Warning: No reconstruction folder found in sparse_dir")
            return
        
        input_model = os.path.join(sparse_dir, recon_folders[0])
        
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", image_dir,
            "--input_path", input_model,
            "--output_path", undistorted_dir,
            "--output_type", "COLMAP"
        ]
        
        print("Running COLMAP image_undistorter...")
        print("  This converts SIMPLE_RADIAL to PINHOLE (3DGS-compatible format)")
        try:
            subprocess.run(cmd, check=True)
            print("Image undistortion completed.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error during undistortion: {e}\n")
            raise
    
    def run_colmap_full_pipeline(self, video_folder: str, folder_name: str):
        """
        Run complete COLMAP pipeline for a video folder
        
        Args:
            video_folder: Path to folder containing video
            folder_name: Name of the folder (e.g., 'object_1')
        """
        print(f"\n{'='*60}")
        print(f"Processing: {folder_name}")
        print(f"{'='*60}\n")
        
        # Setup directories - Save results inside the video folder
        result_dir = Path(video_folder) / "colmap_results"
        frames_dir = result_dir / "frames"
        database_path = str(result_dir / "database.db")
        sparse_dir = str(result_dir / "sparse")
        
        os.makedirs(result_dir, exist_ok=True)
        
        # Clean up old database if it exists AND we're not skipping feature steps
        if os.path.exists(database_path) and not (self.skip_feature_extraction and self.skip_feature_matching):
            print(f"Removing old database to regenerate with new camera model settings...")
            os.remove(database_path)
            print(f"Old database removed.\n")
        
        # Step 1: Extract frames or use existing ones
        print(f"Step 1: Checking frames...")
        # Check if we should extract frames
        if self.skip_frame_extraction and os.path.exists(str(frames_dir)) and len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) > 0:
            print(f"Using existing frames\n")
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        else:
            # Find and extract from video (overwrite existing frames)
            video_files = [f for f in os.listdir(video_folder) 
                          if f.lower().endswith(('.mov', '.mp4', '.avi', '.mkv'))]
            
            if not video_files:
                print(f"Error: No video file found in {video_folder}")
                return
            
            video_path = os.path.join(video_folder, video_files[0])
            print(f"Extracting frames from video (overwriting existing)...")
            frame_count = self.extract_frames_from_video(video_path, str(frames_dir))
            if frame_count == 0:
                print(f"Error: No frames extracted")
                return
        
        # Step 2: Feature extraction
        print(f"Step 2: Feature extraction...")
        if self.skip_feature_extraction:
            print(f"Skipping feature extraction\n")
        else:
            self.run_colmap_feature_extraction(str(frames_dir), database_path)
        
        # Step 3: Feature matching
        print(f"Step 3: Feature matching...")
        if self.skip_feature_matching:
            print(f"Skipping feature matching\n")
        else:
            self.run_colmap_feature_matching(database_path)
        
        # Step 4: Sparse reconstruction (mapper)
        print(f"Step 4: Sparse reconstruction...")
        if self.skip_mapper:
            print(f"Skipping sparse reconstruction (bundle adjustment)\n")
        else:
            self.run_colmap_mapper(database_path, str(frames_dir), sparse_dir)
        
        # Step 5: Image undistortion (convert to PINHOLE for 3DGS)
        undistorted_dir = str(result_dir / "undistorted")
        print(f"Step 5: Image undistortion...")
        self.run_colmap_image_undistorter(str(frames_dir), sparse_dir, undistorted_dir)
        
        # From now on, use undistorted sparse model
        undist_sparse_dir = os.path.join(undistorted_dir, "sparse")
        
        # Step 6: Convert binary to text format
        print(f"Step 6: Converting model to text format...")
        self.check_and_convert_binary_to_text(undist_sparse_dir)
        
        # Step 7: Extract and save camera parameters
        print(f"Step 7: Extracting camera parameters...")
        params_json = str(result_dir / "cameras_undistorted.json")
        self.extract_camera_params(undist_sparse_dir, params_json)
        
        # Step 8: Print camera model info
        print(f"Step 8: Checking camera model (from undistorted)...")
        self.print_camera_model_info(undist_sparse_dir)
        
        print(f"\n{'='*60}")
        print(f"Completed: {folder_name}")
        print(f"Results saved to: {result_dir}")
        print(f"{'='*60}\n")
    
    def process_all_folders(self):
        """Process all video folders in data directory"""
        folders = [d for d in os.listdir(self.data_root) 
                  if os.path.isdir(os.path.join(self.data_root, d))]
        
        print(f"\nStarting COLMAP SfM pipeline for {len(folders)} video folder(s)...\n")
        
        for folder_name in sorted(folders):
            video_folder = os.path.join(self.data_root, folder_name)
            try:
                self.run_colmap_full_pipeline(video_folder, folder_name)
            except Exception as e:
                print(f"\nError processing {folder_name}: {e}\n")
                continue
        
        print(f"\n{'='*60}")
        print(f"All processing completed!")
        print(f"Results saved in each folder's colmap_results directory")
        print(f"{'='*60}\n")
    
    def check_and_convert_binary_to_text(self, sparse_dir: str):
        """Check if binary files exist and convert them to text format"""
        recon_folders = [d for d in os.listdir(sparse_dir) 
                        if os.path.isdir(os.path.join(sparse_dir, d))]
        
        if not recon_folders:
            return False
        
        recon_path = os.path.join(sparse_dir, recon_folders[0])
        
        # Check if binary files exist
        bin_files = [f for f in os.listdir(recon_path) if f.endswith('.bin')]
        txt_files = [f for f in os.listdir(recon_path) if f.endswith('.txt')]
        
        if bin_files and not txt_files:
            print(f"Binary files found in {recon_path}")
            print(f"Converting binary to text format...")
            
            cmd = [
                "colmap", "model_converter",
                "--input_path", recon_path,
                "--output_path", recon_path,
                "--output_type", "TXT"
            ]
            
            try:
                subprocess.run(cmd, check=False)
                print(f"Binary to text conversion completed.\n")
                return True
            except Exception as e:
                print(f"Error converting binary to text: {e}\n")
                return False
        
        return False
    
    def print_camera_model_info(self, sparse_dir: str):
        """Print camera model information from cameras.txt"""
        recon_folders = [d for d in os.listdir(sparse_dir) 
                        if os.path.isdir(os.path.join(sparse_dir, d))]
        
        if not recon_folders:
            print("Warning: No reconstruction folder found")
            return
        
        recon_path = os.path.join(sparse_dir, recon_folders[0])
        cameras_file = os.path.join(recon_path, "cameras.txt")
        
        if not os.path.exists(cameras_file):
            print(f"Warning: cameras.txt not found at {cameras_file}")
            return
        
        print(f"\n{'='*60}")
        print(f"CAMERA MODEL INFO")
        print(f"{'='*60}")
        
        with open(cameras_file, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    camera_id = parts[0]
                    camera_model = parts[1]
                    print(f"Camera {camera_id}: {camera_model}")
                    print(f"  Full line: {line.strip()}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Initialize and run SfM pipeline
    sfm = ColmapSfM(data_root="data")
    
    print("\n" + "="*70)
    print("COLMAP SfM Pipeline for 3D Gaussian Splatting")
    print("="*70)
    print("\nSettings:")
    print("  ✓ Feature Extraction: SIFT on GPU with 12000 max features")
    print("  ✓ Feature Matching: Sequential matcher (video-friendly)")
    print("  ✓ Single Camera: Yes (same camera for all images)")
    print("  ✓ Mapper: Bundle adjustment with focal length & principal point refinement")
    print("="*70 + "\n")
    
    # Skip already completed steps
    sfm.skip_feature_extraction = False   # Set True to skip feature extraction
    sfm.skip_feature_matching = False     # Set True to skip feature matching
    sfm.skip_mapper = False                # Set True to skip mapper (bundle adjustment)
    
    # Choose which folders to process
    print("\n" + "="*70)
    print("FOLDER OPTIONS:")
    print("  1. Process all folders (object_1, object_2, scene, scene_2~scene_9)")
    print("  2. Process only object_1 (fastest for testing)")
    print("  3. Process object_1, object_2, scene (main 3)")
    print("="*70)
    
    # Modify this to choose which folders to process
    folders_to_process = "minion_2"  # Options: "all", "main", "object_1_only", "scene_3_onwards", "scene_4"
    
    if folders_to_process == "object_1_only":
        print("\nProcessing: object_1 only\n")
        sfm.run_colmap_full_pipeline("data/object_1", "object_1")
    elif folders_to_process == "main":
        print("\nProcessing: object_1, object_2, scene (main 3 folders)\n")
        for folder_name in ["object_1", "object_2", "scene"]:
            try:
                sfm.run_colmap_full_pipeline(f"data/{folder_name}", folder_name)
            except Exception as e:
                print(f"\nError processing {folder_name}: {e}\n")
                continue
        print("\n" + "="*70)
        print("Main folders processing completed!")
        print("="*70 + "\n")
    elif folders_to_process == "minion":
        print("\nProcessing: minion\n")
        sfm.run_colmap_full_pipeline("data/minion", "minion")
    elif folders_to_process == "minion_2":
        print("\nProcessing: minion_2 only (with optimizations)\n")
        sfm.run_colmap_full_pipeline("data/minion_2", "minion_2")
    elif folders_to_process == "scene_3_onwards":
        print("\nProcessing: scene_3, scene_4, scene_5, scene_6, scene_7, scene_8, scene_9\n")
        for folder_name in ["scene_3", "scene_4", "scene_5", "scene_6", "scene_7", "scene_8", "scene_9"]:
            try:
                sfm.run_colmap_full_pipeline(f"data/{folder_name}", folder_name)
            except Exception as e:
                print(f"\nError processing {folder_name}: {e}\n")
                continue
        print("\n" + "="*70)
        print("Remaining scenes processing completed!")
        print("="*70 + "\n")
    else:  # "all"
        print("\n⚠️  Processing ALL folders (15-20 hours estimated)\n")
        sfm.process_all_folders()
