"""
Final Composition Animation Script

Frame breakdown:
- 0-30: Scene only (camera 200→230)
- 31-90: Object2 (minion) appears and moves to position (camera 230)
- 91-160: Camera moves (230→300) with object2 fixed
- 161-220: Object1 (minion_2) appears and moves to position (camera 300)
- 221-280: Both objects grow and shrink (camera 300)
- 281-586: Depth-aware composition - camera 300→605, objects fixed

This script:
1. Generates animation.json with per-frame camera and object transforms
2. Renders animation from the JSON file (reproducible)
"""

import torch
import numpy as np
from scene import Scene, GaussianModel
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scipy.spatial.transform import Rotation as R

# Import compose logic
from compose import ComposedGaussianModel, load_gaussian_model_simple

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def interpolate_transform(start_transform, end_transform, alpha):
    """
    Interpolate between two transforms
    alpha: 0.0 = start, 1.0 = end
    """
    result = {}
    
    # Interpolate translation
    start_t = np.array(start_transform['translation'])
    end_t = np.array(end_transform['translation'])
    result['translation'] = list(start_t + alpha * (end_t - start_t))
    
    # Interpolate rotation
    start_r = np.array(start_transform['rotation'])
    end_r = np.array(end_transform['rotation'])
    result['rotation'] = list(start_r + alpha * (end_r - start_r))
    
    # Interpolate scale
    start_s = start_transform['scale']
    end_s = end_transform['scale']
    result['scale'] = start_s + alpha * (end_s - start_s)
    
    return result


def generate_animation_json(output_json_path):
    """
    Generate JSON file with per-frame camera and object transforms
    """
    print("=" * 80)
    print("GENERATING ANIMATION JSON")
    print("=" * 80)
    
    # Define object transforms (from compose.py)
    # Object 1 (minion_2_gaussian)
    obj1_start = {
        'translation': [1.2, -1.5, -1.5],
        'rotation': [-25, 50, -10],
        'scale': 1.4
    }
    obj1_end = {
        'translation': [1.2, -1.5, 0.55],
        'rotation': [-25, 50, -10],
        'scale': 1.4
    }
    
    # Object 2 (minion_gaussian)
    obj2_start = {
        'translation': [1.9, 0.0, 2.8],
        'rotation': [0, 30, 15],
        'scale': 0.9
    }
    obj2_end = {
        'translation': [1.9, 0.0, 1.3],
        'rotation': [0, 30, 15],
        'scale': 0.9
    }
    
    # Original animation: 0-280 frames (up to grow/shrink)
    # Depth-aware extension: 281 onwards, camera 300→605
    animation_end_frame = 280
    depth_aware_start_camera = 300
    depth_aware_end_camera = 605  # Sufficient to show depth-aware composition
    depth_aware_frames = depth_aware_end_camera - depth_aware_start_camera + 1  # 306 frames
    total_frames = animation_end_frame + depth_aware_frames  # 280 + 306 = 586
    
    animation_data = {
        "metadata": {
            "total_frames": total_frames,
            "fps": 30,
            "duration_seconds": total_frames / 30,
            "description": "Animation with object movements and depth-aware camera trajectory (300→605)",
            "animation_end_frame": animation_end_frame,
            "depth_aware_start_frame": animation_end_frame + 1,
            "depth_aware_frames": depth_aware_frames,
            "depth_aware_camera_range": f"{depth_aware_start_camera}-{depth_aware_end_camera}"
        },
        "frames": []
    }
    
    print(f"Calculating transforms for {total_frames} frames...")
    print(f"  Animation with objects: 0-{animation_end_frame}")
    print(f"  Depth-aware extension: {animation_end_frame+1}-{total_frames-1} (camera {depth_aware_start_camera}→{depth_aware_end_camera})")
    
    for frame_idx in tqdm(range(total_frames), desc="Generating JSON"):
        frame_data = {
            "frame": frame_idx,
            "camera_index": None,
            "objects": []
        }
        
        # Frame 0-30: Scene only, camera 200→230
        if 0 <= frame_idx <= 30:
            frame_data["camera_index"] = 200 + frame_idx
        
        # Frame 31-90: Object2 appears, camera 230
        elif 31 <= frame_idx <= 90:
            frame_data["camera_index"] = 230
            
            # Interpolate object2 from start to end
            alpha = (frame_idx - 31) / (90 - 31)
            obj2_transform = interpolate_transform(obj2_start, obj2_end, alpha)
            frame_data["objects"].append({
                "name": "object2_minion",
                "visible": True,
                "transform": obj2_transform
            })
        
        # Frame 91-160: Camera 230→300, object2 fixed at end position
        elif 91 <= frame_idx <= 160:
            cam_progress = (frame_idx - 91) / (160 - 91)
            cam_idx = int(230 + cam_progress * (300 - 230))
            frame_data["camera_index"] = cam_idx
            
            # Object2 at end position
            frame_data["objects"].append({
                "name": "object2_minion",
                "visible": True,
                "transform": obj2_end
            })
        
        # Frame 161-220: Object1 appears, camera 300
        elif 161 <= frame_idx <= 220:
            frame_data["camera_index"] = 300
            
            # Object2 at end position
            frame_data["objects"].append({
                "name": "object2_minion",
                "visible": True,
                "transform": obj2_end
            })
            
            # Interpolate object1 from start to end
            alpha = (frame_idx - 161) / (220 - 161)
            obj1_transform = interpolate_transform(obj1_start, obj1_end, alpha)
            frame_data["objects"].append({
                "name": "object1_minion_2",
                "visible": True,
                "transform": obj1_transform
            })
        
        # Frame 221-280: Both objects grow and shrink (translation x changes)
        elif 221 <= frame_idx <= 280:
            frame_data["camera_index"] = 300
            
            # Calculate growth factor (0→0.5→0)
            progress = (frame_idx - 221) / (280 - 221)
            # Create a smooth curve: 0 at start, peak at middle, 0 at end
            growth = 0.5 * np.sin(progress * np.pi)
            
            # Object2 with modified x
            obj2_modified = {
                'translation': [obj2_end['translation'][0] + growth, 
                               obj2_end['translation'][1], 
                               obj2_end['translation'][2]],
                'rotation': obj2_end['rotation'],
                'scale': obj2_end['scale']
            }
            frame_data["objects"].append({
                "name": "object2_minion",
                "visible": True,
                "transform": obj2_modified
            })
            
            # Object1 with modified x
            obj1_modified = {
                'translation': [obj1_end['translation'][0] + growth, 
                               obj1_end['translation'][1], 
                               obj1_end['translation'][2]],
                'rotation': obj1_end['rotation'],
                'scale': obj1_end['scale']
            }
            frame_data["objects"].append({
                "name": "object1_minion_2",
                "visible": True,
                "transform": obj1_modified
            })
        
        # Frame 281+: Depth-aware composition - camera 300→771, objects FIXED
        elif frame_idx > 280:
            # Camera moves from 300 to 771
            cam_offset = frame_idx - 281
            cam_idx = depth_aware_start_camera + cam_offset
            frame_data["camera_index"] = cam_idx
            
            # Both objects at end positions (FIXED)
            frame_data["objects"].append({
                "name": "object2_minion",
                "visible": True,
                "transform": obj2_end
            })
            frame_data["objects"].append({
                "name": "object1_minion_2",
                "visible": True,
                "transform": obj1_end
            })
        
        animation_data["frames"].append(frame_data)
    
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(animation_data, f, indent=2)
    
    print(f"\n✅ Animation JSON saved to: {output_json_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   File size: {os.path.getsize(output_json_path) / 1024:.2f} KB")
    
    return animation_data


def render_from_json(
    json_path,
    scene_path,
    object1_path,
    object2_path,
    output_path,
    iteration=-1
):
    """
    Render animation from JSON file (REPRODUCIBLE)
    """
    print("\n" + "=" * 80)
    print("RENDERING FROM JSON")
    print("=" * 80)
    
    # Load JSON
    print(f"Loading animation JSON from: {json_path}")
    with open(json_path, 'r') as f:
        animation_data = json.load(f)
    
    total_frames = animation_data["metadata"]["total_frames"]
    print(f"Total frames: {total_frames}")
    
    # Load scene model for cameras
    print(f"\nLoading scene from {scene_path}")
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    args = parser.parse_args([])
    
    cfg_path = os.path.join(scene_path, "cfg_args")
    with open(cfg_path, 'r') as f:
        cfg_string = f.read()
    cfg_args = eval(cfg_string, {"Namespace": Namespace})
    
    for key, value in vars(cfg_args).items():
        setattr(args, key, value)
    args.model_path = scene_path
    args.resolution = 2  # Downsample for faster rendering
    args.data_device = 'cpu'
    
    # Load scene (for cameras and background)
    print("Loading scene cameras...")
    gaussians_temp = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians_temp, load_iteration=iteration, shuffle=False)
    train_cams = scene.getTrainCameras()
    print(f"Loaded {len(train_cams)} cameras")
    
    # Load scene gaussians
    print("Loading scene Gaussians...")
    scene_gaussians = load_gaussian_model_simple(scene_path, iteration)
    
    # Load objects
    print(f"Loading object 1 (minion_2) from {object1_path}")
    object1_gaussians = load_gaussian_model_simple(object1_path, iteration)
    
    print(f"Loading object 2 (minion) from {object2_path}")
    object2_gaussians = load_gaussian_model_simple(object2_path, iteration)
    
    # Setup rendering
    pipeline_parser = ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)
    pipeline_args_namespace = pipeline_parser.parse_args([])
    pipeline_args = pipeline_params.extract(pipeline_args_namespace)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Create output directory
    makedirs(output_path, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("RENDERING FRAMES")
    print("=" * 80)
    
    # Render frames based on JSON
    with torch.no_grad():
        for frame_data in tqdm(animation_data["frames"], desc="Rendering frames"):
            frame_idx = frame_data["frame"]
            cam_idx = frame_data["camera_index"]
            
            # Get camera
            cam_idx = min(cam_idx, len(train_cams) - 1)
            view = train_cams[cam_idx]
            
            # Compose scene
            composed = ComposedGaussianModel()
            composed.add_model(scene_gaussians)
            
            # Add objects based on JSON
            for obj_data in frame_data["objects"]:
                if not obj_data["visible"]:
                    continue
                
                obj_name = obj_data["name"]
                transform = obj_data["transform"]
                
                # Select correct object
                if obj_name == "object1_minion_2":
                    obj_gaussians = object1_gaussians
                elif obj_name == "object2_minion":
                    obj_gaussians = object2_gaussians
                else:
                    continue
                
                # Add object with transform from JSON
                composed.add_model(
                    obj_gaussians,
                    translation=transform['translation'],
                    rotation=transform['rotation'],
                    scale=transform['scale'],
                    remove_pattern=True,
                    color_threshold=0.75
                )
            
            # Create temporary Gaussian model for rendering
            temp_gaussians = GaussianModel(args.sh_degree)
            # Convert numpy arrays to torch tensors and move to GPU
            temp_gaussians._xyz = torch.tensor(composed._xyz, dtype=torch.float32, device="cuda")
            temp_gaussians._features_dc = torch.tensor(composed._features_dc, dtype=torch.float32, device="cuda")
            temp_gaussians._features_rest = torch.tensor(composed._features_rest, dtype=torch.float32, device="cuda")
            temp_gaussians._scaling = torch.tensor(composed._scaling, dtype=torch.float32, device="cuda")
            temp_gaussians._rotation = torch.tensor(composed._rotation, dtype=torch.float32, device="cuda")
            temp_gaussians._opacity = torch.tensor(composed._opacity, dtype=torch.float32, device="cuda")
            
            # Render frame
            rendering = render(view, temp_gaussians, pipeline_args, background,
                             use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            
            # Save frame
            output_file = os.path.join(output_path, f'frame_{frame_idx:04d}.png')
            torchvision.utils.save_image(rendering, output_file)
    
    print(f"\n✅ Animation complete! {total_frames} frames saved to: {output_path}")
    print("\nTo create video, run:")
    print(f"ffmpeg -framerate 30 -i {output_path}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}/animation.mp4")


if __name__ == "__main__":
    parser = ArgumentParser(description="Final composition animation with JSON")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene model")
    parser.add_argument("--object1", type=str, required=True, help="Path to object1 (minion_2)")
    parser.add_argument("--object2", type=str, required=True, help="Path to object2 (minion)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--json", type=str, default="animation_config.json", help="Path to animation JSON file")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load")
    parser.add_argument("--json-only", action="store_true", help="Only generate JSON, don't render")
    
    args = parser.parse_args()
    
    safe_state(False)
    
    # Step 1: Generate JSON
    json_path = args.json
    animation_data = generate_animation_json(json_path)
    
    # Step 2: Render from JSON (unless --json-only)
    if not args.json_only:
        render_from_json(
            json_path,
            args.scene,
            args.object1,
            args.object2,
            args.output,
            args.iteration
        )
    else:
        print("\n✅ JSON-only mode: Skipping rendering")
        print(f"To render later, run without --json-only flag")
