"""
3D Gaussian Splatting Composition Script
Combines scene and object models with transformations
"""

import torch
import numpy as np
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.system_utils import searchForMaxIteration
import os
from scipy.spatial.transform import Rotation as R

class ComposedGaussianModel:
    """Composed model from multiple Gaussian models"""
    def __init__(self):
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        
    def add_model(self, gaussian_model, translation=[0, 0, 0], rotation=[0, 0, 0], scale=1.0, 
                  remove_pattern=False, color_threshold=0.75):
        """
        Add a Gaussian model with transformation
        
        Args:
            gaussian_model: GaussianModel to add
            translation: [x, y, z] translation
            rotation: [rx, ry, rz] rotation in degrees
            scale: uniform scale factor
            remove_pattern: If True, remove white/light beige checkered pattern
            color_threshold: RGB brightness threshold (0-1) for pattern removal
        """
        # Get Gaussian parameters
        xyz = gaussian_model.get_xyz.detach().cpu().numpy()
        features_dc = gaussian_model._features_dc.detach().cpu().numpy()
        features_rest = gaussian_model._features_rest.detach().cpu().numpy()
        scaling = gaussian_model._scaling.detach().cpu().numpy()
        quaternions = gaussian_model._rotation.detach().cpu().numpy()
        opacity = gaussian_model._opacity.detach().cpu().numpy()
        
        # COLOR-BASED FILTER: Remove white/light beige checkered pattern
        if remove_pattern and self._xyz is not None:  # Only for objects, not scene
            original_count = len(xyz)
            
            # Convert SH DC features to RGB
            # features_dc shape: [N, 1, 3] -> [N, 3]
            rgb = features_dc[:, 0, :]  # Extract RGB from DC component
            # Apply sigmoid to get [0, 1] range
            rgb = 1 / (1 + np.exp(-rgb))
            
            # Calculate brightness (average of R, G, B)
            brightness = rgb.mean(axis=1)
            
            # Remove bright white/beige colors (high brightness)
            # Also check if all RGB channels are high (white/light beige)
            is_bright = brightness > color_threshold
            is_light_color = (rgb.min(axis=1) > 0.6) & (rgb.max(axis=1) > 0.7)  # All channels bright
            
            # Combine: remove if bright AND light-colored (white/beige checkered pattern)
            remove_mask = is_bright & is_light_color
            
            keep_mask = ~remove_mask  # Keep everything else
            
            xyz = xyz[keep_mask]
            features_dc = features_dc[keep_mask]
            features_rest = features_rest[keep_mask]
            scaling = scaling[keep_mask]
            quaternions = quaternions[keep_mask]
            opacity = opacity[keep_mask]
            
            removed = original_count - len(xyz)
            print(f"  🔹 Removed white/beige pattern: {original_count} → {len(xyz)} ({removed} Gaussians)")
        # Objects already have their natural positions from scanning
        
        # Apply transformations
        # 1. Scale
        xyz = xyz * scale
        scaling = scaling + np.log(scale)  # scaling is in log space
        
        # 2. Rotate
        rotation_angles = np.array(rotation)
        if np.any(rotation_angles != 0):
            rot_matrix = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
            xyz = xyz @ rot_matrix.T
            
            # Rotate quaternions
            rot_quat = R.from_euler('xyz', rotation_angles, degrees=True).as_quat()
            original_rot = R.from_quat(quaternions)
            new_rot = R.from_quat(rot_quat) * original_rot
            quaternions = new_rot.as_quat()
        
        # 3. Translate
        xyz = xyz + np.array(translation)
        
        # Concatenate to existing data
        if self._xyz is None:
            self._xyz = xyz
            self._features_dc = features_dc
            self._features_rest = features_rest
            self._scaling = scaling
            self._rotation = quaternions
            self._opacity = opacity
        else:
            self._xyz = np.concatenate([self._xyz, xyz], axis=0)
            self._features_dc = np.concatenate([self._features_dc, features_dc], axis=0)
            self._features_rest = np.concatenate([self._features_rest, features_rest], axis=0)
            self._scaling = np.concatenate([self._scaling, scaling], axis=0)
            self._rotation = np.concatenate([self._rotation, quaternions], axis=0)
            self._opacity = np.concatenate([self._opacity, opacity], axis=0)
    
    def to_gaussian_model(self, sh_degree=3):
        """Convert to GaussianModel for rendering"""
        model = GaussianModel(sh_degree)
        
        # Initialize with dummy scene
        model._xyz = torch.tensor(self._xyz, dtype=torch.float32).cuda()
        model._features_dc = torch.tensor(self._features_dc, dtype=torch.float32).cuda()
        model._features_rest = torch.tensor(self._features_rest, dtype=torch.float32).cuda()
        model._scaling = torch.tensor(self._scaling, dtype=torch.float32).cuda()
        model._rotation = torch.tensor(self._rotation, dtype=torch.float32).cuda()
        model._opacity = torch.tensor(self._opacity, dtype=torch.float32).cuda()
        
        return model


def load_gaussian_model_simple(model_path, iteration=-1):
    """Load only Gaussian parameters from PLY file (no images, cameras)"""
    # Find PLY file
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    print(f"  Loading from: {ply_path}")
    
    # Load config to get sh_degree
    cfg_path = os.path.join(model_path, "cfg_args")
    with open(cfg_path, 'r') as f:
        cfg_string = f.read()
    cfg_args = eval(cfg_string, {"Namespace": Namespace})
    sh_degree = cfg_args.sh_degree if hasattr(cfg_args, 'sh_degree') else 3
    
    # Create Gaussian model and load PLY
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    
    return gaussians

# function with bg filtering
# def compose_scene(scene_path, object_paths, object_transforms, output_path, iteration=-1, data_device='cpu',
#                 remove_pattern=False, color_threshold=0.75):

def compose_scene(scene_path, object_paths, object_transforms, output_path, iteration=-1, data_device='cpu', remove_pattern=False, color_threshold=0.75):
    """
    Compose scene with multiple objects
    
    Args:
        scene_path: path to scene model
        object_paths: list of paths to object models
        object_transforms: list of dicts with 'translation', 'rotation', 'scale'
        output_path: where to save composed model
        iteration: which iteration to load (-1 for latest)
        data_device: 'cpu' or 'cuda' (not used in simple version)
        remove_pattern: If True, remove white/light beige checkered pattern
        color_threshold: RGB brightness threshold (0-1) for pattern removal
    """
    print(f"Loading scene from {scene_path}")
    scene_gaussians = load_gaussian_model_simple(scene_path, iteration)
    
    # Create composed model
    composed = ComposedGaussianModel()
    
    # Add scene
    print("Adding scene...")
    composed.add_model(scene_gaussians)
    
    # Add objects with transformations
    for obj_path, transform in zip(object_paths, object_transforms):
        print(f"Loading object from {obj_path}")
        obj_gaussians = load_gaussian_model_simple(obj_path, iteration)
        
        translation = transform.get('translation', [0, 0, 0])
        rotation = transform.get('rotation', [0, 0, 0])
        scale = transform.get('scale', 1.0)
        
        print(f"  Translation: {translation}")
        print(f"  Rotation: {rotation}")
        print(f"  Scale: {scale}")
        
        composed.add_model(obj_gaussians, translation, rotation, scale,
                          remove_pattern=remove_pattern, color_threshold=color_threshold)
    
    # Save composed model
    print(f"Saving composed model to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    composed_model = composed.to_gaussian_model()
    composed_model.save_ply(os.path.join(output_path, "composed.ply"))
    
    print("Composition complete!")
    return composed_model


if __name__ == "__main__":
    # Example usage
    parser = ArgumentParser(description="Compose scene with objects")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene model")
    parser.add_argument("--objects", type=str, nargs="+", required=True, help="Paths to object models")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load")
    parser.add_argument("--data_device", type=str, default='cpu', help="Device for image storage (cpu or cuda)")
    parser.add_argument("--test_camera", type=int, default=228, help="Test camera to render (default: 228)")
    parser.add_argument("--skip_test_render", action='store_true', help="Skip test camera rendering")
    parser.add_argument("--remove_pattern", action='store_true', help="Remove white/light beige checkered pattern from objects")
    parser.add_argument("--color_threshold", type=float, default=0.75, help="RGB brightness threshold (0-1) for pattern removal")
    
    args = parser.parse_args()
    
    # Define transformations for each object
    # MOVE BOTH MUCH MORE TO THE RIGHT
    transforms = [
        # Object 1 (object_3) - minion2
        {
            # start point
            # 'translation': [1.2, -1.5, -1.5], 
            # 'rotation': [-25, 50, -10],   
            # 'scale': 1.4
            # end point
            'translation': [1.2, -1.5, 0.55],  
            'rotation': [-25, 50, -10],     # [0, 90, 0]    
            'scale': 1.4
        },
        # Object 2 (object_4) - minion1
        {
            # start point
            # 'translation': [1.9, 0.0, 2.8],  # [1.0, 0.0, 0.0]
            # 'rotation': [0, 30, 15],  # [0, 15, 15]
            # 'scale': 0.9
            # end point
            'translation': [1.9, 0.0, 1.0],  # [1.0, 0.0, 0.0]
            'rotation': [0, 30, 15],  # [0, 15, 15]
            'scale': 0.9
        },
        # Add more objects here if needed...
    ]
    
    # Ensure we have enough transforms defined
    if len(transforms) < len(args.objects):
        print(f"Warning: Not enough transforms defined. Using default for remaining objects.")
        for i in range(len(transforms), len(args.objects)):
            transforms.append({
                'translation': [0, 0, 0],
                'rotation': [0, 0, 0],
                'scale': 1.0
            })
    
    composed_model = compose_scene(args.scene, args.objects, transforms[:len(args.objects)], args.output, 
                                    args.iteration, args.data_device, args.remove_pattern, args.color_threshold)
    
    print("\n✅ Composition complete! PLY file saved.")
    
    # Automatically render test camera after composing
    if False:  # Disabled by default - use render_composed.py instead
    # if not args.skip_test_render:
        print(f"\n📷 Rendering test camera {args.test_camera} to verify composition...")
        
        # Quick inline rendering
        from scene import Scene
        from gaussian_renderer import render as gaussian_render
        from arguments import PipelineParams
        from utils.general_utils import safe_state
        import torchvision
        
        try:
            from diff_gaussian_rasterization import SparseGaussianAdam
            SPARSE_ADAM_AVAILABLE = True
        except:
            SPARSE_ADAM_AVAILABLE = False
        
        safe_state(False)
        
        # Load scene for camera
        cfg_path = os.path.join(args.scene, "cfg_args")
        with open(cfg_path, 'r') as f:
            cfg_string = f.read()
        cfg_args = eval(cfg_string, {"Namespace": Namespace})
        
        model_parser = ArgumentParser()
        model_params = ModelParams(model_parser, sentinel=True)
        model_args = model_parser.parse_args([])
        for key, value in vars(cfg_args).items():
            setattr(model_args, key, value)
        model_args.model_path = args.scene
        model_args.resolution = 2
        model_args.data_device = args.data_device
        
        gaussians_temp = GaussianModel(model_args.sh_degree)
        scene = Scene(model_args, gaussians_temp, load_iteration=args.iteration, shuffle=False)
        
        # Get test camera
        train_cams = scene.getTrainCameras()
        if args.test_camera < len(train_cams):
            view = train_cams[args.test_camera]
            
            # Setup pipeline
            pipeline_parser = ArgumentParser()
            pipeline_params = PipelineParams(pipeline_parser)
            pipeline_args_namespace = pipeline_parser.parse_args([])
            pipeline_args = pipeline_params.extract(pipeline_args_namespace)
            
            bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            # Render
            rendering = gaussian_render(view, composed_model, pipeline_args, background,
                                       use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            
            test_output = os.path.join(args.output, f"test_cam_{args.test_camera}.png")
            torchvision.utils.save_image(rendering, test_output)
            print(f"✅ Test render saved: {test_output}")
            print(f"👀 Check this image to verify your composition!")
        else:
            print(f"⚠️  Camera {args.test_camera} not found (max: {len(train_cams)-1})")

