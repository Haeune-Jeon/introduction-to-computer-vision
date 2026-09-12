"""
Render a composed Gaussian Splatting model with custom PLY
"""

import torch
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_composed(
    scene_model_path,
    composed_ply_path,
    output_path,
    iteration=-1,
    skip_train=False,
    skip_test=False,
    resolution=1,
    data_device='cpu'
):
    """
    Render a composed PLY using cameras from scene model
    
    Args:
        scene_model_path: path to scene model (for camera info)
        composed_ply_path: path to composed PLY file
        output_path: where to save rendered images
        iteration: which iteration cameras to use
        skip_train: skip training cameras
        skip_test: skip test cameras
        resolution: resolution scale
        data_device: 'cpu' or 'cuda'
    """
    print(f"Loading cameras from scene: {scene_model_path}")
    print(f"Loading Gaussians from: {composed_ply_path}")
    
    # Parse scene config
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    args = parser.parse_args([])
    
    # Load scene config
    cfg_path = os.path.join(scene_model_path, "cfg_args")
    with open(cfg_path, 'r') as f:
        cfg_string = f.read()
    cfg_args = eval(cfg_string, {"Namespace": Namespace})
    
    # Update args
    for key, value in vars(cfg_args).items():
        setattr(args, key, value)
    args.model_path = scene_model_path
    args.resolution = resolution
    args.data_device = data_device
    
    # Load scene (for cameras only)
    print("Loading cameras...")
    gaussians_temp = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians_temp, load_iteration=iteration, shuffle=False)
    
    # Load composed Gaussians from PLY
    print("Loading composed Gaussians...")
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(composed_ply_path)
    
    # Setup rendering
    pipeline_parser = ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)
    pipeline_args_namespace = pipeline_parser.parse_args([])
    pipeline_args = pipeline_params.extract(pipeline_args_namespace)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Render
    with torch.no_grad():
        if not skip_train:
            # Original code (all cameras)
            # print("Rendering training views...")
            # render_path = os.path.join(output_path, "train", "renders")
            # makedirs(render_path, exist_ok=True)
            # 
            # train_cams = scene.getTrainCameras()
            # for idx, view in enumerate(tqdm(train_cams, desc="Training views")):
            #     rendering = render(view, gaussians, pipeline_args, background,
            #                      use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            #     torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
            
            # Modified: Render from camera 300 to end
            print("Rendering training views (from camera 300 to end)...")
            render_path = os.path.join(output_path, "train", "renders")
            makedirs(render_path, exist_ok=True)
            
            train_cams = scene.getTrainCameras()
            start_idx = 230
            print(f"Total cameras: {len(train_cams)}, rendering from {start_idx} to {len(train_cams)-1}")
            
            for idx in tqdm(range(start_idx, len(train_cams)), desc="Training views (300+)"):
                view = train_cams[idx]
                rendering = render(view, gaussians, pipeline_args, background,
                                 use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        
        if not skip_test:
            print("Rendering test views...")
            render_path = os.path.join(output_path, "test", "renders")
            makedirs(render_path, exist_ok=True)
            
            test_cams = scene.getTestCameras()
            if len(test_cams) > 0:
                for idx, view in enumerate(tqdm(test_cams, desc="Test views")):
                    rendering = render(view, gaussians, pipeline_args, background,
                                     use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
            else:
                print("No test cameras found.")
    
    print(f"\nRendering complete! Images saved to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render composed Gaussian Splatting model")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene model (for cameras)")
    parser.add_argument("--composed_ply", type=str, required=True, help="Path to composed PLY file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Scene iteration for cameras")
    parser.add_argument("--skip_train", action="store_true", help="Skip training views")
    parser.add_argument("--skip_test", action="store_true", help="Skip test views")
    parser.add_argument("-r", "--resolution", type=int, default=1, help="Resolution downscale factor")
    parser.add_argument("--data_device", type=str, default='cpu', help="Device for images")
    
    args = parser.parse_args()
    
    safe_state(False)
    
    render_composed(
        args.scene,
        args.composed_ply,
        args.output,
        args.iteration,
        args.skip_train,
        args.skip_test,
        args.resolution,
        args.data_device
    )

