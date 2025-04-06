import sys, os
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dust3r_local_dir = os.path.join(project_folder, "third_party/dust3r")
print(dust3r_local_dir)
sys.path.append(dust3r_local_dir)
sys.path.append(project_folder)
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from dust3r.model import AsymmetricCroCo3DStereo

import torch
import numpy as np

std_model_pth = f"checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
"""
Several utils for running dust3r and extracting camera poses and points cloud.
Their functionality is literally explained by their name.
"""

def running_dust3r(out_file, file_paths,
                   batch_size=8, img_size=512,
                   schedule="cosine", lr=0.01, niter=500, 
                   device="cuda", model_path=std_model_pth):
    model = AsymmetricCroCo3DStereo.from_pretrained(std_model_pth).to(device)
    images = load_images(file_paths, size=img_size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    _ = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    imgs = scene.imgs # array 
    focals = scene.get_focals() #
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    
    tensors_to_save = {
        'poses': poses.detach(), # tensor
        'pts': [xyzs.detach() for xyzs in pts3d], # list of tensor
        'images': imgs, # array 
        'masks': [mask.detach() for mask in confidence_masks],
        'est_focals': focals.detach(), # tensor
    }
    torch.save(tensors_to_save, out_file)
    print("-"*10)
    print(f"dust3r initial RAW output saved at {out_file}")
    print("-"*10)
    return out_file
    

def load_pose_ptc(dict_loc):
    d = torch.load(dict_loc)
    poses = d['poses']
    pts_tor = d['pts']
    rgb_colors = d['rgb_colors']
    loc_info = d['loc_info']
    # may not needed
    compressed_imgs = d['images']
    masks = d['masks']
    return poses, pts_tor, rgb_colors, loc_info

def extract_pos_ori(transform_matrices):
    positions = transform_matrices[:, :3, 3]
    forward_directions = transform_matrices[:, :3, 2] # Negate to if cam face -Z direction
    return positions, forward_directions
