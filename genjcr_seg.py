import torch, glob, os
import numpy as np
from src.helpers.data_process import read_pkl
from PIL import Image
from rgb_demo import main

def read_biarm_imgs(directory):
    """
    Load can_left and can_right images from a specified directory.

    Parameters:
        directory (str): The path to the directory containing the image files.

    Returns:
        tuple: Two lists containing numpy arrays of the can_left and can_right images, respectively.
    """
    # Build file patterns with the directory name
    left_pattern = os.path.join(directory, "can_left_*.png")
    right_pattern = os.path.join(directory, "can_right_*.png")

    # Retrieve and sort file paths
    left_paths = sorted(glob.glob(left_pattern))
    right_paths = sorted(glob.glob(right_pattern))

    # Open each image and convert it into a numpy array
    left_images = np.stack([np.array(Image.open(path)) for path in left_paths])
    right_images = np.stack([np.array(Image.open(path)) for path in right_paths])

    return left_images, right_images

def ensure_stacked(x):
    """
    If x is a list of tensors, stack them along a new 0-dim and return the tensor.
    Otherwise, return x unchanged.
    """
    if isinstance(x, list):
        # assumes each element of x is a torch.Tensor of the same shape
        return torch.stack(x, dim=0)
    return x

def read_biarm_data_rev(data_dir, n_imgs_per_side, factor=1000):
    left_eef_poses_file = f"{data_dir}/can_left_poses.pt"
    right_eef_poses_file = f"{data_dir}/can_right_poses.pt"
    
    left_eef_poses = ensure_stacked(torch.load(left_eef_poses_file, weights_only=False))
    right_eef_poses = ensure_stacked(torch.load(right_eef_poses_file, weights_only=False))
    l, r = read_biarm_imgs(data_dir)
    
    
    l = l[:len(left_eef_poses)]
    r = r[:len(right_eef_poses)]
    # print(l.shape, r.shape, left_eef_poses.shape, right_eef_poses.shape)
    right_eef_poses[:, 3:] /= factor
    left_eef_poses[:, 3:] /= factor
    return l[-n_imgs_per_side:], left_eef_poses[-n_imgs_per_side:], r[-n_imgs_per_side:], right_eef_poses[-n_imgs_per_side:]


if __name__ == "__main__":
    n_imgs_per_side = 8
    exp_name = 4
    data_dir = f"../genJCR_light/data/data_3/{exp_name}"
    genjcr_res_path = f"../genJCR_light/output/data_3/{exp_name}_{n_imgs_per_side}_gjcr_sgd.pkl"
    res_dict = read_pkl(genjcr_res_path)
    conf_mask = res_dict['conf_masks'].reshape(-1,)
    xyz = res_dict['xyzs'].numpy().reshape(-1, 3)[conf_mask]        # (n,3)
    print(len(xyz), len(res_dict['xyzs'].reshape(-1, 3)))
    rgb = res_dict['rgbs'].reshape(-1, 3)[conf_mask] # (n,3)  uint8 0–255
    left_imgs, left_eef_poses, right_imgs, right_eef_poses = read_biarm_data_rev(data_dir=data_dir, n_imgs_per_side=n_imgs_per_side)

    all_imgs = np.concatenate((left_imgs, right_imgs))
    all_poses = torch.concatenate((left_eef_poses, right_eef_poses))
    cam_poses = res_dict['cam_poses']    # (m,4,4) camera→world, already numpy
    eef_poses = res_dict['eef_poses'].numpy() # (m,4,4) eef->world
    br2bl = res_dict['br2bl'].numpy()

    main(rgbs_o=all_imgs, xyzs=res_dict['xyzs'], rgbs=res_dict['rgbs'], conf_3d_masks=res_dict['conf_masks'].cuda(), poses=cam_poses, save_name=exp_name, rerun_anyway=False)