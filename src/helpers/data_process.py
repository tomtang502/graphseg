import os, pickle
import numpy as np
from PIL import Image


def read_pkl(file_path):
    """
    Reads a pickle file and returns its content.

    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        Object: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_grnet_data(root_dir, scene_name, step_size, s=1000):
    out_scene_dir = os.path.join(root_dir, scene_name)
    pose_dir = os.path.join(out_scene_dir, "pose")
    depth_dir = os.path.join(out_scene_dir, "depth")
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])[::step_size]
    depths = [Image.open(os.path.join(depth_dir, filepath)).convert("L") for filepath in depth_files]
    # grnet internal processing
    depths = np.stack(depths)
    fault_mask = depths < 200
    depths[fault_mask] = 0
    depths[(np.abs(depths) < 10)] = 0
    depths = depths/s
    
    ptc_file_path = os.path.join(out_scene_dir, f"{scene_name}_{step_size}.pkl")
    with open(ptc_file_path, 'rb') as f:
        data = pickle.load(f)
    rgbs = data['rgbs']
    xyzs = data['xyzs']
    labels = data['labels']
    
    txt_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    matrices = []
    for filename in txt_files:
        filepath = os.path.join(pose_dir, filename)
        try:
            matrix = np.loadtxt(filepath)
            if matrix.shape != (4, 4):
                print(f"Warning: {filename} does not contain a 4x4 matrix.")
            matrices.append(matrix)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    poses = np.stack(matrices, axis=0)
    return rgbs, xyzs, poses[::step_size], labels, depths