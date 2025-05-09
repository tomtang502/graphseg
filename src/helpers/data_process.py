import os, pickle, torch
import numpy as np
from PIL import Image
from typing import Any, Optional, Union

def ensure_tensor(
    x: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Ensure `x` is a torch.Tensor.    
    - If `x` is already a Tensor, returns it (cast if dtype/device given).  
    - If `x` is a numpy.ndarray, converts via from_numpy.  
    - If `x` is a list/tuple of Tensors or of numpy.ndarrays, stacks them.  
    - Else, wraps `x` via torch.tensor (e.g. scalar or list of scalars).
    
    Args:
        x: input to convert.
        dtype: optional target dtype.
        device: optional device (e.g. 'cuda:0' or torch.device).
    """
    # 1) Already a tensor?
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (dtype or device) else x

    # 2) A single NumPy array?
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(device=device, dtype=dtype) if (dtype or device) else t

    # 3) A sequence?
    if isinstance(x, (list, tuple)):
        # 3a) list of Tensors → stack
        if all(isinstance(elem, torch.Tensor) for elem in x):
            stacked = torch.stack(x)
            return stacked.to(device=device, dtype=dtype) if (dtype or device) else stacked

        # 3b) list of NumPy arrays → convert & stack
        if all(isinstance(elem, np.ndarray) for elem in x):
            tensor_list = [torch.from_numpy(elem) for elem in x]
            stacked = torch.stack(tensor_list)
            return stacked.to(device=device, dtype=dtype) if (dtype or device) else stacked

        # 3c) mixed or list of scalars → fallback to torch.tensor
        t = torch.tensor(x, dtype=dtype)
        return t.to(device=device) if device else t

    # 4) Anything else (scalar, etc.) → torch.tensor
    t = torch.tensor(x, dtype=dtype)
    return t.to(device=device) if device else t

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