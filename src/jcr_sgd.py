import torch
import numpy as np
from src.helpers.scale_calib_helper import compute_cost_tr, compute_A_B_Rx_tr
from src.helpers.batched_geometry import compute_cost_tr_multi_scale, rotation_angle_diff_batched
from src.helpers.geometric import average_rotations

def transform_points(points, matrix):
    
    # Check if the inputs are torch tensors
    if not isinstance(points, torch.Tensor) or not isinstance(matrix, torch.Tensor):
        raise ValueError("Both points and matrix must be torch.Tensor objects.")
    
    # Check the shape of the points and the matrix
    if points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Invalid shape for points or matrix. Points should be (n, 3) and matrix should be (4, 4).")
    
    # Add an extra dimension of ones to the points tensor to make it compatible with the transformation matrix
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_homogeneous = torch.cat([points, ones], dim=1)  # Convert points to homogeneous coordinates
    
    # Apply the transformation matrix to the points
    transformed_points_homogeneous = torch.mm(points_homogeneous, matrix.t())  # Multiply by the transpose of the matrix
    
    # Convert back from homogeneous coordinates by dropping the last dimension
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points

def transpose_poses_ptc(poses, ptc, trans_mat):
    poses_trans = trans_mat.float()@poses
    ptc_trans = transform_points(ptc, trans_mat.float())
    return poses_trans, ptc_trans

def rpyxyz_to_T(pose):
    roll, pitch, yaw, x, y, z = pose
    # Rotation about the X-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Rotation about the Y-axis (pitch)
    R_y = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation about the Z-axis (yaw)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: apply roll, then pitch, then yaw
    R = R_z @ R_y @ R_x  # matrix multiplication
    
    # Construct the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T
    

def axxb_loss(A: torch.Tensor, X: torch.Tensor, B: torch.Tensor):
    lside, rside = A@X, X@B
    R_lside, R_rside = lside[:, :3, :3], rside[:, :3, :3]
    t_lside, t_rside = lside[:, :3, -1], rside[:, :3, -1]
    R_diff = rotation_angle_diff_batched(R_lside, R_rside)
    t_diff = torch.linalg.norm(t_lside - t_rside, dim=-1)
    
    # print(R_diff.mean())
    # print(t_diff.mean())
    return R_diff.mean(), t_diff.mean()
    
def pm_solver_with_scale(eef_poses, c2w_poses, scale_min=0.01, scale_max=10., num_samples=int(2e5), device='cuda'):
    eef_poses = eef_poses.clone().double()
    c2w_poses = c2w_poses.clone().double()
    vals_sc=torch.linspace(scale_min, scale_max, num_samples, dtype=torch.double).reshape(-1)
    
    A, B, Rx = compute_A_B_Rx_tr(eef_poses, c2w_poses)
    A, B, Rx = A.double(), B.double(), Rx.double()
    
    Js, _ = compute_cost_tr_multi_scale(scales=vals_sc, A=A, B=B, Rx=Rx, device=device)
    return Js, vals_sc

def pm_solver(eef_poses, c2w_poses, scale):
    eef_poses = eef_poses.clone().double()
    c2w_poses = c2w_poses.clone().double()
    c2w_poses[:, :3, 3] = c2w_poses[:, :3, 3]*scale
    A, B, Rx = compute_A_B_Rx_tr(eef_poses, c2w_poses)
    A, B, Rx = A.double(), B.double(), Rx.double()
    _, tx=compute_cost_tr(1., A, B, Rx) # the scale factor is just 1 as we have already rescaled
    X = torch.eye(4, dtype=torch.double)
    X[0:3, 0:3] = Rx
    X[0:3, -1] = tx.reshape(-1)
    return A, B, X

def retrieve_w2b_mat(eef_poses, c2w_poses, X):
    w2b_list = []
    for i in range(len(c2w_poses)):
        e_i, c_i = eef_poses[i], c2w_poses[i]
        w2b = e_i@X@torch.linalg.inv(c_i)
        w2b_list.append(w2b)
    
    w2b_T = torch.stack(w2b_list)
    R_avg = average_rotations(R=w2b_T[:, :3, :3].clone())
    t_avg = w2b_T[:, :3, -1].clone().mean(dim=0)
    w2b_mat = torch.eye(4, dtype=w2b_T.dtype, device=w2b_T.device)
    w2b_mat[:3, :3] = R_avg
    w2b_mat[:3, -1] = t_avg
    return w2b_mat

def compute_single_arms(eef_poses_all, c2w_poses_all, device='cuda'):
    
    eef_poses_all = eef_poses_all.double()
    c2w_poses_all = c2w_poses_all.double()

    #we can just brute force
    Js, val_sc = pm_solver_with_scale(eef_poses_all, c2w_poses_all, device=device)
    min_inds = torch.argmin(Js)
    scale = val_sc[min_inds].to('cpu').item()
    print("scale found: {}".format(scale))
    
    A, B, X = pm_solver(eef_poses_all, c2w_poses_all, scale=scale)
    
    R_diff, t_diff = axxb_loss(A=A, X=X, B=B)
    
    c2w_poses_all[:, :3, -1] = c2w_poses_all[:, :3, -1]*scale
    w2b_T = retrieve_w2b_mat(eef_poses=eef_poses_all, c2w_poses=c2w_poses_all, X=X)
    
    return w2b_T, X, scale, Js, R_diff, t_diff