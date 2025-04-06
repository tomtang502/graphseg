from src.scale_calib_helper import *


def compute_arm(eef_poses_selected, w2c_poses_selected, arm1=False):
    
    w2c_poses_selected_scale_cpy = w2c_poses_selected.clone()
    if arm1:
        vals_sc=np.linspace(0.01, 10, 200000).reshape(-1)
    else:
        vals_sc=np.linspace(0.01, 10, 200000).reshape(-1)
    A, B, Rx, N=compute_A_B_Rx(eef_poses_selected,w2c_poses_selected_scale_cpy)


    #we can just brute force
    costs_list=[]
    tx_list=[]
    for i in range(len(vals_sc)):
        J_res, tx_res=compute_cost(vals_sc[i],A, B, Rx, N)
        costs_list.append(J_res)
        tx_list.append(tx_res)
    costs_tor=torch.tensor(costs_list)

    min_inds=torch.argmin(costs_tor)
    scale=vals_sc[min_inds]
    #scale=0.025
    print("scale found: {}".format(scale))

    w2c_poses_selected[:, :3,3]=w2c_poses_selected[:, :3, 3] * scale
    A, B, Rx, N=compute_A_B_Rx(eef_poses_selected, w2c_poses_selected)
    J,tx=compute_cost(1.,A, B, Rx, N) # the scale factor is just 1 as we have already rescaled

    X = eye(4)
    X[0:3, 0:3] = Rx
    X[0:3, -1] = tx.reshape(-1)
    
    d = A@X - X@B
    t_diff = d[:, :3, -1]#.mean(axis=0)
    R_diff = d[:, :3, :3]#.mean(axis=0)
    #print(t_diff)
    R_L = []
    for r in R_diff:
        R_L.append(np.linalg.norm(r))
    R_L = np.array(R_L)
    t_L = np.linalg.norm(t_diff, axis=1)
    tmp_list=[]
    for i in range(len(w2c_poses_selected)):
        rob = eef_poses_selected[i]
        obj = w2c_poses_selected[i]
        tmp = dot(rob, dot(X, inv(obj)))
        tmp_list.append(tmp)
    tmp_tor=torch.tensor(np.array(tmp_list))
    world_pose=tmp_tor.mean(dim=0)

    return world_pose, scale, J, R_L, t_L

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
    