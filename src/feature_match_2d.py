import torch, pickle, time, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.visualize.viz_masks import visualize_images


def rotation_angle_from_poses(T1, T2, to_cpu=True):
    """
    Given two 4x4 transformation matrices T1 and T2, returns the absolute
    rotation angle (in radians) needed to go from T1's orientation to T2's orientation.
    """
    # Extract 3x3 rotation components
    if to_cpu:
        R1 = T1[:3, :3].cpu()
        R2 = T2[:3, :3].cpu()
    else:
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]

    # Compute relative rotation
    R_diff = R1.T @ R2  # or R2 @ R1.T, depending on convention

    # To handle floating-point inaccuracy, clip trace to valid domain of arccos ([-1, 1])
    trace_val = (np.trace(R_diff) - 1) / 2.0
    trace_val = np.clip(trace_val, -1.0, 1.0)

    # Angle in radians
    angle = np.arccos(trace_val)
    return angle

def compute_poses_egde_cost(T1, T2, w_t=10., w_r=1.):
    t_diff = torch.norm(T2[:3, 3] - T2[:3, 3])
    r_diff = rotation_angle_from_poses(T1, T2, to_cpu=T1.is_cuda)
    return w_t*t_diff + w_r*r_diff, t_diff, r_diff



def convert_to_uint8(img):
    # For float arrays, clip values to [0, 1] and scale to [0, 255].
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0, 1)
        img = (img * 255).round()
    # For integer arrays, if values are binary, scale them.
    elif np.issubdtype(img.dtype, np.integer) and img.max() <= 1:
        img = img * 255
    
    return img.astype(np.uint8)

def generate_color_palette(n=100):
    """Generates a list of `n` distinct colors using a colormap."""
    return [plt.get_cmap('tab20')(i % 20) for i in range(n)]  # Cycling through tab20

def visualize_keypoint_matches(image1, image2, keypoints1, keypoints2, interval=500, c_interval=10):
    """
    Visualizes keypoint matches between two images with color cycling every 'interval' lines.
    For each match, a connecting line is drawn and a dot is placed at each keypoint.

    Parameters:
    - image1: First image (NumPy array of shape (480, 640, 3)).
    - image2: Second image (NumPy array of shape (480, 640, 3)).
    - keypoints1: NumPy array of shape (N, 2), containing (x, y) locations in image1.
    - keypoints2: NumPy array of shape (N, 2), containing (x, y) locations in image2.
    - interval: Number of matches per color cycle (default: 500).
    """
    assert image1.shape == image2.shape, "Both images must have the same dimensions"
    assert keypoints1.shape == keypoints2.shape, "Keypoint arrays must have the same shape"

    extra_shift = 50
    white_val = 255 if image1.dtype == np.uint8 else 1.0
    gap = np.full((image1.shape[0], extra_shift, image1.shape[2]), white_val, dtype=image1.dtype)
    combined_image = np.hstack((image1, gap, image2))
    # Create a combined image by stacking image2 to the right of image1
    # combined_image = np.hstack((image1, image2))

    # Sample keypoints with the given interval
    keypoints1, keypoints2 = keypoints1[::interval], keypoints2[::interval]

    # Shift keypoints in the second image to match their position in the combined image
    keypoints2_shifted = keypoints2.copy()
    keypoints2_shifted[:, 0] += image1.shape[1]+extra_shift  # Shift x-coordinates by image width

    # Generate at least 100 distinct colors
    color_palette = generate_color_palette(100)

    # Plot the images and keypoint matches
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_image)
    ax.axis("off")  # Remove axis
    plt.tight_layout()  # Tight layout

    # Draw keypoint matches with changing colors and plot dots at the keypoints
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(keypoints1, keypoints2_shifted)):
        color = color_palette[(i // c_interval) % len(color_palette)]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1., alpha=1.)
        ax.scatter(x1, y1, color=color, s=15)  # Dot on keypoint in image1
        ax.scatter(x2, y2, color=color, s=15)  # Dot on keypoint in image2

    plt.savefig("fm_cor.png", dpi=500)
    
    plt.show()


def roma_pair_match(roma_model, im_rgb1, im_rgb2, num_points=2000, device='cuda', viz_match=False):
    im_rgb1 = convert_to_uint8(im_rgb1)
    im_rgb2 = convert_to_uint8(im_rgb2)
    H, W = roma_model.get_output_resolution()
    im1 = Image.fromarray(im_rgb1).resize((W, H))
    im2 = Image.fromarray(im_rgb2).resize((W, H))
    warp, certainty = roma_model.match(im1, im2, device=device)
    H_A, W_A = im_rgb1.shape[:2]
    H_B, W_B = im_rgb2.shape[:2]
    matches, certainty = roma_model.sample(warp, certainty, num=num_points)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B) 
    if viz_match:
        visualize_keypoint_matches(im_rgb1, im_rgb2, kpts1.cpu().numpy(), kpts2.cpu().numpy(), interval=200, c_interval=5)
    return kpts1, kpts2, certainty

"""
mode:
path -> (n-1) pairs TSP path
thresh -> all pairs under specific thresh
bithresh -> <90 degree difference and < pose_edge_cost_thresh translational difference
complete -> all pairs
"""
def fm_images(roma_model, poses, rgbs, device, save_path="", num_points=2000, viz_rgb=False, viz_feature_match=True, mode='thresh', 
              pose_edge_cost_thresh=1.0, bi_thresh_rad_thresh=1.0,
              time_single_roma=False):
    rgb_images = rgbs
    if viz_rgb:
        visualize_images(rgb_images)
        
    poses_egde_costs = dict()
    for i in range(len(poses)):
        for j in range(len(poses)):
            if (i != j):
                poses_egde_costs[(i, j)] = compute_poses_egde_cost(poses[i], poses[j])
                # print(f"edge cost: {i, j} : {poses_egde_costs[(i, j)]}")
                # assert poses_egde_costs[(i, j)]>=0, "egde cost lower than 0, error!"
    fm_res = []
    
    if mode == 'bithresh':
        print(f"Bi-Threshold bottlneck cost: {math.degrees(bi_thresh_rad_thresh)} degree and {pose_edge_cost_thresh}")
    elif mode == 'thresh':
        print(f"Threshold bottlneck cost: {pose_edge_cost_thresh}")
    else:
        print(f"Complete mode, watch for runtime!")
    for i in range(len(poses)):
        for j in range(i+1, len(poses)): 
            find_match = False
            if mode == 'bithresh':
                _, t_diff, r_diff = poses_egde_costs[(i, j)]
                find_match = (t_diff <= pose_edge_cost_thresh) and (r_diff <= bi_thresh_rad_thresh)
            elif mode == 'thresh':
                cost, _, _ = poses_egde_costs[(i, j)]
                find_match = cost <= pose_edge_cost_thresh
            else: 
                find_match = True
            
            if find_match:
                if time_single_roma:
                    start_time = time.perf_counter()
                kpt1, kpt2, certainty = roma_pair_match(roma_model=roma_model, im_rgb1=rgb_images[i], im_rgb2=rgb_images[j], 
                                                        device=device, viz_match=viz_feature_match, num_points=num_points)
                if time_single_roma:
                    end_time = time.perf_counter()
                    runtime_model = end_time - start_time
                    print(f"SINGLE ROMA model runtime: {runtime_model:.4f} seconds")
                
                fm_res.append({
                    'img_idx_pair': (i, j), 
                    'certainty': certainty, 
                    'kpts_pair': (kpt1, kpt2)
                })
    
    if save_path != "":
        with open(save_path, "wb") as f:
            pickle.dump(fm_res, f)
    return fm_res