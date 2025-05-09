import os, torch
import shutil
import tempfile
import numpy as np
from PIL import Image
from src.dust3r_api import running_dust3r
from src.helpers.data_process import read_pkl
from src.visualize.viz_ptc import plot_eef_and_camera_poses
from src.eth_jcr import compute_arm, transpose_poses_ptc, rpyxyz_to_T

def save_images_to_temp(tensor):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created at: {temp_dir}")
    
    # Save each image in the tensor as a PNG file
    for idx, img_array in enumerate(tensor):
        # Ensure image data is in uint8 format
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        img_path = os.path.join(temp_dir, f"image_{chr(ord('a')+idx)}.png")
        img.save(img_path)
    return temp_dir

def get_sorted_png_files(directory):
    # List all files ending with .png (case-insensitive) and sort them
    png_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.png')
    ]
    return sorted(png_files)

def reconstruct_3d(exp_name, rgbs, eef_poses=None, out_dir="output/reconstruct_3d_out", 
                   ckpt_path="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                   visualize=True, visualize_stride=50, caliberate=False, rerun_anyway=False):
    """reconstruct 3d from rgb images

    Args:
        exp_name (str): a nick name for experiment
        rgbs (array or tensor): (n_images x H x W x 3) RGB images
        out_dir (str): output directory for 3d reconstruction
    
    Return:
        xyzs, rgbs (reshaped), cam_poses, eef_poses
    """
    out_file = f"{out_dir}/{exp_name}.pth"
    caliberated_out_file = f"{out_dir}/{exp_name}_caliberated.pth"
    if rerun_anyway or (not os.path.exists(out_file)):
        temp_dir = save_images_to_temp(rgbs)

        out_file = running_dust3r(out_file=out_file, file_paths=get_sorted_png_files(temp_dir), model_path=ckpt_path)
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} deleted.")
        print(f"3D reconstruction from RGB images done, raw output saved at {out_file}")
    else:
        print(f"3D reconstruction from RGB images already processed, raw output saved at {out_file}")
    
    dust3r_raw = torch.load(out_file, weights_only=False)
    masks = torch.stack(dust3r_raw['masks'])
    N, H, W, _ = np.stack(dust3r_raw['images']).shape
    rgbs, xyzs, w2c_poses_unaligned = np.stack(dust3r_raw['images']).reshape(-1, 3), torch.stack(dust3r_raw['pts']).reshape(-1, 3).to('cpu'), dust3r_raw['poses'].to('cpu')
    if caliberate:
        assert eef_poses is not None, "eef_poses need to be provided for eye to hand caliberation"
        eef_poses_tor =  torch.tensor(np.stack([rpyxyz_to_T(pose) for pose in eef_poses])).to('cpu')
        assert eef_poses_tor.shape == w2c_poses_unaligned.shape, "Number of eef != Number of cam poses!"
        
        T, scale, J, R_L, t_L = compute_arm(eef_poses_tor.clone(), w2c_poses_unaligned.clone(), arm1=True)

        w2c_poses_unaligned[:,:3,3]=w2c_poses_unaligned[:,:3,3]*scale
        xyzs = xyzs*scale
        cam_pose, ptc = transpose_poses_ptc(w2c_poses_unaligned, xyzs, T)

        loss_info = f'{exp_name} trans loss: {t_L.mean()}, rot loss: {R_L.mean()}\n'
        print(loss_info)

        save_dict = {
            "xyzs": ptc.reshape(N, H, W, 3),
            'rgbs': rgbs.reshape(N, H, W, 3),
            'c2w_poses': cam_pose,
            'eef_poses': eef_poses_tor.numpy()
        }
        
        torch.save(save_dict, caliberated_out_file)
        if visualize:
            plot_eef_and_camera_poses(eef_poses_tor.numpy(), cam_pose, ptc[::visualize_stride], rgb=rgbs.reshape(-1, 3)[::visualize_stride])
        return ptc.reshape(N, H, W, 3),  rgbs.reshape(N, H, W, 3), cam_pose, eef_poses_tor.numpy(), masks
    else:
        if visualize:
            plot_eef_and_camera_poses(w2c_poses_unaligned.numpy(), w2c_poses_unaligned.numpy(), xyzs[::visualize_stride], rgb=rgbs.reshape(-1, 3)[::visualize_stride])
        return xyzs.reshape(N, H, W, 3),  rgbs.reshape(N, H, W, 3), w2c_poses_unaligned.numpy(), None, masks

# if __name__ == "__main__":
#     exp_name = "1"
#     data_o = read_pkl(f"data/{exp_name}.pkl")
#     temp_dir = save_images_to_temp(data_o['rgb'][..., ::-1])

#     out_file = running_dust3r(out_file=f"output/dust3r_raw/{exp_name}.pth", file_paths=get_sorted_png_files(temp_dir), output_raw=True)
#     shutil.rmtree(temp_dir)
#     print(f"Temporary directory {temp_dir} deleted.")