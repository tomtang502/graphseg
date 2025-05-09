import os, torch, pickle, gc, pynvml
import numpy as np
import plotly.graph_objects as go

from src.seg_2d import seg_table, seg_images, resize_boolean_mask
from src.feature_match_2d import fm_images
from src.gcseg import (
    compose_seg_masks, 
    get_bg_mask, 
    build_edges, 
    uf_graph_contraction,
    visualize_images_with_composed_masks_interactive
)
from src.visualize.viz_ptc import viz_ptc
from src.pt_filter import(
    compresse_mask_ids,
    filter_gc
)
from src.helpers.data_process import get_grnet_data, read_pkl
from lang_sam import LangSAM
from romatch import roma_outdoor
from sam2.build_sam import build_sam2
from src.ptc_reconstruct import reconstruct_3d


pynvml.nvmlInit()
   
def print_gpu_usage(label=""):
    """
    Prints the GPU memory usage for GPU index 0.
    If you have multiple GPUs, adjust the index accordingly.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = mem_info.used / 1024**2   # MiB
    total = mem_info.total / 1024**2 # MiB
    if label:
        print(f"{label}: {used:.2f} MiB / {total:.2f} MiB used")
    else:
        print(f"GPU usage: {used:.2f} MiB / {total:.2f} MiB used")

def main(
    rgbs_o, # initial rgb file, should be >= 480 x 640 resolution
    xyzs, rgbs, poses, conf_3d_masks,
    save_name='demo',
    bg_prompt = "white table",
    dist_thresh = 0.7, # adjust this, depending on camera configuration.
    fm_numpoints = 10000,
    fm_2d_percentile = 80, 
    one_way_3d_group_thresh = 5e-5,
    viz_res = False,
    rerun_anyway = False,
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt",
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
    fm_out_dir = "output/fm_raw",
    seg_out_dir = "output/seg_raw",
    stage_1_dir = "output/seg_res_gc1",
    stage_2_dir = "output/seg_res_gc2",
    seg_group_thresh = None, # set this, otherwise use percentile
):
    print(rgbs_o.shape, rgbs.shape, xyzs.shape, poses.shape, conf_3d_masks.shape)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    """
    generate table masks and sam masks
    """
    print("Step 1: [Segmentation]")
    mask_save_file = f"{seg_out_dir}/{save_name}_masks.pkl"
    if rerun_anyway or (not os.path.exists(mask_save_file)):
        H_resize, W_resize = rgbs.shape[1], rgbs.shape[2]
        langsam_model = LangSAM(sam_type="sam2.1_hiera_large", device=device)
        table_segs = seg_table(langsam_model, 
                            rgb_images=rgbs_o,
                            text_prompt=bg_prompt, viz=False)
    
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        imgs_segs = seg_images(sam2_model=sam2_model, rgb_images=rgbs_o, viz_rgb=False, viz_segs=False)
        
        masks_meta = {
            'rgbs': rgbs,
            'obj_segs': imgs_segs,
            'table_segs': table_segs,
        }
        """
        resize if needed, specifically if point cloud and pose need to be predicted by sfm model with different resolution
        """
        for i in range(len(masks_meta['table_segs'])):
            masks_meta['table_segs'][i] = np.bitwise_or.reduce(masks_meta['table_segs'][i].astype(bool), axis=0)
            masks_meta['table_segs'][i] = resize_boolean_mask(masks_meta['table_segs'][i], H_resize, W_resize)[np.newaxis, ...]
        
        for i in range(len(masks_meta['obj_segs'])):
            for j in range(len(masks_meta['obj_segs'][i])):
                masks_meta['obj_segs'][i][j] = resize_boolean_mask(masks_meta['obj_segs'][i][j], H_resize, W_resize)
            masks_meta['obj_segs'][i] = np.stack(masks_meta['obj_segs'][i])
        with open(mask_save_file, "wb") as f:
            pickle.dump(masks_meta, f)
        
        del sam2_model
        del langsam_model
        gc.collect()
        torch.cuda.empty_cache()
        
    with open(mask_save_file, 'rb') as file:
        masks_meta = pickle.load(file)
    print_gpu_usage("After [1]")
    """
    run optical feature matching
    """
    print("Step 2: [FM]")
    fm_save_file = f"{fm_out_dir}/{save_name}_fm.pkl"
    if len(rgbs) > 5: 
        mode='bithresh'
    else:
        mode='complete'
    if rerun_anyway or (not os.path.exists(fm_save_file)):
        roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
        fm_images(roma_model=roma_model, poses=torch.tensor(poses), rgbs=rgbs, device=device, save_path=fm_save_file, 
                num_points=fm_numpoints, viz_feature_match=False, mode=mode,
            pose_edge_cost_thresh=1.5, bi_thresh_rad_thresh=1.2, time_single_roma=False)
        
        del roma_model
        torch.cuda.empty_cache()
        gc.collect()
    with open(fm_save_file, 'rb') as file:
            fm_raw = pickle.load(file)
    print_gpu_usage("After [2]")
    
    """
    generate dist masks and compose bg mask
    """
    print("Step 3: [BG Masks]")
    bg_mask = get_bg_mask(xyzs, rgbs, poses, masks_meta['table_segs'], use_depth=False, dist_thresh=dist_thresh, viz_bg=False, viz_dist=False)
    
    """
    compose object masks
    """
    seg_masks, seg_vertex_cnt, mid_to_img_id, mid_to_mask = compose_seg_masks(rgbs, masks_meta['obj_segs'], bg_mask)
    
    """
    build graph and run graph contraction
    """
    print("Step 4: [GC]")
    gc1_file_path = f"{stage_1_dir}/{save_name}_s1.pkl"
   
    if rerun_anyway or (not os.path.exists(gc1_file_path)):
        seg_edge_cost = build_edges(seg_masks, fm_raw, mid_to_mask=mid_to_mask, xyzs=xyzs, conf_masks=conf_3d_masks)
        if seg_group_thresh is None:
            seg_group_thresh = np.percentile(list(seg_edge_cost.values()), fm_2d_percentile)
        print(f"seg_group_thresh: {seg_group_thresh}")
        seg_masks_new, root_verts_cnt = uf_graph_contraction(seg_edge_cost, seg_vertex_cnt, 
                                            mid_to_img_id=mid_to_img_id, mid_to_mask=mid_to_mask, 
                                            obj_masks=masks_meta['obj_segs'], bg_mask=bg_mask,
                                            seg_group_thresh=seg_group_thresh)
        
        semi_consist_masks = np.stack(seg_masks_new)
        semi_consist_masks = compresse_mask_ids(semi_consist_masks)
       
        semi_consist_data = {
            'rgbs': rgbs,
            'bg_mask': bg_mask,
            # 'composed_mask': seg_masks, 
            'masks': semi_consist_masks,
            'xyzs': xyzs.numpy()
        }
        with open(gc1_file_path, "wb") as f:
            pickle.dump(semi_consist_data, f)
    else:
        with open(gc1_file_path, 'rb') as file:
            semi_consist_data = pickle.load(file)
    torch.cuda.empty_cache()
    # print(semi_consist_data['xyzs'].shape, semi_consist_data['xyzs'].type())
    """
    one way nearest neighbor
    """
    print("Step 5: [Filter and 1-way nearest neighbor graph contraction]")
    gc2_file_path = f"{stage_2_dir}/{save_name}_res.pkl"
    if rerun_anyway or (not os.path.exists(gc2_file_path)):
        res_masks = filter_gc(seg_raw=semi_consist_data, conf_3d_masks=conf_3d_masks, filter_invalid_depth=False, device=device, one_way_group_thresh=one_way_3d_group_thresh)
        with open(gc2_file_path, "wb") as f:
            pickle.dump(res_masks, f)
    else:
        with open(gc2_file_path, 'rb') as file:
            res_masks = pickle.load(file)
    
    # """
    # visualize results
    # """
    
    if viz_res:
        print("plotting")
        visualize_images_with_composed_masks_interactive(rgbs, res_masks, len(np.unique(res_masks)), cmap_mask='hsv', cmap_skip=15)
        

        sample_steps=10    
        xyzs, rgbs = xyzs.reshape(-1, 3)[::sample_steps], rgbs.reshape(-1, 3)[::sample_steps]
        res_masks = res_masks.reshape(-1,)[::sample_steps]
        
        viz_ptc(xyzs, rgbs, res_masks, cmap_name='tab20', cmap_skip=15)
    
if __name__ == "__main__":
    
    dust3r_ckpt_path="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    exp_name = "tools0"
    data_o = read_pkl(f"data/{exp_name}.pkl")
    rgbs_o = data_o['rgb'][..., ::-1] # a bgr camera output should be reverse to rgb
    eef_poses = data_o['poses'] # end effector poses, theoretically this is not needed, if this is None, also set caliberate below to be False
    rerun = False
    xyzs, rgbs, cam_poses, _, masks = reconstruct_3d(exp_name=exp_name, rgbs=rgbs_o, eef_poses=eef_poses, ckpt_path=dust3r_ckpt_path, caliberate=True, rerun_anyway=rerun)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Notice if cuda run out of memory here, it is possible that the 3D tensors and models are not cleaned correctly from gpu vram, 
    # simply run this script again will initialize the segmentation based on processed 3D scene.
    
    main(rgbs_o=rgbs_o, xyzs=xyzs, rgbs=rgbs, conf_3d_masks=masks, poses=cam_poses, save_name=exp_name, rerun_anyway=rerun)

    