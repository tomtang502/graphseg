import os, torch, pickle, gc, pynvml, argparse
import numpy as np

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
from src.helpers.data_process import get_grnet_data
from lang_sam import LangSAM
from romatch import roma_outdoor
from sam2.build_sam import build_sam2


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


def main(scene_id=0,
    bg_prompt = "dark green table",
    dist_thresh = 0.7, # adjust this, depending on camera configuration.
    fm_numpoints = 10000,
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt",
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
    fm_out_dir = "output/fm_raw",
    seg_out_dir = "output/seg_raw",
    stage_1_dir = "output/seg_res_gc1",
    stage_2_dir = "output/seg_res_gc2",
    root_dir = "../ri_other/seg3d/graspnet/grnet_processed",
    seg_group_thresh = None, # set this, otherwise use percentile
    step_size = 26
):
    scene_name = f"scene_{scene_id:04d}"
    print(scene_name)
    rgbs, xyzs, poses, _, depths = get_grnet_data(root_dir, scene_name, step_size)
    print(rgbs.shape, xyzs.shape, poses.shape)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    """
    generate table masks and sam masks
    """
    print("Step 1: [Segmentation]")
    mask_save_file = f"{seg_out_dir}/{scene_name}_{step_size}_masks.pkl"
    if not os.path.exists(mask_save_file):
        H_resize, W_resize = 480, 640
        langsam_model = LangSAM(sam_type="sam2.1_hiera_large", device=device)
        table_segs = seg_table(langsam_model, 
                            rgb_images=rgbs,
                            text_prompt=bg_prompt, viz=False)
    
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        imgs_segs = seg_images(sam2_model=sam2_model, rgb_images=rgbs, viz_rgb=False, viz_segs=False)
        
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
    fm_save_file = f"{fm_out_dir}/{scene_name}_{step_size}_fm.pkl"
    if len(rgbs) > 8: 
        mode='bithresh'
    else:
        mode='complete'
    if not os.path.exists(fm_save_file):
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
    gc1_file_path = f"{stage_1_dir}/{scene_name}_{step_size}_s1.pkl"
    if not os.path.exists(gc1_file_path):
        seg_edge_cost = build_edges(seg_masks, fm_raw, mid_to_mask=mid_to_mask, xyzs=xyzs)
        if seg_group_thresh is None:
            seg_group_thresh = np.percentile(list(seg_edge_cost.values()), 78)
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
            'xyzs': xyzs,
            'depths': depths,
        }
        with open(gc1_file_path, "wb") as f:
            pickle.dump(semi_consist_data, f)
    else:
        with open(gc1_file_path, 'rb') as file:
            semi_consist_data = pickle.load(file)
    torch.cuda.empty_cache()
    """
    one way nearest neighbor
    """
    print("Step 5: [Filter and 1-way nearest neighbor graph contraction]")
    gc2_file_path = f"{stage_2_dir}/{scene_name}_{step_size}_res.pkl"
    if not os.path.exists(gc2_file_path):
        res_masks = filter_gc(seg_raw=semi_consist_data, filter_invalid_depth=False, device=device)
        with open(gc2_file_path, "wb") as f:
            pickle.dump(res_masks, f)
    else:
        with open(gc2_file_path, 'rb') as file:
            res_masks = pickle.load(file)
    
    # """
    # visualize results
    # """
    # visualize_images_with_composed_masks_interactive(rgbs, res_masks, len(np.unique(res_masks)), cmap_skip=10)
    # print("plotting")

    # sample_steps=10    
    # xyzs, rgbs = xyzs.reshape(-1, 3)[::sample_steps], rgbs.reshape(-1, 3)[::sample_steps]
    # res_masks = res_masks.reshape(-1,)[::sample_steps]
    
    # viz_ptc(xyzs, rgbs, res_masks, cmap_skip=10)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo for graspnet-1B dataset")
    parser.add_argument('--scene_id', type=int, default=1, help='scene_id')
    parser.add_argument('--step_size', type=int, default=26, help='scene_id')
    args = parser.parse_args()
    main(scene_id=args.scene_id, step_size=args.step_size)

    