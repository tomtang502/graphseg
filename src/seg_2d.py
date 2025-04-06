import torch, time, pickle, os
from PIL import Image
import numpy as np

from src.visualize.viz_masks import *
import PIL.Image
from lang_sam import LangSAM
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

measure_perf = False
        
def seg_images(sam2_model, rgb_images, viz_rgb=False, viz_segs=False):
    rgb_images = rgb_images.copy()
    if viz_rgb and not measure_perf:
        visualize_images(rgb_images)
        
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model,
        points_per_side=48, # num pixel one side / 10
        points_per_batch=16,
        pred_iou_thresh = 0.7,
        stability_score_thresh = 0.7, 
        box_nms_thresh=0.95,
        min_mask_region_area=50,
        multimask_output=False,
    )
    
    res_masks = []
    for image in rgb_images:
        masks = mask_generator.generate(image.copy())
        res_masks.append(masks)
        
    res_masks = [[seg['segmentation'] for seg in img_res] for img_res in res_masks]
    
    if viz_segs and not measure_perf:
        visualize_images_with_masks(rgb_images, res_masks)
    
    return res_masks

def seg_table(langsam_model, rgb_images, text_prompt="bacground wall and the table", batch_size = 10, viz=False):
    rgb_images = rgb_images.copy()
    image_pils = [Image.fromarray(img, 'RGB') for img in rgb_images]
    text_prompts = [text_prompt] * len(image_pils)
    
    results_all = []
    for i in range(0, len(image_pils), batch_size):
        batch_images = image_pils[i:i+batch_size]
        batch_texts = text_prompts[i:i+batch_size]
        batch_results = langsam_model.predict(batch_images, batch_texts)
        results_all.extend(batch_results)
    
    table_masks = [results_all[i]['masks'] for i in range(len(results_all))]
    
    if viz and not measure_perf:
        visualize_images_with_masks(rgb_images, table_masks)
    return table_masks

def resize_boolean_mask(mask_bool, new_h, new_w):
    """
    Resize a boolean mask using nearest neighbor interpolation so that
    the discrete True/False values remain intact.
    """
    # Step 1: Convert to uint8 for PIL (0 or 1)
    mask_uint8 = mask_bool.astype(np.uint8)

    # Step 2: Convert to a PIL image (mode 'L' => 8-bit grayscale)
    pil_mask = Image.fromarray(mask_uint8, mode='L')

    # Step 3: Resize with nearest-neighbor
    pil_mask_resized = pil_mask.resize((new_w, new_h), Image.NEAREST)

    # Step 4: Convert back to NumPy array, then to boolean
    mask_resized_uint8 = np.array(pil_mask_resized)
    mask_resized_bool = mask_resized_uint8 > 0

    return mask_resized_bool


if __name__ == "__main__":
    # size = 512
    output_dir = "output/seg_raw"
    exp_name = 'kitchen'
    data_dir = 'data'
    pkl_file_path = f"{data_dir}/{exp_name}_z1.pkl"
    rgb_meta = read_pkl(pkl_file_path)
    rgb_images = rgb_meta['rgb'][..., ::-1]
    H_resize, W_resize = 384, 512
    # rgb_images = np.stack([_resize_using_pil(img, size) for img in rgb_images])
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[Using device {device}]")
    # << ---- performance measuremets ---- >> #
    if measure_perf:
        start_time = time.perf_counter()
    # ---- performance measuremets ---- #
    
    langsam_model = LangSAM(sam_type="sam2.1_hiera_large", device=device)
    table_segs = seg_table(langsam_model, rgb_images=rgb_images, viz=True)
    
    # ---- performance measuremets ---- #
    if measure_perf:
        end_time = time.perf_counter()
        runtime_model1 = end_time - start_time
        if torch.cuda.is_available():
            vram_model1 = torch.cuda.max_memory_allocated(device)
            print(f"First model runtime: {runtime_model1:.4f} seconds, VRAM peak: {vram_model1/1024**2:.2f} MB")
        else:
            print(f"First model runtime: {runtime_model1:.4f} seconds")
    # << ---- performance measuremets ---- >> #
    
    # << ---- performance measuremets ---- >> #
    if measure_perf:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
    # ---- performance measuremets ---- #
    
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    
    imgs_segs = seg_images(sam2_model=sam2_model, rgb_images=rgb_images, viz_rgb=True, viz_segs=True)
    
    # ---- performance measuremets ---- #
    if measure_perf:
        end_time = time.perf_counter()
        runtime_model2 = end_time - start_time
        if torch.cuda.is_available():
            vram_model2 = torch.cuda.max_memory_allocated(device)
            print(f"Second model runtime: {runtime_model2:.4f} seconds, VRAM peak: {vram_model2/1024**2:.2f} MB")
        else:
            print(f"Second model runtime: {runtime_model2:.4f} seconds")
            
        print(f"Combined runtime: {runtime_model1+runtime_model2:.4f} seconds")
    # << ---- performance measuremets ---- >> #
    
    masks_meta = {
        'rgbs': rgb_images,
        'obj_segs': imgs_segs,
        'table_segs': table_segs,
    }
    
    for i in range(len(masks_meta['table_segs'])):
        masks_meta['table_segs'][i] = np.bitwise_or.reduce(masks_meta['table_segs'][i].astype(np.bool), axis=0)
        masks_meta['table_segs'][i] = resize_boolean_mask(masks_meta['table_segs'][i], H_resize, W_resize)[np.newaxis, ...]
    
    for i in range(len(masks_meta['obj_segs'])):
        for j in range(len(masks_meta['obj_segs'][i])):
            masks_meta['obj_segs'][i][j] = resize_boolean_mask(masks_meta['obj_segs'][i][j], H_resize, W_resize)
        masks_meta['obj_segs'][i] = np.stack(masks_meta['obj_segs'][i])
    
    
    
    with open(f"{output_dir}/{exp_name}_masks.pkl", "wb") as f:
        pickle.dump(masks_meta, f)
    
    