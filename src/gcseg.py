import torch
from src.helpers.uf_tool import UnionFind
from src.helpers.data_process import ensure_tensor
from src.visualize.viz_masks import *
import matplotlib.pyplot as plt

def get_dist_mask(xyzs, rgbs, poses, depths=None, use_depth=False, dist_thresh=0.38, viz=False):
    if use_depth:
        dist_masks = [d_m>dist_thresh for d_m in depths]
    else:
        dist_from_cam = [xyzs[i]-poses[i][:3, 3] for i in range(len(xyzs))]
        dists_l2 = [torch.norm(dist_from_cam[i], dim=2) for i in range(len(xyzs))]
        dist_masks = [d_m>dist_thresh for d_m in dists_l2]
    if viz:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        cmap = plt.cm.jet
        cmap.set_bad(color='black')
        for i in range(len(rgbs)):
            img = rgbs[i]
            if use_depth:
                depth = depths[i].cpu()
            else:
                depth = dists_l2[i].cpu()

            depth_masked = np.ma.masked_where(depth > dist_thresh, depth)
            
            # Display the image
            axes[i].imshow(img)
            # Overlay the depth map with a colormap (using 'jet' here) and some transparency
            im = axes[i].imshow(depth_masked, cmap=cmap, alpha=0.5)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')

        # Add a colorbar to one of the subplots (or collectively to all)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='Depth')
        plt.show()
    return dist_masks

def get_bg_mask(xyzs, rgbs, poses, table_masks, depths=None, use_depth=False, dist_thresh=0.38, viz_dist=False, viz_bg=False):
    dist_masks = get_dist_mask(ensure_tensor(xyzs), rgbs, ensure_tensor(poses), 
                               None if depths is None else ensure_tensor(depths), use_depth=use_depth, dist_thresh=dist_thresh, viz=viz_dist)
    bg_mask = [np.logical_or(table_masks[i][0], dist_masks[i].cpu().numpy()) for i in range(len(table_masks))]
    if viz_bg:
        if isinstance(rgbs, list):
            rgbs = np.stack(rgbs)
        visualize_images_with_masks(rgbs, bg_mask)
    return bg_mask

def compose_seg_masks(rgb_images, obj_masks, bg_mask, overlap_ratio_thresh=0.09):
    seg_vertex_cnt = 1 # 0 reserved for bg segs
    seg_masks = [np.zeros(obj_masks[0][0].shape, dtype=np.uint16) for _ in obj_masks]
    mid_to_img_id = dict()
    mid_to_mask = dict()
    for img_id in range(len(rgb_images)):
        masks_of_image = obj_masks[img_id]
        img_mask_idx = 0
        for mask in masks_of_image:
            overlap_count = np.sum(np.logical_and(mask, bg_mask[img_id]))
            mask_count = np.sum(mask)
            overlap_ratio = 0.0 if mask_count == 0 else overlap_count / mask_count
            if overlap_ratio < overlap_ratio_thresh:
                seg_masks[img_id][mask] = seg_vertex_cnt
                mid_to_img_id[seg_vertex_cnt] = img_id
                mid_to_mask[seg_vertex_cnt] = mask.copy()
                seg_vertex_cnt += 1
                img_mask_idx += 1
        # apply bg _mask
        seg_masks[img_id][bg_mask[img_id]] = 0
    return seg_masks, seg_vertex_cnt, mid_to_img_id, mid_to_mask

def pairwise_intersection_numpy(masks: np.ndarray) -> np.ndarray:
    """
    Given a boolean array of shape (k, m, n),
    return the k x k matrix where the (i, j)-th entry
    is the size of the intersection between mask i and mask j.
    """
    # Flatten each mask into a 1D vector (0 or 1).
    # shape: (k, m*n)
    A = masks.reshape(masks.shape[0], -1).astype(np.int32)
    
    # The dot product of these flattened rows
    # yields the intersection counts.
    # shape: (k, k)
    intersection_matrix = A @ A.T
    
    return intersection_matrix

def build_edges(seg_masks, fm_raw, mid_to_mask, xyzs, conf_masks, device='cuda'):
    seg_masks = np.stack(seg_masks)
    seg_edge_hit, seg_edge_cost, seg_vertex_hit = dict(), dict(), dict()
    y_max, x_max = seg_masks[0].shape
    for fm_pair in fm_raw:
        img1_id, img2_id = fm_pair['img_idx_pair']
        kpt1, kpt2 = fm_pair['kpts_pair']
        certainty = fm_pair['certainty'].unsqueeze(-1)
        kpts_certainty = torch.cat((kpt1, kpt2, certainty), dim=1)
        kpts_certain = kpts_certainty[kpts_certainty[:, 4] >= 1.0][:, :-1].round().long()
        kpts_certain[:, 0] = kpts_certain[:, 0].clamp(0, x_max-1)
        kpts_certain[:, 1] = kpts_certain[:, 1].clamp(0, y_max-1)
        kpts_certain[:, 2] = kpts_certain[:, 2].clamp(0, x_max-1)
        kpts_certain[:, 3] = kpts_certain[:, 3].clamp(0, y_max-1)
        seg_ms1, seg_ms2 = seg_masks[img1_id], seg_masks[img2_id]
        ptc_ms1, ptc_ms2 = xyzs[img1_id], xyzs[img2_id]
        conf_ms1, conf_ms2 = conf_masks[img1_id], conf_masks[img2_id]
        c1, r1, c2, r2 = kpts_certain.T.to('cpu')
        us = seg_ms1[r1, c1]
        vs = seg_ms2[r2, c2]
        coords1 = ptc_ms1[r1, c1]
        coords1_valid = conf_ms1[r1, c1]
        coords2 = ptc_ms2[r2, c2]
        coords2_valid = conf_ms2[r2, c2]
        
        uv = np.stack((us, vs), axis=-1)
        non_bg_mask = torch.tensor((uv[:, 0] != 0) & (uv[:, 1] != 0)).to(device)
        both_valid = torch.logical_and(coords1_valid, coords2_valid)
        valid_mask = torch.logical_and(both_valid, non_bg_mask).cpu().numpy()
        # print(non_bg_mask.sum(), both_valid.sum(), valid_mask.shape, valid_mask.sum())

        diff = np.linalg.norm(coords2-coords1, axis=1)

        diff_sel = diff[valid_mask]
        uv_sel = uv[valid_mask]
        mean_distance = np.percentile(diff_sel, 25)
        diff_tsel_mask = diff_sel < mean_distance
        
        uv_sel = uv_sel[diff_tsel_mask]
        # diff_tsel = diff_sel[diff_tsel_mask]
        for u, v in uv_sel:
            if (u, v) not in seg_edge_hit:
                seg_edge_hit[(u, v)] = 0
            seg_edge_hit[(u, v)] += 1
            
            seg_vertex_hit[u] = seg_vertex_hit.get(u, 0) + 1
            seg_vertex_hit[v] = seg_vertex_hit.get(v, 0) + 1
    
    for (u, v) in seg_edge_hit.keys():
        seg_edge_cost[(u, v)] = seg_edge_hit[(u, v)] / min(seg_vertex_hit[u], seg_vertex_hit[v])
            
    # anti-mask to be developed
    # masks = np.stack([mid_to_mask[i] for i in range(1, len(mid_to_mask)+1)])
    # intersections = pairwise_intersection_numpy(masks)
    # for u in range(len(intersections)):
    #     for v in range(len(intersections)):
    #         if intersections[u][v] < 1:
    #             seg_edge_cost[(u+1, v+1)] = 0
                
    
    return seg_edge_cost



def uf_graph_contraction(seg_edge_cost, seg_vertex_cnt, mid_to_img_id, mid_to_mask, obj_masks, bg_mask, seg_group_thresh=132):
    uf = UnionFind(seg_vertex_cnt)
    for (u, v) in seg_edge_cost:
        cost = seg_edge_cost[(u, v)]
        if cost > seg_group_thresh:
            uf.union(u, v)

    root_verts_cnt = 1
    root_verts = dict()
    seg_masks_new = [np.zeros(obj_masks[0][0].shape, dtype=np.uint16) for _ in obj_masks]
    for i in range(1, seg_vertex_cnt):
        img_id = mid_to_img_id[i]
        root = uf.find(i)
        if root not in root_verts:
            root_verts[root] = root_verts_cnt
            root_verts_cnt += 1
        seg_masks_new[img_id][mid_to_mask[i]] = root_verts[root]
        
    for i in range(len(seg_masks_new)):
        seg_masks_new[i][bg_mask[i]] = 0
    return seg_masks_new, root_verts_cnt