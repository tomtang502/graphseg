import pickle, torch, math, cudf
import numpy as np
import plotly.graph_objects as go
from cuml.cluster import DBSCAN
from src.visualize.viz_masks import visualize_images_with_composed_masks_interactive
from src.helpers.uf_tool import UnionFind
from src.helpers.ptc_dist import one_way_nn_distances
from tqdm import tqdm

def compresse_mask_ids(seg_masks):
    if isinstance(seg_masks, np.ndarray):
        unique_ids = np.unique(seg_masks)
    else:
        unique_ids = torch.unique(seg_masks)
    new_id = 0
    for i in unique_ids:
        seg_masks[seg_masks==i] = new_id
        new_id += 1
    return seg_masks

def plot_point_cloud(xyz, rgb, marker_size=1, opacity=.8):
    """
    Plots a 3D point cloud using Plotly.
    
    Parameters:
      xyz : np.ndarray of shape (n, 3)
          Array containing x, y, and z coordinates.
      rgb : np.ndarray of shape (n, 3)
          Array containing RGB color values (assumed to be in the 0-255 range) for each point.
      marker_size : int, optional
          Size of the markers (default is 1).
      opacity : float, optional
          Opacity of the markers (default is 0.8).
    
    Returns:
      fig : plotly.graph_objects.Figure
          The resulting Plotly figure.
    
    Note:
      With over 2.3 million points, rendering might be heavy. Consider downsampling or using a subset
      for interactive visualization if needed.
    """
    # Verify that the input arrays have the correct shape
    if xyz.shape[1] != 3 or rgb.shape[1] != 3:
        raise ValueError("Both xyz and rgb arrays must have shape (n, 3).")
    
    # Extract coordinates
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    # Convert RGB values to Plotly color strings.
    colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in rgb]
    
    # Create the scatter3d plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=colors,
            opacity=opacity
        )
    )])
    
    # Update layout to label axes
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    
    fig.show()
    # return fig
    
def filter_gc(
    seg_raw, 
    filter_invalid_depth=True, 
    grid_size=10, 
    DBSCAN_eps = 0.01,          # maximum distance between two samples to be considered neighbors
    DBSCAN_min_samples = 60,   # minimum number of neighbors to form a dense region
    one_way_group_thresh = 5e-5,
    process_pts_max = 15000, # adjust based on GPU VRAM, this is for 16GB laptop 4090
    device='cuda'):
    
    xyzs, rgbs = seg_raw['xyzs'], seg_raw['rgbs']
    if 'depth' in seg_raw:
        depths = seg_raw['depths']
    else:
        depths = None
    seg_masks = seg_raw['masks']

    if filter_invalid_depth:
        invalid_pt_mask = np.logical_not(depths > 0)
        seg_masks[invalid_pt_mask] = 0

    N, H, W = seg_masks.shape
    uids = np.unique(seg_masks)
    min_mask_thresh = len(seg_masks.reshape(-1))//((H//grid_size)*(W//grid_size))
    print(f"min_mask_thresh: {min_mask_thresh}")
    for uid in uids[1:]:
        if np.sum(seg_masks==uid)<min_mask_thresh:
            seg_masks[seg_masks==uid] = 0

    seg_masks = compresse_mask_ids(seg_masks)
    uids = np.unique(seg_masks)

    filtered_seg_masks = seg_masks.copy()
    hids, nids, wids = np.meshgrid(list(range(H)), list(range(N)), list(range(W)))
    # visualize_images_with_composed_masks_interactive(rgbs, seg_masks, vertex_cnt=len(uids))
    for uid in uids[1:20]:
        xyz = xyzs[seg_masks==uid].reshape(-1, 3)
        nid, hid, wid = nids[seg_masks==uid].reshape(-1,), hids[seg_masks==uid].reshape(-1,), wids[seg_masks==uid].reshape(-1,)
       
        df_xyz = cudf.DataFrame(xyz, columns=['x', 'y', 'z'])
        dbscan = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples)
        dbscan.fit(df_xyz)
        labels = dbscan.labels_ # labels for each point (-1 means noise)
        # Filter out noise points:
        outlier_mask = (labels == -1).to_numpy()
        nid, hid, wid = nid[outlier_mask], hid[outlier_mask], wid[outlier_mask]
        filtered_seg_masks[nid, hid, wid] = 0
        
    xyzs_tr = torch.tensor(xyzs.reshape(-1, 3), device=device)
    semi_consist_masks_tr = torch.tensor(filtered_seg_masks.reshape(-1,), device=device, dtype=torch.int32)
    
    semi_consist_masks_tr = compresse_mask_ids(semi_consist_masks_tr).to(torch.uint16)
    new_vertices = torch.unique(semi_consist_masks_tr).cpu().tolist()
    uf = UnionFind(len(new_vertices))
    # skip 0 background class
    for i, u in tqdm(enumerate(new_vertices[1:]), total=len(new_vertices[1:])):
        for v in (new_vertices[i+1:]):
            mask_u, mask_v = semi_consist_masks_tr==u, semi_consist_masks_tr==v
            xyz_u, xyz_v = xyzs_tr[mask_u], xyzs_tr[mask_v]
            np_u, np_v = xyz_u.shape[0], xyz_v.shape[0]
            if np_u >= process_pts_max:
                xyz_u = xyz_u[::(math.ceil(np_u/process_pts_max))]
            if np_v >= process_pts_max:
                xyz_v = xyz_v[::(math.ceil(np_v/process_pts_max))]
            
            dist_1 = torch.mean(one_way_nn_distances(xyz_u, xyz_v))
            dist_2 = torch.mean(one_way_nn_distances(xyz_v, xyz_u))
            
            dist_w = torch.min(dist_1, dist_2)
            # print(dist_w)
            if u != v and dist_w <= one_way_group_thresh:
                uf.union(u, v)

    res_masks = np.zeros_like(filtered_seg_masks)
    for i in new_vertices:
        res_masks[filtered_seg_masks==i] = uf.find(i)
    res_masks = compresse_mask_ids(res_masks)
    return res_masks

if __name__ == "__main__":
            
    if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 2, 12, 189
    with open("output/seg_res_gc1/scene_0022_26_s1.pkl", 'rb') as file:
        seg_raw = pickle.load(file)
    res_masks = filter_gc(seg_raw=seg_raw, device=device)
    visualize_images_with_composed_masks_interactive(seg_raw['rgbs'], res_masks, len(np.unique(res_masks)), cmap_skip=10)


