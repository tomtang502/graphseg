import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def viz_ptc(xyzs, rgbs, labels, cmap_skip=10, cmap_name='hsv', marker_size=1, label_size=1.5, alpha=0.5):
    # ---------------------------------------------------------
    # 2) Segmentation overlay
    #    - Only include points with label != 0
    #    - Color by label, with partial opacity
    # ---------------------------------------------------------

    num_classes = len(np.unique(labels))
    cmap = plt.get_cmap(cmap_name, num_classes)
    colors_rgba = [cmap((i * cmap_skip) % num_classes) for i in range(num_classes)]
    colors_rgba[0] = [128/255., 128/255., 128/255., 0.]
    
    point_colors = alpha*rgbs + (1-alpha)*(np.array([colors_rgba[label] for label in labels])[:, :-1])

    # Pick a discrete color mapping for classes. 
    # For small integer classes, you can simply feed them to Plotly as “color=overlay_labels”
    # plus a discrete color scale. Or you can manually map them to colors.
    trace_overlay = go.Scatter3d(
        x=xyzs[:, 0],
        y=xyzs[:, 1],
        z=xyzs[:, 2],
        mode='markers',
        marker=dict(
            size=label_size,  # slightly bigger if you want the segmentation to stand out
            # color can be directly set to your labels with a colorscale:
            color=[f'rgb({r},{g},{b})' for r, g, b in point_colors],
            opacity=0.95,         # semi-transparent overlay
        ),
        name='Segmentation Overlay'
    )

    # ---------------------------------------------------------
    # Combine and show
    # ---------------------------------------------------------
    fig = go.Figure(data=[trace_overlay])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()
    

# def viz_ptc(xyzs, rgbs, labels, cmap_skip=10, cmap_name='hsv', marker_size=1, label_size=1.5):
        
#     # ---------------------------------------------------------
#     # 1) Base layer: scatter3d with the raw RGB color
#     # ---------------------------------------------------------

#     trace_base = go.Scatter3d(
#         x=xyzs[:, 0],
#         y=xyzs[:, 1],
#         z=xyzs[:, 2],
#         mode='markers',
#         marker=dict(
#             size=marker_size,
#             # convert your (r, g, b) into Plotly-readable color strings (assuming 0..255):
#             color=[f'rgb({r},{g},{b})' for r, g, b in rgbs]
#         ),
#         name='Base Point Cloud'
#     )

#     # ---------------------------------------------------------
#     # 2) Segmentation overlay
#     #    - Only include points with label != 0
#     #    - Color by label, with partial opacity
#     # ---------------------------------------------------------

#     num_classes = len(np.unique(labels))
#     cmap = plt.get_cmap(cmap_name, num_classes)
#     colors_rgba = [cmap((i * cmap_skip) % num_classes) for i in range(num_classes)]
#     def mpl_rgba_to_plotly_str(rgba):
#         r, g, b, _ = rgba
#         return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
#     plotly_colors = [mpl_rgba_to_plotly_str(rgba) for rgba in colors_rgba]
#     point_colors = np.array([plotly_colors[label] for label in labels]) 
    
#     overlay_mask = (labels != 0)
#     overlay_xyzs = xyzs[overlay_mask]
#     overlay_point_colors = point_colors[overlay_mask]

#     # Pick a discrete color mapping for classes. 
#     # For small integer classes, you can simply feed them to Plotly as “color=overlay_labels”
#     # plus a discrete color scale. Or you can manually map them to colors.
#     trace_overlay = go.Scatter3d(
#         x=overlay_xyzs[:, 0],
#         y=overlay_xyzs[:, 1],
#         z=overlay_xyzs[:, 2],
#         mode='markers',
#         marker=dict(
#             size=label_size,  # slightly bigger if you want the segmentation to stand out
#             # color can be directly set to your labels with a colorscale:
#             color=overlay_point_colors,
#             opacity=0.95,         # semi-transparent overlay
#         ),
#         name='Segmentation Overlay'
#     )

#     # ---------------------------------------------------------
#     # Combine and show
#     # ---------------------------------------------------------
#     fig = go.Figure(data=[trace_base, trace_overlay])
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z'
#         )
#     )
#     fig.show()

def plot_eef_and_camera_poses(eef_poses, cam_poses, point_cloud, rgb):
    """
    Plots end-effector, camera poses, and a point cloud (with per-point RGB) in the robot base frame in 3D using Plotly.
    
    Parameters
    ----------
    eef_poses : ndarray of shape (n, 4, 4)
        Homogeneous transformations for end-effector (base->EEF).
    cam_poses : ndarray of shape (n, 4, 4)
        Homogeneous transformations for camera (base->camera).
    point_cloud : ndarray of shape (m, 3)
        3D points in base frame (or whichever frame you intend to visualize).
    rgb : ndarray of shape (m, 3)
        Per-point colors in the form [R, G, B], typically either 0-1 floats or 0-255 ints.
    """

    # --- Original code unchanged: ---
    # Extract positions from the transformation matrices
    eef_positions = eef_poses[:, :3, 3]
    cam_positions = cam_poses[:, :3, 3]

    # Create 3D scatter for EEF
    trace_eef = go.Scatter3d(
        x=eef_positions[:, 0],
        y=eef_positions[:, 1],
        z=eef_positions[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),  # EEF in blue
        name='End Effector'
    )
    
    # Create 3D scatter for camera
    trace_cam = go.Scatter3d(
        x=cam_positions[:, 0],
        y=cam_positions[:, 1],
        z=cam_positions[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),  # Camera in red
        name='Camera'
    )
    # --------------------------------

    # Convert each (R,G,B) in 'rgb' to a Plotly-compatible color string, e.g. "rgb(255, 120, 60)"
    # Adjust logic if your rgb array is [0..255], or [0..1]
    color_strings = []
    for (r_val, g_val, b_val) in rgb:
        # If rgb is in [0..1], multiply by 255
        # If rgb is already [0..255], omit the multiplication.
        r = int(r_val * 255)  
        g = int(g_val * 255)
        b = int(b_val * 255)
        color_strings.append(f"rgb({r},{g},{b})")

    # Create a new 3D scatter trace for the point cloud
    trace_cloud = go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1, color=color_strings),  # Per-point color
        name='Point Cloud'
    )

    # Combine all traces into one figure
    fig = go.Figure(data=[trace_eef, trace_cam, trace_cloud])
    
    # Axis labels and layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='EEF, Camera Poses, and Point Cloud in Base Frame'
    )

    fig.show()