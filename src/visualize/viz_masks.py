import pickle, math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(16325)
def show_anns(anns, borders=True, all_green=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        if all_green:
            color_mask = np.concatenate([[0., 1., 0.], [0.5]])
        else:
            color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def read_pkl(file_path):
    """
    Reads a pickle file and returns its content.

    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        Object: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def visualize_images(images):
    # assert images.shape == (10, 480, 640, 3), "Input array must have shape (10, 480, 640, 3)"
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 7 columns
    fig.suptitle("Visualization of 10 Images", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])  # Display image
        ax.axis("off")  # Hide axes
    
    plt.tight_layout()
    plt.show()
    
def visualize_pair_of_images(image1, image2):
    """
    Displays two images side by side using Matplotlib.

    Parameters:
    - image1: First image as a NumPy array.
    - image2: Second image as a NumPy array.
    """
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    # Ensure both images have the same dimensions
    assert image1.shape == image2.shape, "Both images must have the same dimensions"

    # Create a figure with two subplots in one row
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    axes[0].imshow(image1)
    axes[0].set_title('Image 1')
    axes[0].axis('off')  # Hide axes

    # Display the second image
    axes[1].imshow(image2)
    axes[1].set_title('Image 2')
    axes[1].axis('off')  # Hide axes

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        
def show_masks_over_rgb_image(image, masks, cmap_mask='jet', alpha=0.5, figsize=(8, 6)):
    """
    Display an RGB image with overlaid masks using matplotlib.

    Parameters:
    - image: 3D numpy array of shape (480, 640, 3), representing the base RGB image.
    - masks: numpy array of shape (n, 480, 640) containing n binary masks.
    - cmap_mask: colormap for the masks overlay.
    - alpha: transparency level for the masks overlay.
    - figsize: tuple specifying the figure size.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)  # Display the RGB image without a colormap
    
    for mask in masks:
        # Create a masked array: only display areas where mask is nonzero.
        masked_mask = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked_mask, cmap=cmap_mask, alpha=alpha)
    
    plt.axis('off')
    plt.show()
    

def visualize_images_with_masks(images, masks_list, cmap_mask='hsv', alpha=0.5, cols=5, figsize=(15, 6)):
    """
    Visualize multiple RGB images with overlaid masks on the same matplotlib figure.

    Parameters:
    - images: numpy array of shape (n, height, width, 3), where each image is in RGB.
    - masks_list: list of length n, where each element is a list or array of masks
                  corresponding to that image (each mask should have shape (height, width)).
    - cmap_mask: colormap to use for the mask overlay.
    - alpha: transparency level for the mask overlay.
    - cols: number of columns for the subplot grid.
    - figsize: size of the overall matplotlib figure.
    """
    n = images.shape[0]
    rows = math.ceil(n / cols)
    
    max_masks = max(len(masks) for masks in masks_list)
    
    # Generate a list of colors using a discrete colormap (here using HSV).
    # This produces 'max_masks' distinct colors.
    cmap = plt.cm.get_cmap(cmap_mask, max_masks)
    colors = [cmap(i) for i in range(max_masks)]
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # In case there is only one row or column, ensure axes is a flat array.
    axes = np.array(axes).reshape(-1)
    
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        if isinstance(masks_list[i], list) or isinstance(masks_list[i], np.ndarray) and len(masks_list[i].shape) == 3:
            for idx, mask in enumerate(masks_list[i]):
                # Create an overlay image with an alpha channel.
                H, W = mask.shape
                overlay = np.zeros((H, W, 4))
                
                # Use the preset color for this mask. Replace the alpha channel with the given alpha.
                r, g, b, _ = colors[idx]
                overlay[mask.astype(bool)] = [r, g, b, alpha]
                
                # Display the colored overlay.
                ax.imshow(overlay)
        else:
            mask = masks_list[i]
            H, W = mask.shape
            overlay = np.zeros((H, W, 4))
            
            # Use the preset color for this mask. Replace the alpha channel with the given alpha.
            r, g, b, _ = colors[0]
            overlay[mask.astype(bool)] = [r, g, b, alpha]
            
            # Display the colored overlay.
            ax.imshow(overlay)
            
        ax.axis('off')
    
    # Hide any unused subplots if n is not a perfect multiple of cols
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig("sam2_output.png")
    plt.show()

def visualize_images_with_composed_masks(images, masks_list, vertex_cnt, cmap_mask='hsv', alpha=0.5, cols=5, figsize=(15, 6)):
    """
    Visualize multiple RGB images with overlaid masks on the same matplotlib figure.

    Parameters:
    - images: numpy array of shape (n, height, width, 3), where each image is in RGB.
    - masks_list: list of length n, where each element is a list or array of masks
                  corresponding to that image (each mask should have shape (height, width)).
    - cmap_mask: colormap to use for the mask overlay.
    - alpha: transparency level for the mask overlay.
    - cols: number of columns for the subplot grid.
    - figsize: size of the overall matplotlib figure.
    """
    n = images.shape[0]
    rows = math.ceil(n / cols)

    cmap = plt.get_cmap(cmap_mask, vertex_cnt*10)
    colors = [cmap(i) for i in range(vertex_cnt*10)]
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # In case there is only one row or column, ensure axes is a flat array.
    axes = np.array(axes).reshape(-1)
    
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
     
        mask = masks_list[i]
        H, W = mask.shape
        overlay = np.zeros((H, W, 4))
        unique_labels = np.unique(mask)
        
        for lbl in unique_labels:
            # Use the preset color for this mask. Replace the alpha channel with the given alpha.   
            if lbl == 0:
                r, g, b = 0, 0, 0
            else:
                r, g, b, _ = colors[lbl*10]

            overlay[mask == lbl] = [r, g, b, alpha]

        ax.imshow(overlay)
            
        ax.axis('off')
    
    # Hide any unused subplots if n is not a perfect multiple of cols
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    


def visualize_images_with_composed_masks_interactive(images, masks_list, vertex_cnt, cmap_mask='tab20', cmap_skip=1, alpha=0.8, cols=5, figsize=(15, 6)):
    """
    Visualize multiple RGB images with overlaid masks and print mask value upon clicking.
    
    Parameters:
    - images: numpy array of shape (n, height, width, 3) with RGB images.
    - masks_list: list of n masks (each mask is of shape (height, width)).
    - cmap_mask: colormap to use for the mask overlay.
    - alpha: transparency for the mask overlay.
    - cols: number of columns for the subplot grid.
    - figsize: size of the overall matplotlib figure.
    """
    n = images.shape[0]
    rows = math.ceil(n / cols)

    cmap = plt.get_cmap(cmap_mask, vertex_cnt)
    colors = [cmap(i*cmap_skip%vertex_cnt) for i in range(vertex_cnt)]
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    # Create a mapping from each axis to its corresponding mask.
    ax_to_mask = {}

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
     
        mask = masks_list[i]
        # Store the mask associated with this axis.
        ax_to_mask[ax] = mask
        H, W = mask.shape
        overlay = np.zeros((H, W, 4))
        unique_labels = np.unique(mask)
        for lbl in unique_labels:
            # Use a preset color for this mask; for label 0, use black.
            if lbl == 0:
                r, g, b = 0, 0, 0
            else:
                r, g, b, _ = colors[lbl]
            overlay[mask == lbl] = [r, g, b, alpha]

        ax.imshow(overlay)
        ax.axis('off')
    
    # Hide any unused subplots if n is not a perfect multiple of cols.
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    def on_click(event):
        # Check if the click occurred on an axis that has an associated mask.
        if event.inaxes in ax_to_mask and event.xdata is not None and event.ydata is not None:
            mask = ax_to_mask[event.inaxes]
            # xdata and ydata correspond to the image coordinate system (x is column, y is row).
            x = int(event.xdata)
            y = int(event.ydata)
            # Validate the coordinates.
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask_value = mask[y, x]
                print(f"Clicked at ({x}, {y}) - Mask value: {mask_value}")
            else:
                print("Clicked outside the image bounds.")

    # Connect the mouse click event to the on_click callback.
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    print("[Due to color scheme, method classified as different class may receive similar color, click on the segment to see which class it belongs to!]")
    # plt.savefig("output.png")
    plt.show()