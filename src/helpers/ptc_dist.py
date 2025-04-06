
import numpy as np
import torch
import pynvml
def one_way_nn_distances(A, B):
    """
    Compute for each point in A the squared distance to its nearest neighbor in B.
    Returns a tensor of shape [N], where N = len(A).
    
    A: Tensor of shape [N, D] (e.g., D=3 for 3D points)
    B: Tensor of shape [M, D]
    
    Both A and B should be on the same device (GPU or CPU).
    """
    # A: [N, D]
    # B: [M, D]
    # Expand dimensions to allow broadcasting:
    # A_expanded: [N, 1, D]
    # B_expanded: [1, M, D]
    A_expanded = A.unsqueeze(1)  # [N, 1, D]
    B_expanded = B.unsqueeze(0)  # [1, M, D]

    # Compute pairwise squared distances:
    # result shape: [N, M]
    distances_sq = torch.sum((A_expanded - B_expanded)**2, dim=-1)

    # Take the min over M for each of the N rows => nearest neighbor distances:
    # nn_distances_sq shape: [N]
    nn_distances_sq, _ = torch.min(distances_sq, dim=1)

    return nn_distances_sq

def one_way_nn_distances_chunked(A, B, chunk_size=10000):
    """
    Computes the one-way nearest neighbor distances from A to B (squared distances).
    A is split into chunks to reduce VRAM usage.

    Args:
        A: (N, D) tensor
        B: (M, D) tensor
        chunk_size: number of points from A to process per chunk

    Returns:
        nn_distances_sq: (N,) tensor of squared distances from each point in A
                         to its nearest neighbor in B.
    """
    device = A.device
    N = A.shape[0]
    
    # We'll accumulate chunks of results in a list, then concatenate
    result_list = []

    # For each chunk of A
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        A_chunk = A[start_idx:end_idx]  # shape: [chunk_size, D] (or smaller for last chunk)

        # Expand dimensions for broadcasting
        # A_expanded: [chunk_size, 1, D]
        # B_expanded: [1, M, D]
        A_expanded = A_chunk.unsqueeze(1)  # [chunk_size, 1, D]
        B_expanded = B.unsqueeze(0)        # [1, M, D]

        # Pairwise squared distances: [chunk_size, M]
        distances_sq = torch.sum((A_expanded - B_expanded) ** 2, dim=-1)

        # Nearest neighbor for each of the 'chunk_size' points in A_chunk
        nn_distances_sq, _ = torch.min(distances_sq, dim=1)  # shape: [chunk_size]
        
        result_list.append(nn_distances_sq)
    
    # Concatenate all chunk results back into a single tensor of shape [N]
    return torch.cat(result_list, dim=0)

if __name__ == "__main__":
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
            
    xyz_u = torch.tensor(np.random.rand(18000, 3)).to('cuda')
    xyz_v = torch.tensor(np.random.rand(18000, 3)).to('cuda')
    
    
    one_way_nn_distances(xyz_u, xyz_v)
    print_gpu_usage()   
    