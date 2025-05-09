import torch
import numpy as np

def colmap_pose2transmat(col_pose_mat):
    return np.vstack((col_pose_mat, np.array([[0,0,0,1]])))


def rotation_angle_diff(R1: np.ndarray, R2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Smallest angle [rad] that rotates R1 into R2.
    """
    R_rel = R1.T @ R2
    # Clamp trace to the valid range to avoid numerical noise
    trace = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    return np.arccos(trace)           # ∈ [0, π]

# ---------- helpers ----------
def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices (…,3,3) to unit quaternions (…,4)
    using a numerically stable branch-free algorithm.
    """
    # trace method with safeguards
    B = R.shape[:-2]
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(*B, 4, device=R.device, dtype=R.dtype)

    cond = t > R[..., 2, 2]
    # branch 1 (trace positive)
    q1 = torch.stack([
        t + 1.0,
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)

    # branch 2 (trace negative, find major diagonal)
    i = torch.argmax(torch.stack([R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]], -1), -1)
    def branch(i0, j0, k0):
        q = torch.stack([
            R[..., j0, k0] - R[..., k0, j0],
            1.0 + R[..., i0, i0] - R[..., j0, j0] - R[..., k0, k0],
            R[..., i0, j0] + R[..., j0, i0],
            R[..., k0, i0] + R[..., i0, k0]
        ], -1)
        # rotate so first component is scalar
        return q[..., [1, 2, 3, 0]]

    q2 = torch.where(
        i.unsqueeze(-1) == 0, branch(0, 1, 2),
        torch.where(i.unsqueeze(-1) == 1, branch(1, 2, 0), branch(2, 0, 1))
    )

    q = torch.where(cond.unsqueeze(-1), q1, q2)
    q = q / (2.0 * torch.sqrt(torch.clamp(q[..., 0:1], min=1e-12)))
    return q

def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions (…,4) to rotation matrices (…,3,3).
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy)], -1),
        torch.stack([2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx)], -1),
        torch.stack([2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz], -1)
    ], -2)
    return R

# ---------- rotation average ----------
def average_rotations(R: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Robust Markley–Davenport quaternion mean of rotations.

    Args:
        R: (N,3,3) tensor of rotation matrices
        weights: optional (N,) positive weights (default uniform)

    Returns:
        (3,3) rotation matrix of the average
    """
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError("R must be (N,3,3) rotation matrices")

    q = _mat_to_quat(R)          # (N,4)
    # resolve antipodal ambiguity so all dot with first > 0
    signs = torch.sign(torch.einsum('...i,i->...', q, q[0]) + 1e-12).unsqueeze(-1)
    q = q * signs

    if weights is None:
        weights = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
    w = weights / weights.sum()

    # Davenport / Markley: build 4x4 symmetric matrix K
    Q = q.unsqueeze(-1)          # (N,4,1)
    K = torch.sum(w.view(-1, 1, 1) * (Q @ Q.transpose(-1, -2)), dim=0)

    # eigenvector corresponding to largest eigenvalue
    eigvals, eigvecs = torch.linalg.eigh(K)
    q_mean = eigvecs[:, eigvals.argmax()].real
    R_mean = _quat_to_mat(q_mean)

    return R_mean