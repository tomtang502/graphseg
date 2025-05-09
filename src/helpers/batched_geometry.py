import torch

def rotation_angle_diff_batched(R1: torch.Tensor,
                         R2: torch.Tensor,
                         eps: float = 1e-7) -> torch.Tensor:
    """
    Smallest angle (in radians) that rotates R1 into R2.

    Parameters
    ----------
    R1, R2 : (..., 3, 3) torch.Tensor (orthogonal, det +1)
        Orthogonal rotation matrices.  Leading batch dims are allowed.
    eps : float
        Safety margin for clamping, keeps acos argument strictly inside (-1, 1).
    -------
    returns:
    theta : (...) torch.Tensor
        Angular distance ∈ [0, π] with the same leading shape as the inputs.
        Differentiable w.r.t. both R1 and R2.
    """
    # relative rotation
    R_rel = R1.transpose(-2, -1) @ R2            # (..., 3, 3)

    # trace-based angle   θ = acos( (tr(R_rel) – 1)/2 )
    trace = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    trace_clamped = torch.clamp(trace, -1.0 + eps, 1.0 - eps)

    return torch.acos(trace_clamped)

def compute_cost_tr_multi_scale(
    scales: torch.Tensor,          # (S,)  – 200 000 values, e.g. torch.linspace(0.01,10,200000)
    A: torch.Tensor,               # (N,4,4)
    B: torch.Tensor,               # (N,4,4)
    Rx: torch.Tensor,               # (3,3)
    device = 'cuda'
):
    """
    Vectorised AX-=-XB translation-scale cost for *all* scales in `scales`.
    Returns:
        J  : (S,)   mean-squared residual for each scale
        tx : (S,3)  optimal translation for each scale
    """
    A = A.to(device)
    B = B.to(device)
    Rx = Rx.to(device)
    device, dtype = A.device, A.dtype
    scales = scales.to(device=device, dtype=dtype)          # (S,)

    # -------------------------------------------------------------------------
    # 1.   Pre-compute building blocks that are independent of 'scale'
    # -------------------------------------------------------------------------
    Ra, ta = A[:, :3, :3], A[:, :3, 3]                      # (N,3,3), (N,3)
    tb     = B[:, :3, 3]                                    # (N,3)

    I3 = torch.eye(3, device=device, dtype=dtype)
    M  = I3.unsqueeze(0) - Ra                               # (N,3,3)   ≡ (I − Ra)

    rtb  = (Rx @ tb.unsqueeze(-1)).squeeze(-1)              # (N,3)     ≡ Rx · tb

    # Normal–equation pieces (3×3 and 3-vectors)
    ATA   = (M.transpose(-2, -1) @ M).sum(0)                # (3,3)
    inv_ATA = torch.linalg.inv(ATA)                         # (3,3)

    ATb0 = (M.transpose(-2, -1) @ ta.unsqueeze(-1)).sum(0).squeeze(-1)   # (3,)
    ATbr = (M.transpose(-2, -1) @ rtb.unsqueeze(-1)).sum(0).squeeze(-1)  # (3,)

    # -------------------------------------------------------------------------
    # 2.   Optimal translation for every scale (broadcasted mat-vec multiply)
    # -------------------------------------------------------------------------
    # rhs(s) = ATb0 − s · ATbr
    rhs = ATb0.unsqueeze(0) - scales.unsqueeze(1) * ATbr.unsqueeze(0)     # (S,3)
    tx  = (inv_ATA @ rhs.unsqueeze(-1)).squeeze(-1)                       # (S,3)

    # -------------------------------------------------------------------------
    # 3.   Loss for every scale  --  J(s) = a0 + 2 s a1 + s² a2   (quadratic)
    # -------------------------------------------------------------------------
    # Pre-compute coefficients a0, a1, a2 (all scalars)
    tx0 = inv_ATA @ ATb0                                                 # (3,)
    txr = inv_ATA @ ATbr                                                 # (3,)

    c = (M @ tx0) - ta                                                   # (N,3)
    d = -(M @ txr) + rtb                                                 # (N,3)

    a0 = (c.norm(dim=1) ** 2).mean()
    a1 = (c * d).sum(dim=1).mean()
    a2 = (d.norm(dim=1) ** 2).mean()

    J = a0 + 2 * scales * a1 + (scales ** 2) * a2                        # (S,)

    return J, tx