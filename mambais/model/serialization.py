"""Spatial serialization used by the hierarchical voxel encoder."""

import torch


def _hilbert_xy(x: torch.Tensor, y: torch.Tensor, bits: int) -> torch.Tensor:
    """Vectorized 2-D Hilbert distance (integer coordinates)."""
    x, y = x.long().clone(), y.long().clone()
    distance = torch.zeros_like(x)
    scale = 1 << (bits - 1)
    while scale:
        rx = (x & scale) != 0
        ry = (y & scale) != 0
        distance += scale * scale * ((3 * rx.long()) ^ ry.long())
        flip = ~ry
        swap = flip & rx
        maxv = scale - 1
        x = torch.where(swap, maxv - x, x)
        y = torch.where(swap, maxv - y, y)
        old_x = x
        x = torch.where(flip, y, x)
        y = torch.where(flip, old_x, y)
        scale >>= 1
    return distance


def hierarchical_hilbert(coords: torch.Tensor, transpose: int = 0):
    """Return permutation and inverse for z-sliced planar Hilbert traversal.

    ``transpose`` cyclically permutes XYZ, implementing the paper's implicit
    scan-direction change without materializing parallel scan branches.
    """
    coords = coords.long()
    if coords.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=coords.device)
        return empty, empty
    dims = torch.roll(torch.arange(3, device=coords.device), shifts=-transpose)
    c = coords[:, dims]
    c = c - c.min(0).values
    max_xy = int(c[:, :2].max().item()) if len(c) else 0
    bits = max(1, max_xy.bit_length())
    plane_capacity = 1 << (2 * bits)
    key = c[:, 2] * plane_capacity + _hilbert_xy(c[:, 0], c[:, 1], bits)
    order = torch.argsort(key, dim=0)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    return order, inverse
