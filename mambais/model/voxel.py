"""Centered voxelization and hierarchical representative construction."""

from dataclasses import dataclass
import torch


@dataclass
class VoxelLevel:
    coords: torch.Tensor
    centers: torch.Tensor
    features: torch.Tensor
    point_to_voxel: torch.Tensor


def _mean(values, index, count):
    out = values.new_zeros((count, values.shape[-1]))
    out.index_add_(0, index, values)
    sizes = torch.bincount(index, minlength=count).clamp_min(1).to(values.dtype)
    return out / sizes[:, None]


def voxelize(points, colors, voxel_size: float) -> VoxelLevel:
    origin = points.min(0).values
    grid = torch.floor((points - origin) / voxel_size).long()
    coords, inverse = torch.unique(grid, dim=0, sorted=True, return_inverse=True)
    n = len(coords)
    xyz = _mean(points, inverse, n)
    rgb = _mean(colors.float(), inverse, n)
    centers = origin + (coords.float() + 0.5) * voxel_size
    # Centered voxel offsets retain geometry lost by quantization (Eq. 5-7).
    features = torch.cat((rgb, xyz - centers), dim=-1)
    return VoxelLevel(coords, centers, features, inverse)


def coarsen(coords, positions, features):
    parent_grid = torch.div(coords, 2, rounding_mode="floor")
    parent, inverse = torch.unique(parent_grid, dim=0, sorted=True, return_inverse=True)
    n = len(parent)
    center = _mean(positions, inverse, n)
    feat = _mean(features, inverse, n)
    # Pick the child representative nearest to the child-set centroid.
    dist = ((positions - center[inverse]) ** 2).sum(-1)
    representatives = []
    for i in range(n):
        child = torch.where(inverse == i)[0]
        representatives.append(child[dist[child].argmin()])
    rep = torch.stack(representatives) if representatives else coords.new_empty(0)
    return parent, positions[rep], feat, inverse

