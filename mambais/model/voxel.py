"""Centered voxelization and hierarchical representative construction."""

from dataclasses import dataclass
import torch


@dataclass
class VoxelLevel:
    coords: torch.Tensor
    centers: torch.Tensor
    features: torch.Tensor
    point_to_voxel: torch.Tensor
    representatives: torch.Tensor


def _mean(values, index, count):
    out = values.new_zeros((count, values.shape[-1]))
    out.index_add_(0, index, values)
    sizes = torch.bincount(index, minlength=count).clamp_min(1).to(values.dtype)
    return out / sizes[:, None]


def voxelize(points, colors, voxel_size: float) -> VoxelLevel:
    if voxel_size <= 0 or points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(
            "Expected a nonempty N x 3 point cloud and positive voxel size"
        )
    if colors.shape != points.shape:
        raise ValueError("Colors must have the same N x 3 shape as points")
    origin = points.min(0).values
    grid = torch.floor((points - origin) / voxel_size).long()
    coords, inverse = torch.unique(grid, dim=0, sorted=True, return_inverse=True)
    n = len(coords)
    xyz = _mean(points, inverse, n)
    rgb = _mean(colors.float(), inverse, n)
    centers = origin + (coords.float() + 0.5) * voxel_size
    # Use mean point positions as initial representatives and retain RGB.
    features = torch.cat((rgb, xyz - centers), dim=-1)
    return VoxelLevel(coords, centers, features, inverse, xyz)


def coarsen(coords, positions, features):
    parent_grid = torch.div(coords, 2, rounding_mode="floor")
    parent, inverse = torch.unique(parent_grid, dim=0, sorted=True, return_inverse=True)
    n = len(parent)
    center = _mean(positions, inverse, n)
    feat = _mean(features, inverse, n)
    # Pick the child representative nearest to the child-set centroid.
    dist = ((positions - center[inverse]) ** 2).sum(-1)
    minimum = dist.new_full((n,), float("inf"))
    minimum.scatter_reduce_(0, inverse, dist, reduce="amin", include_self=True)
    indices = torch.arange(len(coords), device=coords.device)
    candidates = torch.where(dist == minimum[inverse], indices, len(coords))
    rep = coords.new_full((n,), len(coords))
    rep.scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
    return parent, positions[rep], feat, inverse
