"""Sparse Convolutional Embedding and MambaIS U-shaped backbone."""

import torch
from torch import nn
import torch.nn.functional as F
from .serialization import hierarchical_hilbert
from .ssm import SelectiveMixer
from .voxel import coarsen


class SparseNeighborhoodEmbedding(nn.Module):
    """Framework-free 3x3x3 sparse convolution over occupied voxels."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.center = nn.Linear(in_dim, out_dim, bias=False)
        self.neighbor = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, coords, features):
        aggregate = torch.zeros_like(features)
        degree = features.new_zeros(len(features), 1)
        shifted = coords - coords.min(0).values + 1
        extent = shifted.max(0).values + 3
        stride_yz, stride_z = extent[1] * extent[2], extent[2]
        key = shifted[:, 0] * stride_yz + shifted[:, 1] * stride_z + shifted[:, 2]
        sorted_key, sorted_index = key.sort()
        for offset in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            neighbor_key = key + offset[0]*stride_yz + offset[1]*stride_z + offset[2]
            location = torch.searchsorted(sorted_key, neighbor_key)
            safe = location.clamp_max(max(0, len(key)-1))
            valid = (location < len(key)) & (sorted_key[safe] == neighbor_key)
            if valid.any():
                ii = torch.where(valid)[0]; jj = sorted_index[safe[valid]]
                aggregate[ii] += features[jj]; degree[ii] += 1
        aggregate = aggregate / degree.clamp_min(1)
        return F.relu(self.norm(self.center(features) + self.neighbor(aggregate)))


class ImplicitMultiScan(nn.Module):
    def __init__(self, width, level, state_size=16):
        super().__init__()
        self.level = level
        self.input = nn.Linear(width, width * 2)
        self.ssm = SelectiveMixer(width, state_size)
        self.output = nn.Linear(width, width)

    def forward(self, x, coords):
        order, inverse = hierarchical_hilbert(coords, self.level % 3)
        a, gate = self.input(x[order]).chunk(2, -1)
        mixed = self.ssm(a.unsqueeze(0)).squeeze(0)
        out = self.output(mixed * F.silu(gate))[inverse]
        return out


class Mamba3DBlock(nn.Module):
    def __init__(self, width, level, state_size=16, expansion=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ims = ImplicitMultiScan(width, level, state_size)
        self.mlp = nn.Sequential(nn.Linear(width, width * expansion), nn.GELU(),
                                 nn.Linear(width * expansion, width))

    def forward(self, x, coords):
        x = x + self.ims(self.norm1(x), coords)
        return x + self.mlp(self.norm2(x))


class MambaBackbone(nn.Module):
    def __init__(self, input_dim=6, widths=(64, 96, 128, 192), depths=(2,2,3,3), state_size=16):
        super().__init__()
        self.widths = widths
        self.stem = SparseNeighborhoodEmbedding(input_dim, widths[0])
        self.encoders = nn.ModuleList()
        self.down = nn.ModuleList()
        for level, (width, depth) in enumerate(zip(widths, depths)):
            self.encoders.append(nn.ModuleList([Mamba3DBlock(width, level+i, state_size) for i in range(depth)]))
            if level + 1 < len(widths): self.down.append(nn.Linear(width, widths[level+1]))
        self.up = nn.ModuleList([nn.Linear(widths[i+1], widths[i]) for i in range(len(widths)-1)])
        self.fuse = nn.ModuleList([nn.Linear(widths[i]*2, widths[i]) for i in range(len(widths)-1)])

    def forward(self, coords, positions, raw_features):
        levels, skips = [], []
        x = self.stem(coords, raw_features)
        c, p = coords, positions
        for level, blocks in enumerate(self.encoders):
            for block in blocks: x = block(x, c)
            levels.append((c, p, x))
            if level < len(self.down):
                pc, pp, px, child_to_parent = coarsen(c, p, x)
                skips.append(child_to_parent)
                c, p, x = pc, pp, self.down[level](px)
        multi = [levels[-1][2]]
        for level in reversed(range(len(self.up))):
            child_map = skips[level]
            up = self.up[level](x)[child_map]
            x = self.fuse[level](torch.cat((up, levels[level][2]), -1))
            c, p = levels[level][0], levels[level][1]
            multi.append(x)
        # Expose every semantic scale at the finest voxel locations for MIQA.
        aligned = []
        for level, (_, _, level_features) in enumerate(levels):
            expanded = level_features
            for child_level in reversed(range(level)):
                expanded = expanded[skips[child_level]]
            aligned.append(expanded)
        return x, aligned
