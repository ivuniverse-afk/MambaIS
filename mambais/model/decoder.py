"""Mamba-based Instance Query Aggregator (MIQA)."""

import math
import torch
from torch import nn
from .backbone import ImplicitMultiScan
from .serialization import hierarchical_hilbert
from .ssm import SequenceOperator


def farthest_point_sample(points, count):
    if count < 1 or not len(points):
        raise ValueError("FPS requires nonempty points and a positive query count")
    count = min(count, len(points))
    chosen = torch.empty(count, dtype=torch.long, device=points.device)
    distance = torch.full((len(points),), float("inf"), device=points.device)
    farthest = torch.tensor(0, device=points.device)
    for i in range(count):
        chosen[i] = farthest
        distance = torch.minimum(distance, ((points - points[farthest]) ** 2).sum(-1))
        distance[chosen[: i + 1]] = -1
        farthest = distance.argmax()
    return chosen


class FourierPosition(nn.Module):
    def __init__(self, bands, width):
        super().__init__()
        self.bands = bands
        self.project = nn.Sequential(
            nn.Linear(6 * bands, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, xyz):
        scale = (2.0 ** torch.arange(self.bands, device=xyz.device)) * math.pi
        phase = xyz[..., None] * scale
        return self.project(torch.cat((phase.sin(), phase.cos()), -1).flatten(-2))


class MIQALayer(nn.Module):
    def __init__(self, width, index, state_size=16):
        super().__init__()
        self.query_ssm = SequenceOperator(width, state_size)
        self.voxel_ssm = SequenceOperator(width, state_size)
        self.gate = nn.Linear(width, width)
        self.ims = ImplicitMultiScan(width, index, state_size)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, query, query_xyz, voxels, voxel_grid):
        q_order, q_inv = hierarchical_hilbert((query_xyz * 1000).long(), 0)
        v_order, _ = hierarchical_hilbert(voxel_grid, 0)
        q = self.query_ssm(query[q_order].unsqueeze(0)).squeeze(0)[q_inv]
        v = self.voxel_ssm(voxels[v_order].flip(0).unsqueeze(0)).squeeze(0).flip(0)
        # Broadcast the pooled backward context over the forward query sequence.
        context = v.mean(0, keepdim=True)
        fused = torch.sigmoid(self.gate(context)) * (q * context)
        query = query + fused
        query = query + self.ims(self.norm(query), (query_xyz * 1000).long())
        return query + self.ffn(self.norm(query))


class InstanceQueryAggregator(nn.Module):
    def __init__(
        self,
        voxel_dims,
        width=256,
        layers=4,
        train_queries=400,
        eval_queries=200,
        classes=18,
        fourier_bands=10,
        state_size=16,
    ):
        super().__init__()
        self.train_queries, self.eval_queries = train_queries, eval_queries
        if layers < 1 or not voxel_dims:
            raise ValueError("MIQA requires at least one layer and one feature scale")
        self.position = FourierPosition(fourier_bands, width)
        self.voxel_projections = nn.ModuleList(
            [nn.Linear(voxel_dims[i % len(voxel_dims)], width) for i in range(layers)]
        )
        self.layers = nn.ModuleList(
            [MIQALayer(width, i, state_size) for i in range(layers)]
        )
        self.mask_features = nn.Linear(voxel_dims[0], width)
        self.mask_query = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, classes + 1)
        )

    def forward(self, finest, multi_scale, positions, coords, points=None):
        nquery = self.train_queries if self.training else self.eval_queries
        sample_positions = positions if points is None else points
        ids = farthest_point_sample(sample_positions, nquery)
        query_xyz = sample_positions[ids]
        query = self.position(query_xyz)
        auxiliary = []
        sources = [multi_scale[i % len(multi_scale)] for i in range(len(self.layers))]
        for i, layer in enumerate(self.layers):
            source = self.voxel_projections[i](sources[i])
            # All decoded resolutions are aligned to finest by the backbone.
            query = layer(query, query_xyz, source, coords)
            auxiliary.append(self._heads(query, finest))
        labels, masks = auxiliary[-1]
        return {
            "pred_logits": labels,
            "pred_masks": masks,
            "query_positions": query_xyz,
            "aux_outputs": [
                {"pred_logits": a, "pred_masks": b} for a, b in auxiliary[:-1]
            ],
        }

    def _heads(self, query, voxels):
        return (
            self.classifier(query),
            self.mask_query(query) @ self.mask_features(voxels).T,
        )
