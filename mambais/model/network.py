"""Scene-wise instance segmentation with a hierarchical voxel backbone."""

from torch import nn
from .voxel import voxelize
from .backbone import MambaBackbone
from .decoder import InstanceQueryAggregator


class MambaIS(nn.Module):
    def __init__(
        self,
        num_classes=18,
        voxel_size=0.02,
        widths=(64, 96, 128, 192),
        depths=(2, 2, 3, 3),
        decoder_width=256,
        miqa_layers=4,
        train_queries=400,
        eval_queries=200,
        state_size=16,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.backbone = MambaBackbone(6, widths, depths, state_size)
        self.decoder = InstanceQueryAggregator(
            list(widths),
            decoder_width,
            miqa_layers,
            train_queries,
            eval_queries,
            num_classes,
            state_size=state_size,
        )

    def forward(self, batch):
        outputs = []
        for points, colors in zip(batch["points"], batch["colors"]):
            vox = voxelize(points, colors, self.voxel_size)
            features, pyramid = self.backbone(
                vox.coords, vox.representatives, vox.features
            )
            out = self.decoder(
                features, pyramid, vox.representatives, vox.coords, points
            )
            out["point_to_voxel"] = vox.point_to_voxel
            out["voxel_positions"] = vox.representatives
            outputs.append(out)
        return outputs
