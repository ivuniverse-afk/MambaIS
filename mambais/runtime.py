"""Shared dataset, model, and reproducibility setup for command-line tools."""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SceneDataset, collate_scenes
from .model.network import MambaIS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(cfg, training):
    data = cfg["data"]
    split = data["train_split"] if training else data["val_split"]
    ds = SceneDataset(
        data["root"],
        split,
        data["label_file"],
        data["valid_class_ids"],
        training,
        data["max_points"],
        data["voxel_size"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"] if training else 1,
        shuffle=training,
        num_workers=cfg["train"]["workers"],
        collate_fn=collate_scenes,
        pin_memory=True,
    )
    model = MambaIS(
        num_classes=len(data["class_names"]),
        voxel_size=data["voxel_size"],
        **cfg["model"]
    )
    return model, loader


def to_device(batch, device):
    return {
        k: [x.to(device, non_blocking=True) for x in v] if k != "scene_id" else v
        for k, v in batch.items()
    }
