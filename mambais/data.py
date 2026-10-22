"""Dataset readers for preprocessed point-cloud scenes."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    """Read one directory of NumPy arrays per scene.

    Required: coord.npy, color.npy, instance.npy and segment*.npy. Labels are
    remapped to contiguous training IDs by ``valid_class_ids``.
    """

    def __init__(
        self,
        root,
        split,
        label_file="segment20.npy",
        valid_class_ids=(),
        training=False,
        max_points=250000,
        voxel_size=0.02,
    ):
        self.root = Path(root)
        self.scenes = sorted((self.root / split).glob("*"))
        if not self.scenes:
            raise FileNotFoundError(f"No scenes found in {self.root / split}")
        self.label_file = label_file
        self.mapping = {int(v): i for i, v in enumerate(valid_class_ids)}
        self.training, self.max_points, self.voxel_size = (
            training,
            max_points,
            voxel_size,
        )

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        path = self.scenes[index]
        xyz = np.load(path / "coord.npy").astype("float32")
        rgb = np.load(path / "color.npy").astype("float32") / 127.5 - 1.0
        semantic = np.load(path / self.label_file).astype("int64")
        instance = np.load(path / "instance.npy").astype("int64")
        if semantic.ndim > 1:
            semantic = semantic[:, 0]
        if instance.ndim > 1:
            instance = instance[:, 0]
        if self.mapping:
            semantic = np.array(
                [self.mapping.get(int(x), -1) for x in semantic], dtype="int64"
            )
        if self.training:
            if np.random.rand() < 0.5:
                xyz[:, 0] *= -1
            angle = np.random.uniform(-np.pi, np.pi)
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            )
            xyz = xyz @ rot.T
            rgb = np.clip(
                rgb * np.random.uniform(0.8, 1.2)
                + np.random.normal(0, 0.02, rgb.shape),
                -1,
                1,
            )
            if len(xyz) > self.max_points:
                # Keep a spatially contiguous crop around a random scene point.
                center = xyz[np.random.randint(len(xyz))]
                distance = np.square(xyz - center).sum(axis=1)
                keep = np.argpartition(distance, self.max_points - 1)[: self.max_points]
                xyz, rgb, semantic, instance = (
                    xyz[keep],
                    rgb[keep],
                    semantic[keep],
                    instance[keep],
                )
        return {
            "points": torch.from_numpy(xyz.copy()),
            "colors": torch.from_numpy(rgb.astype("float32")),
            "semantic": torch.from_numpy(semantic),
            "instance": torch.from_numpy(instance),
            "scene_id": path.name,
        }


def collate_scenes(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}
