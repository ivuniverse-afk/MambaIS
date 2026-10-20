# MambaIS: Bridging Mamba and 3D Point Clouds: Efficient Instance Segmentation via Implicit State Space Models

Implementation of **MambaIS**. We propose MambaIS, a novel and efficient Mamba-centered framework for 3D instance segmentation that integrates sparse local geometric encoding with selective state-space modeling for global context learning.}
To bridge the structural gap between 1D SSMs and 3D geometry, we introduce Sparse Convolutional Embedding (SCE) equipped with a hierarchical 3D Hilbert scanning strategy, mapping unstructured points into locally coherent sequences. To efficiently capture global spatial contexts, we design an Implicit Multi-directional Scan (IMS) module that leverages coordinate transposition, eliminating the substantial memory overhead of standard multi-branch scanning. Furthermore, we develop the Mamba-based Instance Query Aggregator (MIQA), an end-to-end decoding mechanism that iteratively refines instance queries and voxel features via bidirectional SSMs, thereby bypassing heuristic clustering intermediate steps. 

## Installation

We recommend using Ubuntu 22.04, Python 3.10, CUDA 12.1, and PyTorch 2.2 to create a clean environment and install a fully pinned set of dependencies.

```bash
conda create -n mambais python=3.10 -y
conda activate mambais
git clone https://github.com/ivuniverse-afk/MambaIS.git
cd MambaIS
pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
```

`requirements.txt` includes the CUDA 12.1 PyTorch wheels, NumPy/SciPy, data readers, TensorBoard, `causal-conv1d`, and `mamba-ssm`. The last two packages compile CUDA extensions and therefore require a working CUDA toolkit (`nvcc --version`) compatible with the installed PyTorch wheel. Set `MAX_JOBS` if compilation exhausts host memory:

```bash
MAX_JOBS=4 pip install -r requirements.txt --no-build-isolation
```

Confirm the installation:

```bash
python -c "import torch, mamba_ssm, mambais; print(torch.__version__, torch.cuda.is_available())"
```

## Datasets

We do not provide dataset downloads, nor do we redistribute their terms-protected files. Please request and download each dataset from its official page.

- [ScanNet v2](http://www.scan-net.org/): accept its terms and use the official download script. MambaIS uses the 20-class instance segmentation labels.
- [S3DIS](https://cvg-data.inf.ethz.ch/s3dis/): request access and download the aligned version. Area 5 is the default validation split; the other five areas form `train_area5`.
- [STPLS3D](https://github.com/meidachen/STPLS3D): obtain the official train/validation point clouds and annotations.

Preprocess each scene into a folder containing point-aligned NumPy arrays:

```text
data/
├── scannet/
│   ├── train/scene0000_00/{coord,color,segment20,instance}.npy
│   └── val/scene0011_00/{coord,color,segment20,instance}.npy
├── s3dis/
│   ├── train_area5/<room>/{coord,color,segment,instance}.npy
│   └── Area_5/<room>/{coord,color,segment,instance}.npy
└── stpls3d/
    ├── train/<scene>/{coord,color,segment,instance}.npy
    └── val/<scene>/{coord,color,segment,instance}.npy
```

Each `coord.npy` is float32 `[N,3]` XYZ, `color.npy` is uint8 or float32 `[N,3]` RGB in `[0,255]`, and the semantic and instance arrays are integer `[N]`. Use `-1` for ignored instance labels. ScanNet semantic arrays retain the official NYU40 IDs; the configuration maps the 18 evaluated object classes to contiguous IDs. S3DIS and STPLS3D arrays use the class IDs listed in their YAML files.

If processed data lives elsewhere, use symbolic links:

```bash
mkdir -p data
ln -s /absolute/path/to/processed_scannet data/scannet
ln -s /absolute/path/to/processed_s3dis data/s3dis
ln -s /absolute/path/to/processed_stpls3d data/stpls3d
```

## Training

Configs: a 2 cm voxel for ScanNet v2 and S3DIS, 3 cm for STPLS3D, at most 250,000 input points, 400 training queries, four MIQA layers, 200 epochs, AdamW with learning rate `1e-3` and weight decay `1e-4`, and the weighted classification + BCE + Dice objective. The published batch size is 16 on one RTX 4090; reduce `train.batch_size` when GPU memory is smaller.

```bash
# ScanNet v2
bash scripts/train.sh configs/scannetv2.yaml outputs/scannetv2

# S3DIS Area 5 protocol
bash scripts/train.sh configs/s3dis.yaml outputs/s3dis_area5

# STPLS3D
bash scripts/train.sh configs/stpls3d.yaml outputs/stpls3d
```

The latest checkpoint is written to `<output>/last.pt`; periodic checkpoints follow `epoch_NNN.pt`. Resume an interrupted run directly:

```bash
python tools/train.py \
  --config configs/scannetv2.yaml \
  --output outputs/scannetv2 \
  --resume outputs/scannetv2/last.pt
```

For a specific GPU, set `CUDA_VISIBLE_DEVICES` before the command. Override experiment settings by copying a YAML config and editing it; the checkpoint retains the exact resolved configuration.

## Evaluation

Run class-aware instance evaluation on the validation split:

```bash
bash scripts/eval.sh configs/scannetv2.yaml outputs/scannetv2/last.pt
bash scripts/eval.sh configs/s3dis.yaml outputs/s3dis_area5/last.pt
bash scripts/eval.sh configs/stpls3d.yaml outputs/stpls3d/last.pt
```

Prediction confidence and binary-mask thresholds can be controlled explicitly:

```bash
python tools/evaluate.py \
  --config configs/scannetv2.yaml \
  --checkpoint outputs/scannetv2/last.pt \
  --score-threshold 0.1 \
  --mask-threshold 0.5
```

For S3DIS 6-fold cross-validation, create six configs by assigning each `Area_1` through `Area_6` to `val_split`, train on a corresponding directory containing the remaining areas, then average the six fold metrics.

## Acknowledgement

We thank [Mask3D](https://github.com/JonasSchult/Mask3D) and [Mamba](https://github.com/state-spaces/mamba) for their excellent work.
