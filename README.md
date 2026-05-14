<div align="center">

# The DAWN of World-Action Interactive Models

**Hongbo Lu**<sup>1,2,\*</sup>, **Liang Yao**<sup>1,\*</sup>, **Chenghao He**<sup>1,\*</sup>, **Haoyu Wang**<sup>1,\*</sup>, 

**Xiang Gu**<sup>3</sup>, **Xianfei Li**<sup>1,2</sup>, **Wenlong Liao**<sup>1,&dagger;</sup>, **Tao He**<sup>1</sup>, **Pai Peng**<sup>1,&dagger;,&Dagger;</sup>

<sup>1</sup>COWARobot Co. Ltd &nbsp;&nbsp; <sup>2</sup>Shanghai Jiao Tong University &nbsp;&nbsp; <sup>3</sup>Hohai University

\* Equal Contribution &nbsp;&nbsp; &dagger; Corresponding Author &nbsp;&nbsp; &Dagger; Project Lead

</div>

---

# nuScenes Inference

Offline inference/evaluation repo for world-model planners on nuScenes data converted to the NavSim PKL layout.

The entrypoint loads a training YAML, builds the encoder, predictor, and planner, restores a checkpoint, runs the `val_command.py` validation forward path, and writes metrics plus optional trajectory visualizations.

## Setup

```bash
pip install -e .
```

The code expects the converted nuScenes/NavSim PKL paths and model checkpoints referenced by the YAML to exist on the machine.

## Model Weights

Download the required model weights before running inference.

- Pretrained weights (used for `config.meta.pretrain_checkpoint_full`):
  - `vjepa_pretrain.pt`: https://pan.baidu.com/s/1SgmdMop50yd-Vv2nl0k2Mg 提取码: ukx8 
- Inference checkpoint weights (used for `--checkpoint` or `best_open_loop.pt`):
  - `latest.pt`: https://pan.baidu.com/s/1uVHHEGp8qEJR9Gz3XGS05Q 提取码: xhku 

For example:

```bash
mkdir -p checkpoints
wget -O checkpoints/vjepa_pretrain.pt https://example.com/path/to/vjepa_pretrain.pt
wget -O checkpoints/best_open_loop.pt https://example.com/path/to/best_open_loop.pt
```

Then update your YAML or command line `--checkpoint` path accordingly.

## Data Preparation

Convert raw nuScenes data (labels JSON + GT box NPZ + CAN bus) to the NavSim PKL layout used by `NavSimWorldModelDataset`:

```bash
python tools/convert_nuscenes_to_navsim_pkl.py \
  --nuscenes-root /path/nuScenes \
  --output-root /path/nuScenes/navsim_format \
  --split trainval \
  --workers 8
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--nuscenes-root` | `/path/nuScenes` | nuScenes root containing `labels/`, `samples/`, `can_bus/` |
| `--output-root` | `/path/nuScenes/navsim_format_fix_2` | Output directory |
| `--split` | `trainval` | Which split to convert: `train`, `val`, or `trainval` |
| `--workers` | `8` | Number of parallel workers |
| `--dry-run` | off | List scenes without converting |

**Input directory layout** (`--nuscenes-root`):

```
/path/nuScenes/
├── labels/
│   ├── scene-0001.json          # per-scene frame metadata
│   ├── scene-0001/              # per-scene GT box NPZ files
│   │   ├── 0001.npz
│   │   └── ...
│   └── scene-0002.json
├── samples/                     # original camera images (symlinked, not copied)
│   └── CAM_FRONT/...
└── can_bus/
    └── scene-0001_pose.json     # CAN bus ego pose (velocity, acceleration)
```

**Output directory layout** (`--output-root`):

```
/path/nuScenes/navsim_format/
├── train/
│   ├── scene-0001.pkl           # List[Dict], each entry = one keyframe (2 Hz)
│   └── ...
├── val/
│   ├── scene-0003.pkl
│   └── ...
└── sensor_blobs/
    └── scene-0001/
        └── CAM_F0/
            └── n015-2018-...jpg  # symlinks to original images
```

**What gets converted per keyframe:**

- `ego2global_translation` / `ego2global_rotation` — from nuScenes ego pose matrix
- `ego_dynamic_state` (`[vx, vy, ax, ay]` in ego frame) — interpolated from CAN bus pose data
- `driving_command` (one-hot: GO_STRAIGHT / TURN_LEFT / TURN_RIGHT / U_TURN) — inferred from cumulative yaw change over the scene
- `cams` — only CAM_FRONT is retained; images are symlinked, not copied
- `anns` (GT boxes) — NPZ annotations converted to NavSim format; categories mapped (vehicle/pedestrian/bicycle); barriers and traffic cones are filtered out

Scene split follows the standard nuScenes v1.0-trainval partition: 700 train + 150 val scenes.

## Run

```bash
PYTHONPATH="$PWD" /usr/bin/python3 scripts/infer_nuscenes_val.py \
  --config configs/nuscenes_inference_example.yaml \
  --checkpoint /path/to/best_open_loop.pt \
  --output-dir outputs/nuscenes_eval \
  --disable-vis
```

If `--checkpoint` is omitted, the script tries `<config.folder>/best_open_loop.pt`, then `<config.folder>/latest.pt`, then `meta.resume_checkpoint`.

## Data Contract

This repo targets nuScenes converted to the local NavSim-style PKL format used by `NavSimWorldModelDataset`:

- `data.navsim.val_data_path`: directory containing validation `.pkl` scene files
- `data.navsim.val_sensor_blobs_path`: camera image root
- `data.navsim.camera_name`: camera folder name, usually `CAM_F0`

The validation path computes ADE/FDE/minADE@K/minFDE@K, World4Drive L2 horizons, and collision metrics when BEV segmentation is present in the dataloader batch.
