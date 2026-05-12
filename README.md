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
