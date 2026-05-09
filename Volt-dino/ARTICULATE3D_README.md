# Volt with Articulation Heads for Articulate3D Dataset

This document describes the implementation of additional classification heads for Volt to support the Articulate3D dataset, which includes annotations for interactable and movable objects.

## Overview

The implementation adds three auxiliary heads to the Volt backbone:
1. **Movable Part Segmentation** (3 classes): predicts whether each point is fixed, rotatable, or translatable
2. **Interactable Part Segmentation** (binary): predicts handles, knobs, switches
3. **Axis + Origin Regression** (NEW): per articulated instance, predicts the motion axis direction and a point on that axis line

All heads are trained jointly with the primary semantic segmentation task.

**Regression design.** Heads 1 and 2 operate per-voxel; head 3 operates per-instance. Backbone features are mean-pooled within each ground-truth instance mask and passed through two small MLP heads. This avoids predicting one axis per voxel (redundant and noisier) while keeping training fully differentiable.

## Dataset Structure

### Expected Directory Layout

```
data/
├── scannetpp/                          # ScanNet++ base data (existing)
│   ├── train/
│   │   ├── [scene_id]/
│   │   │   ├── coord.npy               # Point coordinates
│   │   │   ├── color.npy               # Point colors
│   │   │   ├── normal.npy              # Point normals
│   │   │   ├── segment.npy             # Semantic labels
│   │   │   └── instance.npy            # Instance labels
│   └── val/
│   └── test/
│
└── articulate3d_labels/                     # Preprocessed articulation labels
    ├── [scene_id]_movable_label.npy         # Per-vertex movable class
    ├── [scene_id]_interactable_label.npy    # Per-vertex interactable flag
    ├── [scene_id]_instance_artic_label.npy  # Per-vertex instance ID (NEW)
    ├── [scene_id]_axis_label.npy            # Per-vertex unit axis (NEW)
    └── [scene_id]_origin_label.npy          # Per-vertex axis origin (NEW)
```

### Articulation Label Format

**movable_label.npy**: (N,) int64 array
- `0`: Fixed/non-movable
- `1`: Rotatable object
- `2`: Translatable object

**interactable_label.npy**: (N,) int64 array
- `0`: Non-interactable
- `1`: Interactable (handle, knob, etc.)

**instance_artic_label.npy**: (N,) int64 array — (NEW)
- `0`: Background (fixed or unannotated)
- `k > 0`: Vertex belongs to articulated instance k (local to each scene)

**axis_label.npy**: (N, 3) float32 — (NEW)  
Unit axis direction for each vertex's instance. Zero vector for background.

**origin_label.npy**: (N, 3) float32 — (NEW)  
A point on the axis line for each vertex's instance. Zero for background.  
All vertices in the same instance share the same axis and origin values — this redundancy lets GridSample / SphereCrop subsample them without any special handling.

## Setup Instructions

### 1. Prepare ScanNet++ Data

Ensure you have the standard ScanNet++ data structure:
```bash
# Your ScanNet++ data should be in:
/path/to/data/scannetpp/
```

### 2. Preprocess Articulate3D Annotations

Convert the raw Articulate3D annotations (parts.json, artic.json) to per-point labels:

```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt

python tools/preprocess_articulate3d.py \
    --articulate_root /path/to/articulate3d/raw/data \
    --scannetpp_root /path/to/scannetpp/data \
    --output_root data/articulate3d_labels
```

**Input Requirements:**
- `articulate_root`: Must contain subdirectories for each scene with:
  - `parts.json`: Mesh part definitions and triangle indices
  - `artic.json`: Articulation metadata (motion type, etc.)
- `scannetpp_root`: Must contain aligned meshes at `[scene_id]/mesh_aligned_0.05.ply`

**Output:**
- Creates `.npy` files for each scene with per-vertex labels
- Files are named `{scene_id}_movable_label.npy` and `{scene_id}_interactable_label.npy`

### 3. Update Configuration

Edit the training config to point to your data paths:

```bash
# configs/scannetpp/semseg-volt-articulate.py

# Update these paths:
data_root = "data/scannetpp"              # Path to ScanNet++ data
articulation_root = "data/articulate3d_labels"  # Path to preprocessed labels
```

## Training

### Quick Start

```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt

# Single GPU
python tools/train.py configs/scannetpp/semseg-volt-articulate.py

# Multiple GPUs (recommended)
python tools/train.py configs/scannetpp/semseg-volt-articulate.py \
    --num_gpus 4
```

### On HPC

```bash
# Example SLURM job script
#!/bin/bash
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00

cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt
python tools/train.py configs/scannetpp/semseg-volt-articulate.py --num_gpus 4
```

### Configuration Options

Key hyperparameters in `semseg-volt-articulate.py`:

```python
# Model
freeze_backbone    = False  # set True if articulation heads hurt base segmentation
use_regression     = True   # enable AxisHead + OriginHead

# Loss weights (all relative to semantic segmentation loss = 1.0)
articulation_weight = 0.5   # movable + interactable BCE+Dice
regression_weight   = 0.5   # axis angular + origin point-to-line

# Articulation classification loss
lambda_dice = 1.0
lambda_ce   = 1.0

# Regression loss
axis_weight   = 1.0         # weight of axis angular error
origin_weight = 1.0         # weight of origin point-to-line distance

# Training
batch_size = 16
epoch      = 800
optimizer  = dict(type="AdamW", lr=0.001, weight_decay=0.05)
```

### Training Details

**Loss Function:**
```
L_total = L_seg
        + λ_artic  × (L_movable + L_interactable)
        + λ_regres × (L_axis + L_origin)

where:
  L_movable       = λ_ce × BCE + λ_dice × Dice   (per-voxel)
  L_interactable  = λ_ce × BCE + λ_dice × Dice   (per-voxel)

  L_axis   = mean_k [ 1 - |cos(pred_axis_k, gt_axis_k)| ]
             — invariant to ±axis sign

  L_origin = mean_k [ ||delta_k - (delta_k · gt_axis_k) gt_axis_k|| ]
             where delta_k = pred_origin_k - gt_origin_k
             — zero for any point on the axis line (point-to-line distance)
```

**Key Features:**
- Mixed batch training: Scenes with/without articulation labels in same batch
- Articulation loss only computed on scenes with labels (via `has_articulation` flag)
- Gradient checkpointing enabled for efficient memory usage
- Exponential moving average (EMA) for model weights

## Inference & Submission

### Generate Predictions

```python
from tools.articulate_inference import (
    extract_movable_instances,
    predict_instance_axes_origins,
    prepare_articulate_submission,
)

# After loading model and running forward pass:
xyz                 = batch['coord']                # (N, 3)
features            = output['features']            # (N, D) backbone features
movable_logits      = output['movable_logits']      # (N, 3)
interactable_logits = output['interactable_logits'] # (N, 1)

# Step 1: cluster movable points into instances via DBSCAN
instances, motions, interactable_prob = extract_movable_instances(
    xyz, movable_logits, interactable_logits,
    threshold=0.5, min_points=20
)

# Step 2: predict axis + origin for each cluster  (NEW)
pred_axes, pred_origins = predict_instance_axes_origins(
    features,                   # torch tensor on correct device
    instances,                  # (K, N) bool numpy array
    model.model.axis_head,      # AxisHead from VoltArticulate
    model.model.origin_head,
)

# Prepare submission
predictions = {
    'scene_id': {
        'movable_instances': instances,
        'instance_motions':  motions,
        'interactable_prob': interactable_prob,
        'instance_axes':     pred_axes,    # (K, 3) unit axes  (NEW)
        'instance_origins':  pred_origins, # (K, 3) origin points  (NEW)
    }
}

submission_paths = prepare_articulate_submission(predictions, output_dir='submissions')
```

### Submission Format

Predictions are saved as pickle files with structure:
```python
{
    'movable_instances':     np.ndarray,  # (K, N) boolean instance masks
    'instance_motion_types': np.ndarray,  # (K,)   {1: rotation, 2: translation}
    'interactable_prob':     np.ndarray,  # (N,)   probabilities in [0, 1]
    'instance_axes':         np.ndarray,  # (K, 3) unit axis directions  (NEW)
    'instance_origins':      np.ndarray,  # (K, 3) axis origin points    (NEW)
}
```

## Architecture Details

### Modified Files

| File | What it does |
|------|-------------|
| `pointcept/datasets/scannetpp.py` | `ScanNetPPArticulateDataset` loads all 5 label arrays; falls back to zeros when absent |
| `pointcept/models/volt/volt_articulate.py` | `pool_instance_features()`, `AxisHead`, `OriginHead`; `VoltArticulate` with 4 heads; `ArticulateSegmentor` with 3 losses |
| `pointcept/models/losses/articulation.py` | `ArticulationLoss` (BCE+Dice); `ArticulationRegressionLoss` (axis + origin) |
| `configs/scannetpp/semseg-volt-articulate.py` | Adds `regression_criteria`, `regression_weight`, `use_regression`, and new `extra_keys` |
| `tools/preprocess_articulate3d.py` | Preprocessing: now also produces `instance_artic_label`, `axis_label`, `origin_label` |
| `tools/articulate_inference.py` | Adds `predict_instance_axes_origins()` and `evaluate_axis_origin()` |

### Model Outputs

```
                      ┌─────────────────┐
                      │ Volt Backbone   │
                      │ (enc + dec)     │
                      └────────┬────────┘
                               │ features (N, 128)  ← per voxel
          ┌──────────┬─────────┼─────────┬──────────────────────┐
          │          │         │         │ pool per GT instance  │
   ┌──────▼──┐ ┌─────▼──┐ ┌───▼──┐  ┌───▼──────────────────┐   │
   │Seg Head │ │Movable │ │Inter-│  │ pool_instance_feats   │   │
   │(100 cls)│ │Head    │ │actbl │  │ → mean over instance  │   │
   │         │ │(3 cls) │ │(bin) │  │   points → (K, 128)   │   │
   └────┬────┘ └───┬────┘ └──┬───┘  └───┬────────────┬──────┘   │
        │          │         │          │             │           │
   seg_logits mov_logits int_logits ┌───▼───┐    ┌───▼───┐       │
                                    │ Axis  │    │Origin │       │
                                    │ Head  │    │ Head  │       │
                                    └───┬───┘    └───┬───┘       │
                                   pred_axes   pred_origins      │
                                    (K, 3)       (K, 3)          │
```

The regression branch runs **only when `instance_artic_label` is present** in the batch (i.e., during training/eval with GT labels).  During test-time inference, it runs after DBSCAN clustering using `predict_instance_axes_origins()`.

## Performance Considerations

### Memory Usage
- Volta V100 (32GB): Batch size 16 with backbone features works well
- Smaller backbone (embed_dim=256) if needed for smaller GPUs

### Training Time
- ~800 epochs on 4 V100 GPUs: ~7-10 days
- With gradient accumulation, can reduce batch size impact

### Optimization Tips

1. **If joint training hurts semantic segmentation:**
   ```python
   freeze_backbone = True  # Only train articulation heads
   ```

2. **If memory is tight:**
   ```python
   batch_size = 8  # Reduce batch size
   gradient_accumulation_steps = 2  # Accumulate gradients
   ```

3. **If articulation signal is weak:**
   ```python
   articulation_weight = 0.2  # Reduce loss weight
   ```

## Troubleshooting

### Missing articulation labels
- Check that `articulation_root` path is correct
- Verify label files exist: `{scene_id}_movable_label.npy`
- If labels don't exist, `has_articulation` is False and loss is skipped

### CUDA out of memory
- Reduce `batch_size` or `point_max` in GridSample
- Enable gradient checkpointing (if not already)
- Use `empty_cache = True` in config

### Poor articulation performance
- Ensure labels are correctly preprocessed
- Check label distribution (imbalanced classes?)
- Increase `articulation_weight` to emphasize this task
- Verify mesh-to-point correspondence in preprocessing

## References

- Articulate3D Challenge: https://insait-institute.github.io/articulate3d.github.io/challenge.html
- USDNet: https://arxiv.org/abs/2302.00923
- Volt: https://arxiv.org/abs/2404.06242

## Contact

For questions or issues with this implementation, please refer to the original paper and dataset documentation.
