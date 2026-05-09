# Volt with Articulation Heads for Articulate3D Dataset

This document describes the implementation of additional classification heads for Volt to support the Articulate3D dataset, which includes annotations for interactable and movable objects.

## Overview

The implementation adds four auxiliary heads to the Volt backbone:
1. **Movable Part Segmentation** (3 classes): Predicts whether each point is fixed, rotatable, or translatable
2. **Interactable Part Segmentation** (binary): Predicts whether each point is an interactable surface (handle, knob, switch)
3. **Axis Direction Regression** (per-instance): Predicts a unit-vector axis of motion for each articulated instance
4. **Axis Origin Regression** (per-instance): Predicts a point on the motion axis for each articulated instance

Heads 1–2 are point-level; heads 3–4 are instance-level (mean-pooled features). All four are trained jointly with semantic segmentation.

## Dataset Structure

### Expected Directory Layout

```
data/
├── scannetpp/                              # ScanNet++ base data (existing)
│   ├── train/
│   │   ├── [scene_id]/
│   │   │   ├── coord.npy                   # Point coordinates
│   │   │   ├── color.npy                   # Point colors
│   │   │   ├── normal.npy                  # Point normals
│   │   │   ├── segment.npy                 # Semantic labels
│   │   │   └── instance.npy                # Instance labels
│   └── val/
│   └── test/
│
└── articulate3d_labels/                    # Preprocessed articulation labels (NEW)
    ├── [scene_id]_movable_label.npy        # Per-point movable labels {0,1,2}
    ├── [scene_id]_interactable_label.npy   # Per-point interactable labels {0,1}
    ├── [scene_id]_artic_instance_label.npy # Per-point instance IDs (0=bg, N=instance N)
    └── [scene_id]_artic_instances.pkl      # Per-instance axis/origin metadata
```

### Articulation Label Format

**movable_label.npy**: (N,) int64 array
- `0`: Fixed/non-movable
- `1`: Rotatable object
- `2`: Translatable object

**interactable_label.npy**: (N,) int64 array
- `0`: Non-interactable
- `1`: Interactable (handle, knob, etc.)

**artic_instance_label.npy**: (N,) int64 array
- `0`: Background (no articulated instance)
- `k`: Point belongs to the k-th articulated instance (1-indexed)

**artic_instances.pkl**: list of dicts, one per articulated instance
```python
{
    "instance_id": int,           # 1-indexed
    "label": str,                 # part label string
    "motion_type": int,           # 1=rotation, 2=translation
    "axis": np.ndarray,           # (3,) unit vector — axis of motion
    "origin": np.ndarray,         # (3,) point on the motion axis
    "vertex_mask": np.ndarray,    # (V,) bool — mesh vertices in this instance
}
```

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

**Output (4 files per scene):**
- `{scene_id}_movable_label.npy` — per-vertex motion type
- `{scene_id}_interactable_label.npy` — per-vertex interactability
- `{scene_id}_artic_instance_label.npy` — per-vertex instance IDs
- `{scene_id}_artic_instances.pkl` — per-instance axis/origin metadata

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
freeze_backbone = False      # Set to True if joint training hurts base performance
articulation_weight = 0.5    # Loss weight for segmentation heads (movable+interactable)

# Articulation segmentation loss
lambda_dice = 1.0            # Dice loss weight
lambda_ce = 1.0              # Cross-entropy loss weight

# Regression loss (axis + origin)
lambda_axis = 1.0            # Weight for axis direction loss
lambda_origin = 0.5          # Weight for axis origin loss (harder; start lower)

# Training
batch_size = 16              # Batch size per GPU
epoch = 800                  # Total epochs
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
```

### Training Details

**Loss Function:**
```
L_total = L_seg + λ_artic × (L_movable + L_interactable) + L_regression

where:
  L_movable     = λ_ce × BCE + λ_dice × Dice   (per-point, binary: movable vs fixed)
  L_interactable = λ_ce × BCE + λ_dice × Dice  (per-point, binary)
  L_regression  = λ_axis × L_axis + λ_origin × L_origin  (per-instance)

  L_axis   = mean(1 - |cos(pred_axis, gt_axis)|)   ← abs handles ±direction ambiguity
  L_origin = mean(‖(pred_origin − gt_origin) − proj‖)  ← perpendicular-to-axis distance
             where proj = ((pred−gt)·axis) × axis
```

**Key Features:**
- Mixed batch training: Scenes with/without articulation labels in same batch
- Articulation loss only computed on scenes with `has_articulation=True`
- Instance features pooled by mean over `artic_instance_label` voxels (post-GridSample)
- `OriginHead` predicts offsets from instance centroid (more stable than absolute coords)
- `artic_instances` metadata saved/restored around the transform pipeline (not transformed)
- Exponential moving average (EMA) for model weights

## Inference & Submission

### Generate Predictions

```python
from tools.articulate_inference import extract_movable_instances, prepare_articulate_submission

# After loading model and running inference:
xyz = batch['coord']  # (N, 3) point coordinates
movable_logits = output['movable_logits']  # (N, 3)
interactable_logits = output['interactable_logits']  # (N, 1)

# Extract instances
instances, motions, interactable_prob = extract_movable_instances(
    xyz, movable_logits, interactable_logits,
    threshold=0.5, min_points=20
)

# Prepare submission
predictions = {
    'scene_id': {
        'movable_instances': instances,
        'instance_motions': motions,
        'interactable_prob': interactable_prob,
    }
}

submission_paths = prepare_articulate_submission(predictions, output_dir='submissions')
```

### Submission Format

Predictions are saved as pickle files with structure:
```python
{
    'movable_instances': np.ndarray,  # (num_instances, N) boolean masks
    'instance_motion_types': np.ndarray,  # (num_instances,) {1: rotation, 2: translation}
    'interactable_prob': np.ndarray,  # (N,) probabilities in [0, 1]
}
```

## Architecture Details

### Modified Files

1. **pointcept/datasets/scannetpp.py**
   - Added `ScanNetPPArticulateDataset` class
   - Loads articulation labels alongside point cloud data
   - Handles missing labels gracefully with `has_articulation` flag

2. **pointcept/models/volt/volt_articulate.py** (NEW)
   - `pool_instance_features`: Mean-pools voxel features per instance using `artic_instance_label`
   - `AxisHead`: MLP with LayerNorm + GELU; output L2-normalised to unit vector
   - `OriginHead`: MLP conditioned on instance centroid; predicts offset → absolute origin
   - `VoltArticulate`: Volt backbone with four auxiliary heads
   - `ArticulateSegmentor`: Training/inference wrapper; calls `_run_regression` for axis/origin

3. **pointcept/models/losses/articulation.py** (NEW)
   - `ArticulationLoss`: Combined Dice + BCE loss for movable/interactable heads
   - `axis_loss`: Cosine-similarity loss with `abs()` for ±axis ambiguity
   - `origin_loss`: Point-to-line perpendicular distance loss
   - `artic_regression_loss`: Combines axis + origin losses for all valid instances
   - `BinaryCrossEntropyWithDiceLoss`: Utility loss

4. **configs/scannetpp/semseg-volt-articulate.py** (NEW)
   - Training configuration with articulation settings

5. **tools/preprocess_articulate3d.py** (NEW)
   - Preprocessing pipeline for annotations

6. **tools/articulate_inference.py** (NEW)
   - Inference utilities and instance extraction

### Model Outputs

```
                    ┌─────────────────┐
                    │ Volt Backbone   │
                    │ (enc + dec)     │
                    └────────┬────────┘
                             │ point features (M, 128)
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──┐         ┌──────▼──┐        ┌──────▼──┐
   │Seg Head │         │Movable  │        │Interact │
   │(100 cls)│         │Head     │        │Head     │
   │         │         │(3 cls)  │        │(1 cls)  │
   └─────────┘         └─────────┘        └─────────┘
   seg_logits        movable_logits    interactable_logits
   (M, 100)          (M, 3)            (M, 1)

   Instance pooling (via artic_instance_label):
   point features (M, 128) → mean-pool per instance → (N_inst, 128)

          ┌──────────────────┬──────────────────┐
          │                  │                  │
   ┌──────▼──────┐    ┌──────▼──────┐   inst centroids
   │  Axis Head  │    │ Origin Head │   (N_inst, 3)
   │  (MLP+norm) │    │  (MLP+cond) │◄──────────┘
   └─────────────┘    └─────────────┘
   axis_preds          origin_preds
   (N_inst, 3)         (N_inst, 3)
```

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
