# Data Pipeline: Articulate3D with Volt

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR RAW DATA (from Articulate3D)                     │
│                                                                               │
│  articulate3d/[scene_id]/                                                    │
│  ├── parts.json              ← Part definitions with triangle indices        │
│  ├── artic.json              ← Articulation info (motion type, etc.)         │
│  └── mesh_aligned_0.05.ply   ← 3D mesh of the scene                          │
│                                                                               │
│  scannetpp/[scene_id]/                                                       │
│  ├── coord.npy               ← Sampled point coordinates from mesh           │
│  ├── color.npy               ← RGB values per point                          │
│  ├── normal.npy              ← Surface normals per point                     │
│  ├── segment.npy             ← Semantic labels per point (0-99 ScanNet++)    │
│  └── instance.npy            ← Instance IDs per point                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
        ┌──────────────────────────────▼──────────────────────────────────┐
        │ STEP 1: PREPROCESSING                                            │
        │ $ python tools/preprocess_articulate3d.py                       │
        │                                                                  │
        │ For each scene:                                                │
        │  1. Load mesh and part annotations                             │
        │  2. For each vertex, vote over adjacent faces                  │
        │  3. Assign movable_label: 0=fixed, 1=rotation, 2=translation  │
        │  4. Assign interactable_label: 0=not, 1=interactable          │
        │  5. Save as .npy files                                         │
        └──────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSED LABELS (NEW!)                                │
│                                                                               │
│  data/articulate3d_labels/                                                   │
│  ├── [scene_id]_movable_label.npy        (N,) int64: {0, 1, 2}             │
│  ├── [scene_id]_interactable_label.npy   (N,) int64: {0, 1}                │
│  ├── [scene_id+1]_movable_label.npy                                         │
│  └── ...                                                                    │
│                                                                               │
│  Where N = number of points in the point cloud                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │ STEP 2: TRAINING DATA LOADING                            │
        │                                                          │
        │ ScanNetPPArticulateDataset loads per scene:             │
        │  ├── Point cloud: coord, color, normal                 │
        │  ├── Semantic labels: segment, instance                │
        │  └── NEW: movable_label, interactable_label            │
        │                                                          │
        │ + Data augmentations (rotation, color, etc.)            │
        │ + Voxelization (GridSample 0.02m)                       │
        │ + Sphere crops & downsampling                           │
        └──────────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │ STEP 3: MODEL FORWARD PASS                               │
        │                                                          │
        │ ┌────────────────────────────────────────────┐          │
        │ │       Volt Backbone (encoder + decoder)   │          │
        │ │       Output: features (N, 128)            │          │
        │ └──────────────┬───────────────────────────────┘          │
        │               │                                           │
        │     ┌─────────┼─────────┬─────────────┐                 │
        │     │         │         │             │                 │
        │ ┌──▼──┐  ┌───▼────┐ ┌─▼──────────┐                    │
        │ │Seg  │  │Movable │ │Interactable│                    │
        │ │Head │  │Head    │ │Head        │                    │
        │ │ 100 │  │  3 cls │ │  1 cls     │                    │
        │ └──┬──┘  └───┬────┘ └─┬──────────┘                    │
        │    │         │        │                                │
        │    ▼         ▼        ▼                                │
        │ seg_logits  mov_logits int_logits                      │
        │ (N, 100)   (N, 3)     (N, 1)                           │
        └────┬─────────┬─────────┬──────────────────────────────┘
             │         │         │
        ┌────▼─────────▼─────────▼──────────────────────────────┐
        │ STEP 4: LOSS COMPUTATION                              │
        │                                                      │
        │ L_seg = CE(seg_logits, segment_labels)              │
        │       + Lovasz(seg_logits, segment_labels)          │
        │                                                      │
        │ L_movable = CE(movable_logits, movable_binary_label)│
        │          + Dice(movable_logits, movable_binary_label)
        │                                                      │
        │ L_interact = CE(int_logits, interact_labels)        │
        │           + Dice(int_logits, interact_labels)       │
        │                                                      │
        │ L_total = L_seg + 0.5 × (L_movable + L_interact)   │
        └────┬──────────────────────────────────────────────────┘
             │
             ▼
        ┌──────────────────────────────────────┐
        │ STEP 5: BACKPROPAGATION               │
        │ Compute gradients, update weights     │
        │ (via AdamW + OneCycleLR)              │
        └──────────────────────────────────────┘
             │
             ▼
        [Repeat for 800 epochs]
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAINED MODEL CHECKPOINT                                │
│                                                                               │
│  exp/articulate3d_v1/                                                        │
│  ├── best_model.pth        ← Best validation mIoU                            │
│  ├── latest.pth            ← Latest checkpoint                               │
│  ├── config.py             ← Training config                                 │
│  └── logs/                 ← Training curves                                 │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
        ┌──────────────────▼────────────────────────────┐
        │ STEP 6: INFERENCE & EVALUATION                │
        │                                               │
        │ Load checkpoint, run on test set             │
        │ Get 3 outputs per point:                     │
        │  - seg_logits (N, 100)                       │
        │  - movable_logits (N, 3)                     │
        │  - interactable_logits (N, 1)                │
        │                                               │
        │ Extract instances:                           │
        │  1. Get movable probability from logits      │
        │  2. Cluster movable points (DBSCAN)          │
        │  3. Assign motion type per instance          │
        │  4. Extract interactable confidence          │
        └──────────────────┬────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PREDICTIONS FOR SUBMISSION                                 │
│                                                                               │
│  submissions/                                                                │
│  ├── [scene_id]_predictions.pkl                                              │
│  │   ├── 'movable_instances': (num_instances, N) bool array               │
│  │   ├── 'instance_motion_types': (num_instances,) {1, 2}                │
│  │   └── 'interactable_prob': (N,) float [0, 1]                          │
│  │                                                                        │
│  └── [scene_id+1]_predictions.pkl                                          │
│      └── ...                                                               │
│                                                                               │
│  Ready to submit to Articulate3D challenge! 🚀                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Size Estimation

### After Preprocessing
```
For a typical scene with ~100k points:

movable_label.npy       (~400 KB)
interactable_label.npy  (~400 KB)
─────────────────────
Per scene: ~800 KB

For 1000 scenes: ~800 MB total
```

### Training Checkpoint
```
best_model.pth          (~500 MB)  ← Full model weights
latest.pth              (~500 MB)  ← Just checkpointed
config.py               (~20 KB)
logs/                   (~10 MB)   ← Training metrics
─────────────────────
Per experiment: ~1 GB
```

### Predictions (per scene)
```
scene_predictions.pkl   (~2-5 MB)  ← Depends on num_instances
─────────────────────
For 1000 scenes: ~2-5 GB total
```

---

## Storage Requirements

```
Input:
  - ScanNet++ data:           ~50-100 GB (points, colors, normals, etc.)
  - Articulate3D raw:         ~30-50 GB (meshes, JSON files)
  ─────────────────────
  Total input:                ~100-150 GB

Processing:
  - Articulate3D labels:      ~1-2 GB (preprocessed .npy files)
  ─────────────────────
  Total processing:           ~1-2 GB

Output:
  - Trained checkpoint:       ~1 GB (models only)
  - Predictions:              ~2-5 GB (instance masks + probs)
  - Logs:                     ~100 MB
  ─────────────────────
  Total output:               ~3-6 GB

Total disk needed:            ~150-200 GB
```

---

## GPU Memory During Training

### Per GPU (batch size 16)
```
Volta V100 32GB:

Model weights:          ~2 GB
Activations:            ~8 GB (during forward)
Gradients:              ~2 GB (during backward)
Optimizer state:        ~4 GB (Adam: m, v buffers)
PyTorch overhead:       ~2 GB
─────────────────────
Total:                  ~18 GB used / 32 GB available
Utilization:            ~56%
```

### With Different Batch Sizes
```
Batch 32: OOM (exceeds 32GB)
Batch 16: 18 GB (safe)  ← Default
Batch 8:  12 GB (safe, slower)
Batch 4:  8 GB (slow, but works on 16GB)
```

---

## Training Timeline

### Single Pass Through Data
```
Preprocessing:           ~2-4 hours (depends on mesh complexity)
Per epoch:               ~5-8 minutes (on 4 × V100)
──────────────────
800 epochs:              ~65-110 hours (2.7-4.5 days on 4 GPUs)

Typical training duration: 7-10 days (accounting for validation, logging)
```

### Checkpointing
```
Every eval_epoch (50 epochs):  Save checkpoint + evaluate
                               ~5-10 minutes per eval
                               
800 epochs / 50 = 16 evals
16 × 8 min = ~2 hours overhead
```

---

## Data Validation Checklist

Before starting training, verify:

```
✅ ScanNet++ Data
  □ Each scene has: coord.npy, color.npy, normal.npy, segment.npy
  □ All .npy files have matching first dimension (number of points)
  □ Coordinates are float32, colors are uint8, labels are int32
  
✅ Articulate3D Raw Data
  □ Each scene has: parts.json, artic.json, mesh_aligned_0.05.ply
  □ JSON files are valid JSON (can parse with Python json.load)
  □ Meshes have triangles and vertices
  
✅ After Preprocessing
  □ data/articulate3d_labels/ exists and has files
  □ Files: {scene_id}_movable_label.npy, {scene_id}_interactable_label.npy
  □ Shape matches point cloud: (num_points,)
  □ Values in correct range: movable {0,1,2}, interactable {0,1}
  
✅ Training Config
  □ data_root points to ScanNet++ data directory
  □ articulation_root points to label directory
  □ Paths are absolute or relative from repo root
  □ No typos in file paths
```

---

## Common Issues & Data-Related Fixes

| Issue | Check |
|-------|-------|
| "Label file not found" | Verify preprocessing completed successfully |
| "Shape mismatch (1000,) vs (1001,)" | Point cloud and labels must have identical length |
| "Invalid label values" | Labels should be in specified range (0-2 for movable) |
| "Memory error during preprocessing" | Process scenes in batches, or reduce batch size |
| "Mesh loading fails" | Check PLY file is not corrupted, try opening in MeshLab |
| "Out of memory during training" | Reduce batch_size in config or point_max in GridSample |

---

## Tips for Large-Scale Preprocessing

If preprocessing 1000+ scenes:

```bash
# Run with progress bar
python tools/preprocess_articulate3d.py ... # Already has tqdm

# Or parallelize manually (if modifying preprocessing code):
from multiprocessing import Pool

def process_scene(scene_id):
    return preprocess_scene(scene_id, ...)

with Pool(processes=8) as pool:
    results = pool.map(process_scene, all_scene_ids)
```

---

## Verifying Data Quality

```python
# Quick sanity checks
import numpy as np
from pathlib import Path

articulation_root = Path("data/articulate3d_labels")

for npy_file in articulation_root.glob("*_movable_label.npy"):
    label = np.load(npy_file)
    print(f"{npy_file.name}:")
    print(f"  Shape: {label.shape}")
    print(f"  Unique values: {np.unique(label)}")
    print(f"  Value counts: {np.bincount(label)}")
    assert label.dtype == np.int64
    assert set(np.unique(label)).issubset({0, 1, 2})
```

---

## Summary

```
Raw Data                    Preprocessing              Training Data
──────────────────────────  ──────────────────────  ──────────────────────────
parts.json                                         
artic.json            ──→   Per-point labels    ──→  Points + Labels
mesh.ply                    (per-vertex voting)      + Augmentations
                                                      + Spatial transforms
coord/color/normals ──→     (merged with base)   ──→  Batched & loaded
segment/instance                                      into model

                            ↓ Preprocessing script
                            (2-4 hours for 1000 scenes)
                            
                            ↓ Training config + launch
                            (800 epochs on 4 GPUs)
                            
                            ↓ Inference
                            (instance extraction, DBSCAN)
                            
                            ✓ Predictions ready for submission
```

