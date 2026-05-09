# Summary of Changes for Articulate3D Support

## Overview
Added support for the Articulate3D dataset to Volt by implementing two auxiliary classification heads for movable and interactable part segmentation, alongside the primary semantic segmentation task.

## Files Modified/Created

### 1. Dataset Extensions
**File:** `pointcept/datasets/scannetpp.py`
- **NEW CLASS:** `ScanNetPPArticulateDataset`
- **Purpose:** Extended dataset loader that reads articulation labels (movable_label.npy, interactable_label.npy)
- **Key Features:**
  - Gracefully handles missing labels with `has_articulation` flag
  - Loads per-point movable labels (3 classes: 0=fixed, 1=rotation, 2=translation)
  - Loads per-point interactable labels (binary: 0/1)

### 2. Loss Functions
**File:** `pointcept/models/losses/articulation.py` (NEW)
- **NEW CLASSES:**
  - `ArticulationLoss`: Combined Dice + BCE loss for movable/interactable heads
  - `BinaryCrossEntropyWithDiceLoss`: Utility loss function
- **Implementation follows USDNet:** L_seg = λ_dice × L_dice + λ_ce × L_ce
- **Masked computation:** Loss only computed on scenes with annotations

**File:** `pointcept/models/losses/__init__.py`
- Added imports for new loss functions

### 3. Model Architecture
**File:** `pointcept/models/volt/volt_articulate.py` (NEW)
- **NEW CLASSES:**
  - `VoltArticulate`: Volt backbone with three output heads
    - Semantic segmentation head (100 classes, existing)
    - Movable part head (3 classes: background, rotation, translation)
    - Interactable part head (binary)
  - `ArticulateSegmentor`: Training/inference wrapper
    - Manages loss computation for all three tasks
    - Supports optional backbone freezing
    - Returns all three outputs during inference

**File:** `pointcept/models/volt/__init__.py`
- Added imports for articulation models

### 4. Training Configuration
**File:** `configs/scannetpp/semseg-volt-articulate.py` (NEW)
- **Model:** ArticulateSegmentor with Volt backbone
- **Loss Functions:**
  - Semantic segmentation: CrossEntropyLoss + LovaszLoss
  - Articulation: ArticulationLoss (Dice + BCE)
- **Key Hyperparameters:**
  - `articulation_weight = 0.5`: Relative weight of articulation loss
  - `lambda_dice = 1.0, lambda_ce = 1.0`: Dice/BCE balance
  - `freeze_backbone = False`: Option to freeze encoder
- **Dataset:** ScanNetPPArticulateDataset with articulation label loading
- **Training:** 800 epochs, batch size 16, AdamW optimizer

### 5. Inference & Utilities
**File:** `tools/articulate_inference.py` (NEW)
- **Functions:**
  - `extract_movable_instances()`: Clusters movable points into instances using DBSCAN
  - `prepare_articulate_submission()`: Formats predictions for submission
  - `evaluate_articulation()`: Computes evaluation metrics
- **Key Features:**
  - Threshold-based detection (default 0.5)
  - Spatial clustering with configurable parameters
  - Pickle/numpy export formats

**File:** `tools/preprocess_articulate3d.py` (NEW)
- **Purpose:** One-time preprocessing of raw Articulate3D annotations
- **Input:** 
  - Articulate3D raw data (parts.json, artic.json)
  - ScanNet++ meshes (mesh_aligned_0.05.ply)
- **Output:** Per-point label arrays (.npy files)
- **Method:** Vertex voting over adjacent faces for label assignment

### 6. Documentation
**File:** `ARTICULATE3D_README.md` (NEW)
- Complete setup and training guide
- Dataset structure specifications
- Preprocessing instructions
- Training commands and HPC examples
- Inference and submission format
- Troubleshooting guide

**File:** `CHANGES_SUMMARY.md` (THIS FILE)
- Overview of all modifications

## Key Design Decisions

### 1. Joint Training Architecture
- Single backbone shared by all three tasks
- Three separate output heads with minimal coupling
- End-to-end training by default, option to freeze backbone

### 2. Loss Masking
- Articulation loss only computed on scenes with labels
- `has_articulation` flag controls gradient flow
- Prevents label-sparse scenes from dominating the training signal

### 3. Mixed Batch Training
- Both labeled and unlabeled scenes in same batch
- More efficient than separate dataloaders
- Natural semi-supervised learning setup

### 4. Inference Format
- Per-point predictions at voxel resolution
- Instance clustering via DBSCAN
- Motion type assignment from class logits

## Data Flow

```
Input Point Cloud
       │
       ├─► Volt Backbone ──┐
       │                   │
       ├─ Features (N, 128) │
       │                   │
       ├──────────────────────┬──────────────────┬──────────┐
       │                      │                  │          │
       ▼                      ▼                  ▼          ▼
    Seg Head              Movable Head         Interact    
    (100 cls)             (3 cls)              Head (1 cls)
       │                      │                  │
       ▼                      ▼                  ▼
   Seg Loss              Movable Loss          Interact Loss
   (CE+Lovasz)           (Dice+BCE)            (Dice+BCE)
       │                      │                  │
       └──────────────────────┴──────────────────┘
                      │
                      ▼
              Total Combined Loss
                      │
                      ▼
                  Backpropagation
```

## Dataset Format Requirements

### Input (ScanNet++ standard)
```
scene_id/
├── coord.npy              # (N, 3) float32
├── color.npy              # (N, 3) uint8
├── normal.npy             # (N, 3) float32
├── segment.npy            # (N,) int32
└── instance.npy           # (N,) int32
```

### NEW: Articulation Labels
```
{scene_id}_movable_label.npy        # (N,) int64: {0, 1, 2}
{scene_id}_interactable_label.npy   # (N,) int64: {0, 1}
```

## Running Training

### Quick Start
```bash
# Single GPU
python tools/train.py configs/scannetpp/semseg-volt-articulate.py

# Multiple GPUs
python tools/train.py configs/scannetpp/semseg-volt-articulate.py --num_gpus 4
```

### Preprocessing (Required First)
```bash
python tools/preprocess_articulate3d.py \
    --articulate_root /path/to/articulate3d/raw \
    --scannetpp_root /path/to/scannetpp \
    --output_root data/articulate3d_labels
```

### Configuration Paths
Must update in `configs/scannetpp/semseg-volt-articulate.py`:
- `data_root`: Path to ScanNet++ data
- `articulation_root`: Path to preprocessed labels

## Performance Metrics

The model outputs three sets of predictions:
1. **Semantic Segmentation:** mIoU against 100-class ScanNet++ labels
2. **Movable Detection:** Recall/Precision for movable point detection
3. **Interactable Detection:** IoU for interactable point classification

## Backward Compatibility

- **No changes to existing code:** All modifications are additions or isolated extensions
- **Existing models still work:** Original Volt and ScanNetPPDataset unchanged
- **Flexible configuration:** Can train with or without articulation labels
