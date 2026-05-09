# Plan: Multi-Task Heads for DITR on Articulate3D

## Context
DITR (Point Transformer V3 backbone) currently does per-point semantic segmentation with a single linear head. The Articulate3D challenge requires predicting part segmentation + articulation parameters (axis, origin, motion type, motion range) per point. We keep the PT-v3 backbone and add 5 parallel prediction heads.

## Architecture

```
PT-v3 Backbone (frozen or fine-tuned)
    ↓
point.feat (N, 64)
    ↓
┌─────────┬──────────┬──────────┬──────────┬──────────┐
│ seg_head │ axis_head│origin_hdr│ type_head│range_head │
│ Linear   │ MLP→3D   │MLP→3D    │ MLP→3cls │ MLP→2D    │
│ 64→C     │ 64→128→3 │64→128→3  │ 64→128→3 │ 64→128→2  │
└─────────┴──────────┴──────────┴──────────┴──────────┘
```

Origin head predicts an *offset* from point coordinate; absolute origin = `coord + offset`.

## Files to Create/Modify

### 1. NEW: `ditr/pointcept/datasets/articulate3d.py`
- `Articulate3DDataset` extending `DefaultDataset`
- Loads per-point `.npy` files: `coord`, `color`, `normal`, `segment`, `artic_axis`, `artic_origin`, `artic_type`, `artic_range`
- Follows exact pattern of `ScanNetDataset` (`scannet.py`)

### 2. MODIFY: `ditr/pointcept/datasets/__init__.py`
- Add `from .articulate3d import Articulate3DDataset`

### 3. MODIFY: `ditr/pointcept/models/default.py`
- Add `ArticulateSegmentor` class after existing classes
- Heads: seg_head (existing), axis_head (MLP→3), origin_head (MLP→3 offset), type_head (MLP→3 cls), range_head (MLP→2)
- `_compute_losses()` method with masked articulation losses
- Losses computed inline (not via registry) because they need per-point masking

### 4. NEW: `ditr/configs/articulate3d/artic-semseg-pt-v3m1-0-base.py`
- Config modeled on `configs/scannet/semseg-pt-v3m1-0-base.py`
- Uses `ArticulateSegmentor`, `Articulate3DDataset`
- **No rotational augmentation** (RandomRotate/RandomFlip removed) since it would corrupt axis/origin labels
- `Collect` keys include `artic_type`, `artic_axis`, `artic_origin`, `artic_range`

### 5. MODIFY: `ditr/pointcept/engines/hooks/evaluator.py`
- Add `ArticEvaluator` hook for multi-task validation metrics:
  - Segmentation mIoU
  - Type accuracy (3-class)
  - Axis angle error (rotation parts, degrees)
  - Origin distance (point-to-line for rotation, L1 for translation)
  - Range L1 error

### 6. NEW: `ditr/tools/preprocess_articulate3d.py`
- Offline preprocessing script (runs once)
- Uses `SceneDataLoader` from USDNet to read `_parts.json` / `_artic.json`
- Converts face-based annotations → point-based labels
- Samples mesh into point cloud, assigns per-point: segment, axis, origin, type, range
- Saves as `.npy` files in standard Pointcept directory layout

## Loss Functions (in `_compute_losses()`)

| Loss | Scope | Formula |
|------|-------|---------|
| Seg CE + Lovasz | all points | existing criteria |
| Type CE | all points | `F.cross_entropy(type_logits, artic_type)` |
| Axis cosine | rotation pts only | `1 - |cos_sim(pred, gt)|` |
| Axis L1 | translation pts only | `F.l1_loss(normalize(pred), gt)` |
| Origin point-to-line | rotation pts only | `||(o_pred - o_gt) × axis_gt|| / ||axis_gt||` |
| Origin L1 | translation pts only | `F.l1_loss(o_pred, o_gt)` |
| Range L1 | articulable pts only | `F.l1_loss(range_pred, range_gt)` |

All articulation losses masked to 0 when no articulable points exist. Configurable weights via `axis_loss_weight`, `origin_loss_weight`, `type_loss_weight`, `range_loss_weight`.

## Per-Part Aggregation at Evaluation
- **Segmentation**: predicted part labels define segments
- **Type**: majority vote across part's points
- **Axis**: mean of predicted vectors → normalize
- **Origin**: mean of predicted origins
- **Range**: mean of predicted (min, max)

## Implementation Order
1. Preprocessing script (`preprocess_articulate3d.py`)
2. Dataset class (`articulate3d.py` + `__init__.py`)
3. Model class (`ArticulateSegmentor` in `default.py`)
4. Config file
5. Evaluator hook (`ArticEvaluator`)
6. Test & verify

## Key Decisions
- **Per-point (not query-based)**: directly extends DITR's architecture, simpler
- **No rotational augmentation**: avoids axis/origin corruption (can add co-rotation later)
- **Origin as offset**: gives spatial inductive bias (origin near part geometry)
- **Inline losses**: articulation losses computed in model (masked), not via Criteria registry
- **Existing hooks auto-log**: `InformationWriter` hook already logs all dict keys

## Verification
1. Run preprocessing on 1-2 scenes, verify `.npy` files
2. Run training for a few epochs, check all loss terms decrease
3. Run validation, check evaluator metrics are reasonable
4. Compare per-part aggregated predictions against USDNet baseline
