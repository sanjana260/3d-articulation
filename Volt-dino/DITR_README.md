# DITR-Style DINOv2 Injection for Volt

This document covers the integration of 2D DINOv2 features into the Volt 3D segmentation backbone, following the design of [DITR (arXiv 2503.18944)](https://arxiv.org/abs/2503.18944).

---

## Overview

Volt is a purely 3D model: it processes point cloud voxels through a sparse conv tokenizer and a ViT-style transformer. DITR shows that injecting frozen DINOv2 patch features from 2D camera images into a 3D backbone produces a consistent +5–7 mIoU lift on ScanNet200-style benchmarks at no extra inference cost (after offline precomputation).

The core idea: for each 3D voxel, find the corresponding patch in one of the scene's DSLR images and retrieve its DINOv2 embedding. These embeddings carry rich semantic information (material, texture, object identity) that point cloud geometry alone cannot provide.

### Where the features are injected

```
Input voxels (N, 6)
       │
   [tokenizer]  ← SparseConv3d, stride 5
       │
 Patch tokens (T, 384)
       │  ← Level 1 injection: max-pool dino_feat to patch level, project 1024→384, add
   [12× Block]
       │
 Latent tokens (T, 384)
       │
   [decoder]    ← SparseInverseConv3d
       │
 Voxel features (N, 128)
       │  ← Level 2 injection: project dino_feat 1024→128, add
   [seg head]
       │
   Logits (N, C)
```

**Level 1 (patch, pre-transformer):** Enriches each patch token with the best 2D semantic feature from its voxel block before the transformer sees it.  
**Level 2 (voxel, post-decoder):** Directly adds a per-voxel DINOv2 signal to the final features before classification.

Both injections are **additive residuals** (no gating, no cross-attention). The projection layers are the only new learned weights.

---

## Quickstart

### Step 0 — Prerequisites

DINOv2 features are precomputed once and cached. You need:
- Raw ScanNet++ download (specifically `data/<scene_id>/dslr/` and `scans/mesh_aligned_0.05.ply`)
- Pointcept-preprocessed ScanNet++ data at `data/scannetpp/` (run `preprocess_scannetpp.py` first if you haven't)
- A GPU with ~16 GB VRAM for Phase A (DINOv2-ViT/L)

### Step 1 — Phase A: Extract per-image DINOv2 features

Runs DINOv2 on every DSLR image in ScanNet++ and saves spatial feature maps.  
**Run once. Takes ~20–40 min on a single A100 for the full dataset.**

```bash
python scripts/precompute_dino_image_features.py \
    --scannetpp_root  /data/scannetpp \
    --output_dir      /data/dino_image_feats \
    --model           dinov2_vitl14 \
    --num_workers     1
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `dinov2_vitl14` | ViT-L (D=1024). Use `dinov2_vitb14` (D=768) to save ~25% disk. |
| `--splits` | `nvs_sem_train.txt nvs_sem_val.txt` | Which split lists to process |
| `--num_workers` | `1` | Parallel processes. Each spawns its own CUDA context. Only increase if you have multiple GPUs or can share a GPU. |

Output structure:
```
/data/dino_image_feats/
└── <scene_id>/
    ├── <image_stem>_dino.npy    # float16, shape (H_p, W_p, 1024)
    └── <image_stem>_meta.json   # {"H_orig", "W_orig", "H_crop", "W_crop"}
```

Storage estimate: **~60–100 GB** for the full ScanNet++ train+val split with ViT-L.

---

### Step 2 — Phase B: Assign per-point DINOv2 features

For every point in each scene's `coord.npy`, projects it into the precomputed image features and saves a `dino_feat.npy` alongside it.  
**Run once. Takes ~1–3 hours with 8 CPU workers.**

```bash
python scripts/precompute_dino_point_features.py \
    --pointcept_root  data/scannetpp \
    --scannetpp_root  /data/scannetpp \
    --dino_image_dir  /data/dino_image_feats \
    --num_workers     8
```

Key options:

| Flag | Default | Notes |
|------|---------|-------|
| `--num_workers` | `8` | Pure CPU; safe to set high |
| `--overwrite` | off | Add this flag to recompute existing files |
| `--splits` | `train val` | Subdirs of `pointcept_root` to process |

After this step, each scene directory gains one new file:

```
data/scannetpp/
└── {train,val}/
    └── <scene_id>/
        ├── coord.npy
        ├── color.npy
        ├── normal.npy
        ├── segment.npy
        └── dino_feat.npy     ← NEW, float16, shape (N, 1024)
```

The Phase B script prints a coverage percentage for each scene. For well-covered indoor scenes you should see **> 90% assigned**. Low coverage (< 70%) usually means the `transforms.json` path or camera intrinsic fields don't match the expected format.

---

### Step 3 — Train

```bash
python tools/train.py configs/scannetpp/semseg-volt-dino.py
```

The model falls back to vanilla Volt silently if `dino_feat.npy` is missing for a scene — useful for debugging or mixed datasets.

To resume from a Volt-S/B checkpoint:

```bash
python tools/train.py configs/scannetpp/semseg-volt-dino.py \
    --options weight=/path/to/volt_pretrained.pth
```

The two new projection layers (`dino_proj_patch`, `dino_proj_voxel`) are initialised from scratch and will not be present in a pre-DITR checkpoint — that is fine. PyTorch will load the backbone weights and leave the new layers at their random init.

---

## Configuration Reference

All DITR-relevant options live under `model.backbone` in the config:

```python
backbone=dict(
    type="Volt",
    # ... existing params unchanged ...
    use_dino=True,     # False = vanilla Volt (projection layers not created)
    dino_dim=1024,     # Must match the DINOv2 variant used in precomputation:
                       #   dinov2_vitl14 → 1024
                       #   dinov2_vitb14 → 768
                       #   dinov2_vits14 → 384
),
```

And in the `Collect` transform for train/val:

```python
dict(
    type="Collect",
    keys=("coord", "grid_coord", "segment"),
    feat_keys=("color", "normal"),
    extra_keys=("dino_feat",),    # ← add this line
),
```

---

## Files Changed / Created

### New files

| File | Purpose |
|------|---------|
| `scripts/precompute_dino_image_features.py` | Phase A: run DINOv2 on DSLR images |
| `scripts/precompute_dino_point_features.py` | Phase B: project mesh vertices into image features |
| `configs/scannetpp/semseg-volt-dino.py` | Training config with DITR injection enabled |
| `DITR_README.md` | This file |

### Modified files

| File | What changed |
|------|-------------|
| `pointcept/models/volt/volt_base.py` | `use_dino` / `dino_dim` params; two projection layers; Level-1 and Level-2 injection in `forward()`; two module-level helper functions |
| `pointcept/datasets/scannetpp.py` | `dino_feat` added to `VALID_ASSETS`; float16→float32 cast in `get_data()` |

---

## Architecture Details

### Voxel-to-patch mapping

After `SparseConv3d(stride=5)`, each patch token corresponds to a 5×5×5 block of input voxels. The mapping is computed as:

```
patch_coord = grid_coord // stride       # for each input voxel (N, 3)
```

We then match each `(batch, patch_x, patch_y, patch_z)` tuple to the actual patch token index in `[0, T)` using a sort + `searchsorted` approach. This avoids large lookup tables and runs in O(N log N) time and O(N + T) memory.

### Max-pool aggregation

Multiple voxels within the same patch block may have different DINOv2 features (e.g., one part of a chair vs. another). We take the **element-wise max** across all voxels in a patch. This follows DITR §3.2 and tends to surface the most distinctive signal within each spatial block.

### Injection as residual addition

Both injection points add to (not replace) existing features:

```python
patch_tokens = patch_tokens + dino_proj_patch(X2D_patch)   # Level 1
voxel_feats  = voxel_feats  + dino_proj_voxel(dino_feat)   # Level 2
```

Setting `use_dino=False` skips both lines and returns a model identical to the original Volt.

### DINOv2 features are fixed per point

The precomputed `dino_feat.npy` is a property of each mesh vertex in the original scene. When training augmentations (random rotation, scale, flip) transform the 3D coordinates, the DINOv2 feature for each point remains the same — it reflects what that surface looks like in the 2D images. This is intentional: DINOv2 features are semantic identifiers of the surface, not orientation-dependent signals.

---

## Debugging

**Low coverage percentage in Phase B**  
Check that the camera intrinsic keys in `transforms.json` match `fl_x`, `fl_y`, `cx`, `cy`. Some ScanNet++ versions use per-frame vs. top-level placement. The script handles both but logs a warning if intrinsics are missing.

**`KeyError: dino_feat` during training**  
The `extra_keys=("dino_feat",)` line is missing from the `Collect` transform in your config, or Phase B hasn't been run for that split.

**Shape mismatch in `dino_proj_patch`**  
`dino_dim` in the config does not match the DINOv2 model used in Phase A. Verify: ViT-L → 1024, ViT-B → 768, ViT-S → 384.

**`scatter_reduce_` not found**  
Requires PyTorch ≥ 2.0. On PyTorch 1.x, replace with a manual loop or `torch_scatter.scatter_max`.

**OOM during training**  
`dino_feat` adds (N × 1024) floats to each batch, roughly 800 MB for a 204,800-point batch. Reduce `SphereCrop point_max` or switch to ViT-B (D=768) to cut this by 25%.
