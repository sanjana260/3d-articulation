# Quick Start: Training Volt with Articulation Heads

## TL;DR - 3 Steps to Start Training

### Step 1: Preprocess Articulate3D Labels
```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt

python tools/preprocess_articulate3d.py \
    --articulate_root /path/to/your/articulate3d/data \
    --scannetpp_root /path/to/your/scannetpp/data \
    --output_root data/articulate3d_labels
```
**What it does:** Converts raw Articulate3D annotations to per-point label arrays (.npy files)

**Input required:**
- Articulate3D raw data with `parts.json` and `artic.json` per scene
- ScanNet++ meshes at `{scene_id}/mesh_aligned_0.05.ply`

**Output:** Label files in `data/articulate3d_labels/`

---

### Step 2: Update Config Paths
Edit `configs/scannetpp/semseg-volt-articulate.py`:
```python
data_root = "/path/to/your/scannetpp/data"  # Line ~47
articulation_root = "/path/to/your/articulate3d_labels"  # Line ~48
```

---

### Step 3: Run Training
```bash
# Single GPU
python tools/train.py configs/scannetpp/semseg-volt-articulate.py

# Multiple GPUs (recommended)
python tools/train.py configs/scannetpp/semseg-volt-articulate.py --num_gpus 4

# With output directory
python tools/train.py configs/scannetpp/semseg-volt-articulate.py \
    --options save_path=exp/articulate3d_v1
```

**Expected output:** 
- Training logs to console and `wandb`
- Checkpoints saved to `exp/articulate3d_v1/`
- Runs for 800 epochs (~7-10 days on 4 V100s)

---

## Directory Structure for Data

```
/path/to/your/scannetpp/data/
├── train/
│   ├── scene_0/
│   │   ├── coord.npy
│   │   ├── color.npy
│   │   ├── normal.npy
│   │   ├── segment.npy
│   │   └── instance.npy
│   └── scene_1/
│       └── ...
└── val/
    └── ...

/path/to/your/articulate3d_labels/  ← Created by preprocessing
├── scene_0_movable_label.npy
├── scene_0_interactable_label.npy
├── scene_1_movable_label.npy
├── scene_1_interactable_label.npy
└── ...
```

---

## What Gets Trained

The model learns **3 tasks simultaneously**:

| Task | Head | Classes | Loss |
|------|------|---------|------|
| Semantic Segmentation | `seg_head` | 100 (ScanNet++) | CE + Lovasz |
| Movable Part | `movable_head` | 3 (BG, Rotation, Translation) | Dice + BCE |
| Interactable Part | `interactable_head` | 2 (Not interactable, Interactable) | Dice + BCE |

**Loss balance:**
- Semantic segmentation loss: 1.0
- Articulation loss: 0.5 (can be tuned via `articulation_weight`)

---

## Key Configuration Options

In `configs/scannetpp/semseg-volt-articulate.py`:

```python
# Model hyperparameters
freeze_backbone = False      # Set True to only train heads
articulation_weight = 0.5    # Reduce if articulation hurts main task

# Loss weights
lambda_dice = 1.0            # Dice loss component
lambda_ce = 1.0              # Cross-entropy component

# Training
batch_size = 16              # Per GPU
epoch = 800
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
```

---

## Checking Training Progress

### Via Weights & Biases (W&B)
```bash
# Watch real-time metrics at: https://wandb.ai/your-username/Volt
# Set wandb_project in config if needed
```

### Via Logs
```bash
# Check loss values in console output:
# loss_seg_epoch, loss_articulation_epoch, loss/iter
```

### Checkpoint Files
```bash
# Checkpoints saved to exp/articulate3d_v1/
# Latest: exp/articulate3d_v1/latest.pth
# Best: exp/articulate3d_v1/best_model.pth (by validation mIoU)
```

---

## If Something Goes Wrong

### "No such file or directory: /path/to/articulate3d_labels"
✅ Run preprocessing step first (Step 1)

### "CUDA out of memory"
Edit config:
```python
batch_size = 8              # Reduce from 16
gradient_accumulation_steps = 2  # Or enable accumulation
empty_cache = True          # Clear cache after each step
```

### "articulation_root not found"
Verify path in config matches preprocessing output:
```bash
ls data/articulate3d_labels/*.npy | head -5
# Should show your scene files
```

### "has_articulation returns False"
Either:
1. Preprocessing didn't create labels (check output directory)
2. Scene names don't match (verify scene_id format)
3. Label files have wrong naming convention (must be `{scene_id}_*.npy`)

### Training stalls / very slow
- Check if data loading works: reduce to 1 worker for debugging
- Verify GPU utilization: `nvidia-smi`
- Check disk I/O: might be bottleneck if data is remote

---

## Inference & Testing

After training, generate predictions:

```python
# Load checkpoint
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.builder import build_model

cfg = default_config_parser("configs/scannetpp/semseg-volt-articulate.py")
cfg.weight = "exp/articulate3d_v1/best_model.pth"
model = build_model(cfg.model).cuda().eval()

# Run inference
output = model(batch)  # Returns dict with 3 outputs:
# - seg_logits: (N, 100)
# - movable_logits: (N, 3)
# - interactable_logits: (N, 1)

# Extract instances for submission
from tools.articulate_inference import extract_movable_instances
instances, motions, interactable = extract_movable_instances(
    batch['coord'], output['movable_logits'], output['interactable_logits']
)
```

See `tools/articulate_inference.py` for more utilities.

---

## Files Changed & Created

### New Files (no impact on existing code)
- ✨ `pointcept/datasets/scannetpp.py` — Extended with `ScanNetPPArticulateDataset`
- ✨ `pointcept/models/volt/volt_articulate.py` — New model with articulation heads
- ✨ `pointcept/models/losses/articulation.py` — Articulation-specific losses
- ✨ `configs/scannetpp/semseg-volt-articulate.py` — Training config
- ✨ `tools/preprocess_articulate3d.py` — Preprocessing script
- ✨ `tools/articulate_inference.py` — Inference utilities
- 📖 `ARTICULATE3D_README.md` — Full documentation
- 📖 `CHANGES_SUMMARY.md` — Detailed change list
- 📖 `QUICKSTART_ARTICULATE3D.md` — This file

### Modified Files (backward compatible)
- `pointcept/models/losses/__init__.py` — Added imports only
- `pointcept/models/volt/__init__.py` — Added imports only

All changes are **backward compatible** — existing code and configs still work.

---

## Support

For detailed documentation, see:
- **Setup guide:** `ARTICULATE3D_README.md`
- **Technical details:** `CHANGES_SUMMARY.md`
- **API reference:** Docstrings in source files

Good luck with training! 🚀
