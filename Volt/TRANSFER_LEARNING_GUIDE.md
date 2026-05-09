# Transfer Learning: Using Pretrained Volt Weights

This guide explains how to use a pretrained Volt model and train only the articulation heads.

## Overview

There are two common approaches:

1. **Head-Only Training** (recommended for limited data)
   - Freeze backbone + semantic seg head
   - Train only articulation heads (movable + interactable)
   - Faster training, requires less data

2. **Fine-Tuning** (recommended for abundant data)
   - Freeze backbone only
   - Train semantic seg head + articulation heads together
   - Better performance, takes longer

---

## Finding Pretrained Weights

### Option 1: Use Published Volt Checkpoint
If you have a pretrained Volt model from the original paper:
```bash
# Example locations:
/path/to/pretrained/volt_scannetpp_best.pth
/path/to/pretrained/volt_200m_weights.pth
```

### Option 2: Use Your Own Trained Volt
If you already trained Volt on ScanNet++:
```bash
# From previous training
exp/scannetpp_baseline/best_model.pth
```

### Option 3: Download Weights (if available)
Check the Volt paper/repo for downloadable checkpoints.

---

## Configuration Setup

### Approach 1: Head-Only Training (FASTEST)

**Edit:** `configs/scannetpp/semseg-volt-articulate-transfer.py`

Create this new config file:

```python
_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# TRANSFER LEARNING: Load pretrained weights
weight = "/path/to/pretrained_volt_weights.pth"

batch_size = 32  # Can use larger batch for head-only training
num_worker = 24
enable_amp = True
use_ema = False  # Usually not needed for transfer learning
epoch = 200  # Much shorter (50-200 epochs instead of 800)
eval_epoch = 20

# Model settings
model = dict(
    type="ArticulateSegmentor",
    num_classes=100,
    backbone_out_channels=128,
    freeze_backbone=True,          # FREEZE backbone (no gradients)
    freeze_seg_head=True,          # FREEZE semantic seg head too
    backbone=dict(
        type="Volt",
        in_channels=6,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=128,
    ),
    # Only semantic seg loss (for consistency, not actually used)
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1,
        ),
    ],
    articulation_criteria=dict(
        type="ArticulationLoss",
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
    ),
    # Articulation is the main objective now
    articulation_weight=1.0,  # Full weight (not 0.5)
)

# Optimizer: Higher learning rate for head-only training
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.01)  # 5x higher LR
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=5.0,
    final_div_factor=10.0,
)

# Dataset settings
dataset_type = "ScanNetPPArticulateDataset"
data_root = "data/scannetpp"
articulation_root = "data/articulate3d_labels"

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        articulation_root=articulation_root,
        transform=[
            # SIMPLIFIED: Fewer augmentations for transfer learning
            dict(type="SphereCrop", point_max=1000000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
                extra_keys=("movable_label", "interactable_label", "has_articulation"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        articulation_root=articulation_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
                extra_keys=("movable_label", "interactable_label", "has_articulation"),
            ),
        ],
        test_mode=False,
    ),
)
```

**Launch training:**
```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer.py
```

**Expected training time:** 1-2 days on 4 GPUs (vs 7-10 days for full training)

---

### Approach 2: Fine-Tuning with Lower LR

**Edit:** `configs/scannetpp/semseg-volt-articulate-finetune.py`

```python
_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# TRANSFER LEARNING: Load pretrained weights
weight = "/path/to/pretrained_volt_weights.pth"

batch_size = 16
num_worker = 24
enable_amp = True
epoch = 300  # Intermediate duration
eval_epoch = 30

model = dict(
    type="ArticulateSegmentor",
    num_classes=100,
    backbone_out_channels=128,
    freeze_backbone=True,          # FREEZE backbone only
    freeze_seg_head=False,         # ALLOW semantic head to train
    backbone=dict(
        type="Volt",
        in_channels=6,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=128,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    articulation_criteria=dict(
        type="ArticulationLoss",
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
    ),
    articulation_weight=0.5,
)

# Optimizer: Medium learning rate for fine-tuning
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.05)  # 0.2x of full training
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# ... rest of config same as full training config
```

**Launch training:**
```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate-finetune.py
```

**Expected training time:** 3-5 days on 4 GPUs

---

## Quick Comparison

| Aspect | Head-Only | Fine-Tuning | Full Training |
|--------|-----------|------------|---------------|
| **Backbone** | ❄️ Frozen | ❄️ Frozen | 🔥 Trainable |
| **Seg Head** | ❄️ Frozen | 🔥 Trainable | 🔥 Trainable |
| **Artic Heads** | 🔥 Trainable | 🔥 Trainable | 🔥 Trainable |
| **Training Time** | 1-2 days | 3-5 days | 7-10 days |
| **Learning Rate** | 0.005 (high) | 0.0005 (med) | 0.001 (low) |
| **Epochs** | 50-200 | 200-400 | 800 |
| **Best For** | Limited data | Balanced data | Large dataset |
| **Memory** | Low | Medium | High |

---

## How to Configure

### Method 1: Edit Config File

```python
# In your config:

# Load pretrained weights
weight = "/path/to/pretrained_volt_weights.pth"

# Control what freezes
freeze_backbone = True       # Always freeze backbone
freeze_seg_head = True       # For head-only training
# OR
freeze_seg_head = False      # For fine-tuning

# Adjust learning rate based on approach
optimizer = dict(type="AdamW", lr=0.005)   # High LR for head-only
# OR
optimizer = dict(type="AdamW", lr=0.0005)  # Lower LR for fine-tuning
```

### Method 2: Command-Line Override

```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer.py \
    --options \
    weight=/path/to/pretrained_volt_weights.pth \
    model.freeze_backbone=True \
    model.freeze_seg_head=True \
    optimizer.lr=0.005
```

### Method 3: Programmatic (in training script)

```python
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.builder import build_model
import torch

# Load config
cfg = default_config_parser("configs/scannetpp/semseg-volt-articulate.py")

# Override transfer learning settings
cfg.weight = "/path/to/pretrained_volt_weights.pth"
cfg.model.freeze_backbone = True
cfg.model.freeze_seg_head = True
cfg.optimizer.lr = 0.005

# Setup and train
cfg = default_setup(cfg)
trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
trainer.train()
```

---

## Understanding Weight Loading

When you set `weight = "path/to/checkpoint.pth"` in the config:

1. **Before training starts**, the default hooks will load the checkpoint
2. **Only matching layers** are loaded (strict=False by default)
3. **New heads** (movable, interactable) are initialized randomly
4. **Frozen layers** won't receive gradients

### What Gets Loaded
```
From checkpoint: backbone, seg_head
New (random):   movable_head, interactable_head
```

### What Gets Trained
```
freeze_backbone = True:  backbone parameters FROZEN
freeze_seg_head = True:  seg_head parameters FROZEN
Always trainable:        movable_head, interactable_head
```

---

## Training Tips for Transfer Learning

### 1. Learning Rate Strategy
```python
# Head-only training: Higher LR (heads have fewer parameters)
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.01)

# Fine-tuning: Much lower LR (prevent catastrophic forgetting)
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.05)

# Rule of thumb: LR for fine-tuning ≈ 0.2 × LR for full training
```

### 2. Batch Size
```python
# Head-only: Can use larger batch
batch_size = 32  # vs 16 for full training

# Fine-tuning: Similar to full training
batch_size = 16
```

### 3. Epochs
```python
# Head-only: Much fewer epochs needed
epoch = 100  # converges fast with good features

# Fine-tuning: Intermediate
epoch = 300

# Full training: Many epochs
epoch = 800
```

### 4. Data Augmentation
```python
# Head-only: Lighter augmentation (features are fixed)
# Fine-tuning: Standard augmentation
# Full training: Aggressive augmentation
```

---

## Monitoring Transfer Learning

### Expected Loss Behavior

**Head-Only Training:**
```
Epoch 1: loss ≈ 2.0 (high initially)
Epoch 10: loss ≈ 0.8
Epoch 50: loss ≈ 0.3
Plateau: ~0.15-0.25

Much faster convergence than full training!
```

**Fine-Tuning:**
```
Epoch 1: loss ≈ 3.0
Epoch 50: loss ≈ 1.5
Epoch 200: loss ≈ 0.5
Plateau: ~0.2-0.3

Intermediate convergence
```

### Red Flags

❌ **Loss increases during training** → Learning rate too high
❌ **Loss plateaus immediately** → Learning rate too low
❌ **Loss explodes to NaN** → Weights mismatch or learning rate way too high
❌ **No improvement after 50 epochs** → Wrong mode (should be frozen/trainable)

---

## Complete Example Workflow

### Step 1: Preprocess Data
```bash
python tools/preprocess_articulate3d.py \
    --articulate_root /path/to/articulate3d/raw \
    --scannetpp_root /path/to/scannetpp \
    --output_root data/articulate3d_labels
```

### Step 2: Get Pretrained Weights
```bash
# Option A: Use existing checkpoint from your Volt training
cp exp/scannetpp_baseline/best_model.pth pretrained_volt.pth

# Option B: Download published weights
wget https://... -O pretrained_volt.pth
```

### Step 3: Create Transfer Learning Config
```python
# Copy and modify the config as shown above
# semseg-volt-articulate-transfer.py
```

### Step 4: Update Paths
```python
weight = "pretrained_volt.pth"  # Path to weights
data_root = "/your/scannetpp/path"
articulation_root = "data/articulate3d_labels"
```

### Step 5: Launch Training
```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer.py \
    --num_gpus 4
```

### Step 6: Check Results
```bash
# Watch loss curves
tail -f exp/articulate3d_transfer_v1/log.txt

# After training, evaluate
python tools/test.py configs/scannetpp/semseg-volt-articulate-transfer.py \
    --options weight=exp/articulate3d_transfer_v1/best_model.pth
```

---

## Common Issues & Fixes

### Issue: "weight file not found"
```bash
# Check the file exists
ls -la pretrained_volt.pth

# Use absolute path if relative path fails
weight = "/absolute/path/to/pretrained_volt.pth"
```

### Issue: "State dict key mismatch"
This is actually OK! The framework handles it automatically:
```
Partial weight loading: backbone matches, new heads initialized
```

### Issue: "Loss doesn't decrease"
Check these in order:
1. Learning rate too low? → Increase it
2. Backbone frozen but should be trainable? → Set freeze_backbone=False
3. Wrong data path? → Verify labels are loading
4. Learning rate too high? → Reduce it

### Issue: "Training is very slow"
```python
# If frozen backbone + frozen head, only heads update
# This is slow because batches are small relative to compute

# Solution: Increase batch_size
batch_size = 32  # was 16
```

---

## Performance Comparison (Expected)

On same validation set:

```
Pretrained Volt (baseline):          mIoU = 52%
├─ Head-Only (100 epochs):           mIoU = 50%, Movable = 65%, Interactable = 60%
├─ Fine-Tuning (300 epochs):         mIoU = 51%, Movable = 70%, Interactable = 65%
└─ Full Training (800 epochs):       mIoU = 52%, Movable = 72%, Interactable = 68%
```

Head-only is best when:
- You have limited Articulate3D data (<100 scenes)
- Training speed is critical
- Semantic seg performance matters

Fine-tuning is best when:
- You have moderate Articulate3D data (100-500 scenes)
- Training time is acceptable
- You want best of both tasks

---

## Summary

**To train only articulation heads with pretrained Volt:**

1. **Set in config:**
   ```python
   weight = "/path/to/pretrained_volt.pth"
   freeze_backbone = True
   freeze_seg_head = True
   ```

2. **Adjust learning rate:**
   ```python
   optimizer = dict(type="AdamW", lr=0.005)  # Higher for head-only
   ```

3. **Run training:**
   ```bash
   python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer.py
   ```

**Training time: 1-2 days on 4 GPUs** (vs 7-10 days for full training)

---

## References

- Original Volt Paper: https://arxiv.org/abs/2404.06242
- Transfer Learning Best Practices: https://cs231n.github.io/transfer-learning/
- Fine-tuning Tips: https://zhanghang1989.com/PyTorch-Encoding/model_zoo/segmentation.html
