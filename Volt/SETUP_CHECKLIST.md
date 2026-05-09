# Setup & Training Checklist for Articulate3D

Use this checklist to ensure everything is configured correctly before starting training.

---

## ✅ Phase 1: Preparation (Do This First)

### Data Organization
- [ ] ScanNet++ data is ready at: `{data_root}` (where you'll put the path)
  - [ ] Contains subdirectories for each scene_id
  - [ ] Each scene has: coord.npy, color.npy, normal.npy, segment.npy
  - [ ] Example path: `/path/to/scannetpp/train/scene_0/coord.npy`

- [ ] Articulate3D raw data is ready at: `{articulate_root}` 
  - [ ] Contains subdirectories for each scene_id
  - [ ] Each scene has: parts.json, artic.json, mesh_aligned_0.05.ply
  - [ ] Example path: `/path/to/articulate3d/scene_0/parts.json`

### Code Ready
- [ ] You're in the Volt repository: `/Users/sanjanamohan/Documents/Articulate\ 3D/Volt/`
- [ ] You have read/write access to this directory
- [ ] You have write access to data directory (for output labels)

### Dependencies Installed
- [ ] PyTorch (tested with 2.0+)
- [ ] Open3D (for mesh loading): `pip install open3d`
- [ ] SciPy & scikit-learn (for DBSCAN): `pip install scikit-learn`
- [ ] CUDA toolkit installed (for GPU training)

---

## ✅ Phase 2: Preprocessing (One-Time Only)

### Run Preprocessing Script
```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt

python tools/preprocess_articulate3d.py \
    --articulate_root /YOUR/PATH/TO/ARTICULATE/DATA \
    --scannetpp_root /YOUR/PATH/TO/SCANNETPP/DATA \
    --output_root data/articulate3d_labels
```

- [ ] Replace `/YOUR/PATH/TO/...` with actual paths
- [ ] Script runs without errors
- [ ] Check output directory exists: `data/articulate3d_labels/`
- [ ] Verify label files created: `ls data/articulate3d_labels/*.npy | wc -l`
  - [ ] Should show same number as total scenes
- [ ] Sanity check a file:
  ```bash
  python -c "import numpy as np; print(np.load('data/articulate3d_labels/[scene_id]_movable_label.npy').shape)"
  ```
  - [ ] Output should be `(N,)` where N = number of points

**⏱️ This takes 2-4 hours for 1000 scenes**

---

## ✅ Phase 3: Configuration

### Edit Training Config
Edit file: `configs/scannetpp/semseg-volt-articulate.py`

- [ ] Line ~47: Update `data_root`
  ```python
  data_root = "/YOUR/ACTUAL/PATH/TO/SCANNETPP"
  ```

- [ ] Line ~48: Update `articulation_root`
  ```python
  articulation_root = "/YOUR/ACTUAL/PATH/TO/articulate3d_labels"
  ```

- [ ] Verify paths are correct:
  ```bash
  ls {data_root}/train/ | head -1  # Should show a scene_id
  ls {articulation_root}/*.npy | head -1  # Should show label file
  ```

### Optional: Fine-tune Hyperparameters
- [ ] Batch size (default 16):
  - If GPU memory issues: reduce to 8 or 4
  - If you have extra memory: increase to 32

- [ ] Articulation loss weight (default 0.5):
  - If semantic seg performance drops: reduce to 0.2
  - If articulation perf weak: increase to 1.0

- [ ] Training duration (default 800 epochs):
  - Reduce to 400 for quick testing
  - Can increase to 1000 for better convergence

---

## ✅ Phase 4: Launch Training

### Check Prerequisites
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
  - [ ] Output should be: `True`

- [ ] Config loads without errors:
  ```bash
  python -c "from configs.scannetpp.semseg_volt_articulate import *; print('Config OK')"
  ```

- [ ] Model can be instantiated:
  ```bash
  python -c "from pointcept.models.builder import build_model; print('Models OK')"
  ```

### Single GPU (Testing)
```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate.py
```
- [ ] Runs without immediate errors
- [ ] First epoch completes successfully
- [ ] Training loss is reasonable (typically 1-10 range)

### Multi-GPU (Recommended)
```bash
python tools/train.py configs/scannetpp/semseg-volt-articulate.py --num_gpus 4
```
- [ ] If num_gpus > 1: ensure you have that many CUDA devices
  - [ ] Check: `nvidia-smi` shows all GPUs

### On HPC (SLURM)
```bash
sbatch -n 4 --gres=gpu:4 --time=48:00:00 train_job.sh
```
- [ ] Job script is saved and ready
- [ ] Proper environment variables set (if needed)
- [ ] GPU allocation matches num_gpus

---

## ✅ Phase 5: Training Monitoring

### Watch Initial Losses
First 10 iterations should show:
- [ ] `loss_seg_iter`: 2-5 (semantic segmentation)
- [ ] `loss_articulation_iter`: 0.5-2.0 (articulation)
- [ ] `total_loss`: 3-7

### After First Epoch (5-8 minutes)
- [ ] `epoch 1/800` completes
- [ ] Checkpoint saved to `exp/articulate3d_v1/`
- [ ] Validation metrics logged

### Every 50 Epochs (eval_epoch)
- [ ] Model validation runs
- [ ] Best model checkpoint updated
- [ ] Metrics improve or at least not catastrophically decrease

### If Issues Occur
- [ ] **OOM (Out of Memory):** Reduce batch_size to 8
- [ ] **Very slow:** Check `nvidia-smi` for GPU utilization
- [ ] **Loss NaN:** Check data paths are correct, labels not corrupted
- [ ] **No articulation loss:** Verify `has_articulation` is True (should print periodically)

---

## ✅ Phase 6: Inference & Evaluation

### After Training Completes

Load best model:
```python
from pointcept.models.builder import build_model
import torch

# Load config and model
cfg = ... # Load config
model = build_model(cfg.model)
checkpoint = torch.load("exp/articulate3d_v1/best_model.pth")
model.load_state_dict(checkpoint)
model.eval()

# Run on test data
with torch.no_grad():
    output = model(batch)
    seg_logits = output['seg_logits']
    movable_logits = output['movable_logits']
    interactable_logits = output['interactable_logits']
```

- [ ] Model loads without errors
- [ ] Forward pass runs and returns 3 outputs
- [ ] Output shapes match expectations:
  - `seg_logits`: (N, 100)
  - `movable_logits`: (N, 3)
  - `interactable_logits`: (N, 1)

### Extract Instances
```python
from tools.articulate_inference import extract_movable_instances

instances, motions, probs = extract_movable_instances(
    xyz, movable_logits, interactable_logits
)
```

- [ ] Function runs without errors
- [ ] `instances`: (num_instances, N) boolean array
- [ ] `motions`: (num_instances,) with values in {1, 2}
- [ ] `probs`: (N,) with values in [0, 1]

### Generate Submission
```python
from tools.articulate_inference import prepare_articulate_submission

submission_paths = prepare_articulate_submission(
    scene_predictions,
    output_dir="submissions"
)
```

- [ ] Submission files created in `submissions/`
- [ ] One `.pkl` file per scene
- [ ] Each file contains: movable_instances, instance_motion_types, interactable_prob

---

## ✅ Performance Expectations

### Training Time
- [ ] Single GPU: ~20-30 days for 800 epochs
- [ ] 4 GPUs: ~7-10 days (recommended)
- [ ] Single epoch: ~5-8 minutes (4 GPUs)

### Memory Usage
- [ ] 4× V100 (32GB): Batch 16 ✓
- [ ] 4× V100 (32GB): Batch 32 ✗ (OOM)
- [ ] 4× A100 (40GB): Can do batch 32

### Expected Metrics (depends on data)
- [ ] Semantic Segmentation mIoU: 50-55%
- [ ] Movable Recall: 60-75%
- [ ] Interactable IoU: 55-70%

---

## ✅ Troubleshooting Quick Reference

### Issue: "No module named 'pointcept'"
**Fix:**
```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt
# Install in editable mode
pip install -e .
```

### Issue: "CUDA out of memory"
**Fix:** In config, change:
```python
batch_size = 8  # from 16
# Or enable accumulation:
gradient_accumulation_steps = 2
```

### Issue: "articulation_root not found"
**Fix:**
- [ ] Run preprocessing first
- [ ] Verify path in config is correct
- [ ] Path should point to output of preprocessing (not raw data)

### Issue: "Path contains spaces, command fails"
**Fix:** Always quote paths:
```bash
python tools/train.py "configs/scannetpp/semseg-volt-articulate.py"
```

### Issue: "has_articulation is always False"
**Fix:**
- [ ] Check scene IDs match between ScanNet++ and label files
- [ ] Check label files exist in articulation_root
- [ ] Example: if point cloud is `scene_abc`, label should be `scene_abc_movable_label.npy`

---

## ✅ Before Submitting Results

- [ ] Best model checkpoint exists: `exp/articulate3d_v1/best_model.pth`
- [ ] Predictions generated for all test scenes
- [ ] Submission format correct (pickle with 3 keys)
- [ ] Instance counts reasonable (not 0, not millions)
- [ ] Probabilities in valid range [0, 1]
- [ ] Motion types are {1=rotation, 2=translation}

---

## File Organization Reference

After everything is set up:
```
Volt/
├── data/
│   └── articulate3d_labels/        ← Created by preprocessing
│       ├── [scene]_movable_label.npy
│       └── [scene]_interactable_label.npy
│
├── exp/
│   └── articulate3d_v1/            ← Created by training
│       ├── best_model.pth
│       ├── latest.pth
│       └── logs/
│
├── submissions/                    ← Created by inference
│   ├── [scene]_predictions.pkl
│   └── ...
│
└── configs/
    └── scannetpp/
        └── semseg-volt-articulate.py  ← ✏️ EDIT THIS
```

---

## Success Criteria

You're ready to submit when:

- ✅ Training ran for at least 100 epochs without crash
- ✅ Loss curves show reasonable downward trend
- ✅ Validation mIoU > 30% (should be 50%+)
- ✅ Predictions generated for all test scenes
- ✅ Submission files are valid and readable
- ✅ No NaN or Inf values in predictions

---

## Next Steps After Training

1. **Evaluate on validation set** (see test.py)
2. **Fine-tune hyperparameters** if needed (reduce articulation_weight, freeze backbone, etc.)
3. **Generate test predictions** using best checkpoint
4. **Verify submission format** matches Articulate3D requirements
5. **Upload to challenge platform**

---

## Support Resources

| Document | Purpose |
|----------|---------|
| `QUICKSTART_ARTICULATE3D.md` | 3-step overview |
| `ARTICULATE3D_README.md` | Complete guide (200+ lines) |
| `IMPLEMENTATION_SUMMARY.md` | Technical details |
| `DATA_PIPELINE.md` | Visual data flow |
| `CHANGES_SUMMARY.md` | Code modifications |

---

## Final Verification

Run this before training:
```bash
cd /Users/sanjanamohan/Documents/Articulate\ 3D/Volt

# 1. Check imports
python -c "from pointcept.models.volt.volt_articulate import VoltArticulate, ArticulateSegmentor; print('✓ Models OK')"

# 2. Check datasets
python -c "from pointcept.datasets.scannetpp import ScanNetPPArticulateDataset; print('✓ Dataset OK')"

# 3. Check losses
python -c "from pointcept.models.losses.articulation import ArticulationLoss; print('✓ Losses OK')"

# 4. Check config
python -c "from configs.scannetpp.semseg_volt_articulate import *; print('✓ Config OK')"

# All should print ✓
```

---

**You're all set! Good luck with training!** 🚀

*Last updated: 2026-05-06*
