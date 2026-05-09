"""
Phase A — Precompute DINOv2 patch features for every DSLR image in ScanNet++.

For each image, runs DINOv2 once and saves a spatial feature map
  (H_patches × W_patches × D)  as float16 .npy   alongside a JSON sidecar
  recording the exact crop dimensions used.  These files are consumed by Phase B.

Run once before training:

    python scripts/precompute_dino_image_features.py \\
        --scannetpp_root  /data/scannetpp           \\
        --output_dir      /data/dino_image_feats    \\
        --model           dinov2_vitl14             \\
        --num_workers     4

Outputs per scene:
    <output_dir>/<scene_id>/<image_stem>_dino.npy   — float16  (H_p, W_p, D)
    <output_dir>/<scene_id>/<image_stem>_meta.json  — {"H_orig", "W_orig", "H_crop", "W_crop"}

Storage estimate (dinov2_vitl14, D=1024, float16):
    ~500 scenes × ~300 imgs × median(H_p × W_p) × 1024 × 2 bytes ≈ 60-100 GB
    Use --model dinov2_vitb14  (D=768) to cut this by ~25 %.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# ── DINOv2 model loading ────────────────────────────────────────────────────────

def load_dino_model(model_name: str) -> torch.nn.Module:
    """Load a frozen DINOv2 model from torch.hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Per-image processing ────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def process_image(model, img_path: Path, out_dir: Path, patch_size: int = 14) -> None:
    """Run DINOv2 on one image and save the feature map + metadata."""
    stem = img_path.stem
    feat_path = out_dir / f"{stem}_dino.npy"
    meta_path = out_dir / f"{stem}_meta.json"

    if feat_path.exists() and meta_path.exists():
        return  # already done — skip

    img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = img.size

    # Crop to a multiple of patch_size so DINOv2 tiling is exact
    W_crop = (W_orig // patch_size) * patch_size
    H_crop = (H_orig // patch_size) * patch_size
    img = img.crop((0, 0, W_crop, H_crop))

    img_t = _transform(img).unsqueeze(0).cuda()  # (1, 3, H_crop, W_crop)

    with torch.no_grad():
        # Returns a list of length n; each element is (1, H_p*W_p, D)
        # return_class_token=False → only patch tokens, no CLS
        feat = model.get_intermediate_layers(
            img_t, n=1, return_class_token=False
        )[0]  # (1, H_p * W_p, D)

    H_p = H_crop // patch_size
    W_p = W_crop // patch_size
    D   = feat.shape[-1]

    feat_np = feat.squeeze(0).reshape(H_p, W_p, D).cpu().to(torch.float16).numpy()

    np.save(feat_path, feat_np)
    with open(meta_path, "w") as f:
        json.dump(
            {"H_orig": H_orig, "W_orig": W_orig, "H_crop": H_crop, "W_crop": W_crop},
            f,
        )


# ── Per-scene processing ────────────────────────────────────────────────────────

def process_scene(
    scene_id: str,
    scannetpp_root: Path,
    output_dir: Path,
    model_name: str,
    patch_size: int,
) -> None:
    """Process all DSLR images for one scene."""
    img_dir = scannetpp_root / "data" / scene_id / "dslr" / "undistorted_images"
    if not img_dir.exists():
        print(f"  [WARN] image dir not found: {img_dir}")
        return

    out_dir = output_dir / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model fresh in each worker process (GPU is process-local)
    model = load_dino_model(model_name)

    img_paths = sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.jpg"))
    print(f"  Scene {scene_id}: {len(img_paths)} images")

    for img_path in img_paths:
        try:
            process_image(model, img_path, out_dir, patch_size)
        except Exception as e:
            print(f"    [ERROR] {img_path.name}: {e}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def _worker(args):
    scene_id, scannetpp_root, output_dir, model_name, patch_size = args
    process_scene(scene_id, Path(scannetpp_root), Path(output_dir), model_name, patch_size)


def main():
    parser = argparse.ArgumentParser(
        description="Phase A: precompute DINOv2 patch features for every ScanNet++ DSLR image."
    )
    parser.add_argument("--scannetpp_root", required=True,
                        help="Root of the raw ScanNet++ dataset (contains data/<scene_id>/…)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save per-image .npy feature maps")
    parser.add_argument("--model", default="dinov2_vitl14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model variant")
    parser.add_argument("--patch_size", type=int, default=14,
                        help="DINOv2 patch size (14 for all standard variants)")
    parser.add_argument("--splits", nargs="+",
                        default=["nvs_sem_train.txt", "nvs_sem_val.txt"],
                        help="Split files under <scannetpp_root>/splits/")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel processes (each gets one GPU stream)."
                             " Set to 1 if you have a single GPU.")
    args = parser.parse_args()

    scannetpp_root = Path(args.scannetpp_root)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all scene IDs from the requested splits
    scene_ids = []
    for split_file in args.splits:
        split_path = scannetpp_root / "splits" / split_file
        if not split_path.exists():
            print(f"[WARN] split file not found: {split_path}")
            continue
        with open(split_path) as f:
            scene_ids.extend(line.strip() for line in f if line.strip())
    scene_ids = sorted(set(scene_ids))
    print(f"Total scenes to process: {len(scene_ids)}")

    worker_args = [
        (sid, str(scannetpp_root), str(output_dir), args.model, args.patch_size)
        for sid in scene_ids
    ]

    if args.num_workers <= 1:
        for wa in worker_args:
            _worker(wa)
    else:
        # Each worker spawns its own CUDA context — use spawn start method
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=args.num_workers) as pool:
            pool.map(_worker, worker_args)

    print("Phase A complete.")


if __name__ == "__main__":
    main()
