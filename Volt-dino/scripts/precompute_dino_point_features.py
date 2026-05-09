"""
Phase B — Assign DINOv2 patch features to every point in the Pointcept-processed
ScanNet++ point clouds, producing a  dino_feat.npy  file per scene.

Must be run AFTER Phase A (precompute_dino_image_features.py).

Algorithm (mirrors DITR §3.1 "2D-to-3D Mapping"):
  For each point in the scene's coord.npy (world-space mesh vertices):
    1. Try cameras in random order.
    2. Project the point into the camera image.
    3. If it falls inside the frame (depth > 0, pixel in bounds), read the
       DINOv2 patch feature at floor(u/14), floor(v/14).
    4. Assign and stop looking — first-visible camera wins.
  Points with no visible camera get a zero feature vector.

Run once (train+val) before the first training run:

    python scripts/precompute_dino_point_features.py \\
        --pointcept_root  data/scannetpp            \\
        --scannetpp_root  /data/scannetpp            \\
        --dino_image_dir  /data/dino_image_feats     \\
        --num_workers     8

Output per scene:
    <pointcept_root>/{train,val}/<scene_id>/dino_feat.npy  — float16  (N, D)

where N = number of mesh vertices in coord.npy (same as color.npy, normal.npy).

Storage estimate (D=1024, float16):
    ~500 scenes × avg 300 K pts × 1024 × 2 bytes ≈ 300 GB worst case.
    Use --dino_model dinov2_vitb14 (D=768) to save ~25 %.
    Rooms typically have 100-400 K vertices after mesh_aligned_0.05.ply sampling.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np


# ── Camera utilities ────────────────────────────────────────────────────────────

def load_camera_data(scene_id: str, scannetpp_root: Path) -> list[dict]:
    """
    Load intrinsics and world-to-camera extrinsics for every DSLR frame.

    ScanNet++ stores camera-to-world (c2w) transforms in nerfstudio format.
    We invert each c2w to get world-to-camera (w2c) for projection.

    Returns list of dicts with keys:
        image_id (str), K (3×3 float64), T_w2c (4×4 float64)
    """
    transforms_path = (
        scannetpp_root / "data" / scene_id / "dslr" / "nerfstudio" / "transforms.json"
    )
    if not transforms_path.exists():
        return []

    with open(transforms_path) as f:
        data = json.load(f)

    cameras = []
    for frame in data.get("frames", []):
        fl_x = frame.get("fl_x") or data.get("fl_x")
        fl_y = frame.get("fl_y") or data.get("fl_y")
        cx   = frame.get("cx")   or data.get("cx")
        cy   = frame.get("cy")   or data.get("cy")
        if None in (fl_x, fl_y, cx, cy):
            continue

        K = np.array([[fl_x, 0, cx],
                      [0, fl_y, cy],
                      [0,    0,  1]], dtype=np.float64)

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        w2c = np.linalg.inv(c2w)

        image_id = Path(frame["file_path"]).stem
        cameras.append({"image_id": image_id, "K": K, "T_w2c": w2c})

    return cameras


# ── Feature assignment ──────────────────────────────────────────────────────────

def assign_dino_features(
    coord: np.ndarray,          # (N, 3) float32 — world-space coordinates
    cameras: list[dict],
    dino_scene_dir: Path,
    patch_size: int = 14,
    num_cameras: int | None = None,   # limit cameras tried per training call; None = all
) -> np.ndarray:
    """
    For every point in coord, look up its DINOv2 feature from the best camera.

    Returns float16 array of shape (N, D), zeros for invisible points.
    """
    N = coord.shape[0]
    D = None  # determined from first loaded feature map

    # Shuffle cameras — random order provides unbiased assignment across runs
    rng = np.random.default_rng()
    cam_order = rng.permutation(len(cameras)).tolist()
    if num_cameras is not None:
        cam_order = cam_order[:num_cameras]

    # Points not yet assigned: True = still needs a feature
    unassigned = np.ones(N, dtype=bool)
    dino_feat = None  # will be (N, D) once D is known

    for cam_idx in cam_order:
        if not unassigned.any():
            break

        cam = cameras[cam_idx]
        feat_path = dino_scene_dir / f"{cam['image_id']}_dino.npy"
        meta_path = dino_scene_dir / f"{cam['image_id']}_meta.json"
        if not feat_path.exists():
            continue

        feat_map = np.load(feat_path)   # (H_p, W_p, D)  float16
        H_p, W_p, d = feat_map.shape
        H_img = H_p * patch_size
        W_img = W_p * patch_size

        if D is None:
            D = d
            dino_feat = np.zeros((N, D), dtype=np.float16)

        # Project world points into this camera
        T_w2c = cam["T_w2c"]  # (4, 4)
        K = cam["K"]          # (3, 3)

        pts_h = np.concatenate([coord, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, 4)
        pts_cam = (T_w2c @ pts_h.T).T      # (N, 4)
        depth = pts_cam[:, 2]              # (N,)

        # Only in-front points
        valid_depth = depth > 0.0

        # Perspective projection
        pts_proj = (K @ pts_cam[:, :3].T).T   # (N, 3)
        u = pts_proj[:, 0] / pts_proj[:, 2]   # pixel x
        v = pts_proj[:, 1] / pts_proj[:, 2]   # pixel y

        in_frame = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
        visible = valid_depth & in_frame & unassigned  # (N,)

        if not visible.any():
            continue

        # Floor-divide to get patch grid indices (DITR paper: no interpolation)
        pu = np.floor(u[visible] / patch_size).astype(np.int32).clip(0, W_p - 1)
        pv = np.floor(v[visible] / patch_size).astype(np.int32).clip(0, H_p - 1)

        dino_feat[visible] = feat_map[pv, pu]   # (num_visible, D)
        unassigned[visible] = False

    if D is None:
        # No camera had precomputed features — return placeholder zeros
        # We determine D from the first available feature file for this scene
        for cam in cameras:
            fp = dino_scene_dir / f"{cam['image_id']}_dino.npy"
            if fp.exists():
                D = np.load(fp).shape[-1]
                break
        if D is None:
            D = 1024  # fallback; will be all zeros anyway
        dino_feat = np.zeros((N, D), dtype=np.float16)

    assigned_frac = (~unassigned).sum() / N
    return dino_feat, assigned_frac


# ── Per-scene worker ────────────────────────────────────────────────────────────

def process_scene(
    scene_id: str,
    split: str,
    pointcept_root: Path,
    scannetpp_root: Path,
    dino_image_dir: Path,
    patch_size: int,
    overwrite: bool,
) -> None:
    scene_dir = pointcept_root / split / scene_id
    coord_path = scene_dir / "coord.npy"
    out_path   = scene_dir / "dino_feat.npy"

    if out_path.exists() and not overwrite:
        return

    if not coord_path.exists():
        print(f"  [WARN] coord.npy not found for {scene_id} ({split}), skipping.")
        return

    coord = np.load(coord_path).astype(np.float64)  # (N, 3)
    cameras = load_camera_data(scene_id, scannetpp_root)

    if not cameras:
        print(f"  [WARN] no camera data for {scene_id}, saving zeros.")
        dino_feat = np.zeros((coord.shape[0], 1024), dtype=np.float16)
        np.save(out_path, dino_feat)
        return

    dino_scene_dir = dino_image_dir / scene_id
    dino_feat, assigned_frac = assign_dino_features(
        coord, cameras, dino_scene_dir, patch_size
    )

    np.save(out_path, dino_feat)
    print(
        f"  {split}/{scene_id}: {coord.shape[0]} pts, "
        f"{assigned_frac:.1%} assigned, "
        f"shape={dino_feat.shape}, saved to {out_path}"
    )


def _worker(args):
    (scene_id, split, pointcept_root, scannetpp_root,
     dino_image_dir, patch_size, overwrite) = args
    try:
        process_scene(
            scene_id, split,
            Path(pointcept_root), Path(scannetpp_root),
            Path(dino_image_dir), patch_size, overwrite,
        )
    except Exception as e:
        print(f"  [ERROR] {split}/{scene_id}: {e}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase B: project mesh vertices into DINOv2 image features and save "
            "per-point dino_feat.npy alongside each scene's coord.npy."
        )
    )
    parser.add_argument(
        "--pointcept_root", required=True,
        help="Root of Pointcept-processed ScanNet++ data (contains train/ and val/ subdirs)"
    )
    parser.add_argument(
        "--scannetpp_root", required=True,
        help="Root of the raw ScanNet++ download (contains data/<scene_id>/dslr/…)"
    )
    parser.add_argument(
        "--dino_image_dir", required=True,
        help="Output dir from Phase A (precompute_dino_image_features.py)"
    )
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Subdirectories of pointcept_root to process"
    )
    parser.add_argument(
        "--split_lists", nargs="+",
        default=["nvs_sem_train.txt", "nvs_sem_val.txt"],
        help="Split list files under <scannetpp_root>/splits/ (one per --splits entry, same order)"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if dino_feat.npy already exists")
    args = parser.parse_args()

    pointcept_root = Path(args.pointcept_root)
    scannetpp_root = Path(args.scannetpp_root)
    dino_image_dir = Path(args.dino_image_dir)

    # Build list of (scene_id, split) pairs
    tasks = []
    for split, split_list_file in zip(args.splits, args.split_lists):
        split_path = scannetpp_root / "splits" / split_list_file
        if not split_path.exists():
            # Fall back: enumerate directories directly
            split_dir = pointcept_root / split
            if split_dir.exists():
                scene_ids = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
            else:
                print(f"[WARN] Neither {split_path} nor {split_dir} found, skipping {split}.")
                continue
        else:
            with open(split_path) as f:
                scene_ids = [line.strip() for line in f if line.strip()]

        for sid in scene_ids:
            tasks.append((
                sid, split,
                str(pointcept_root), str(scannetpp_root), str(dino_image_dir),
                args.patch_size, args.overwrite,
            ))

    print(f"Total scenes to process: {len(tasks)}")

    if args.num_workers <= 1:
        for t in tasks:
            _worker(t)
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            pool.map(_worker, tasks)

    print("Phase B complete.")


if __name__ == "__main__":
    main()
