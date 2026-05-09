"""
Preprocessing script for Articulate3D dataset.

Converts the USDNet-processed Articulate3D data (from
USDNet/datasets/preprocessing/articulate3d_preprocessing_challenge.py)
into per-scene .npy files compatible with the Pointcept framework.

Each scene gets a directory under data/articulate3d/{split}/{scene_id}/
containing:
  - coord.npy     (N, 3) float32  - point coordinates
  - color.npy     (N, 3) float32  - RGB colors [0-255]
  - normal.npy    (N, 3) float32  - surface normals
  - segment.npy   (N,)   int32    - motion type: 0=static, 1=rotation, 2=translation
  - instance.npy  (N,)   int32    - instance ID (movable part ID, 0=background)
  - artic_axis.npy    (N, 3) float32 - articulation axis direction
  - artic_origin.npy  (N, 3) float32 - articulation origin point
  - artic_type.npy    (N,)   int32   - same as segment (0/1/2)
  - artic_range.npy   (N, 2) float32 - (range_min, range_max), zeros for static

Usage:
    python tools/preprocess_articulate3d.py \\
        --data_dir ./data/raw/articulate3d \\
        --save_dir ./data/articulate3d
"""

import os
import sys
import json
import argparse
import numpy as np

# Add USDNet to path so we can reuse its preprocessing
USDNET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "USDNet")
if os.path.exists(USDNET_PATH):
    sys.path.insert(0, USDNET_PATH)

from datasets.preprocessing.articulate3d_preprocessing_challenge import (
    Articulate3DPreprocessing,
    SceneParser,
    splits,
)
from loguru import logger


def process_scene(scene_folder, save_dir, split, ignore_index=0,
                  interaction_as_movement=False, exclude_stuff=False):
    """Process a single scene and save per-point .npy files."""
    scan_id = os.path.basename(scene_folder)

    try:
        scene_parser = SceneParser(
            scene_folder,
            interaction_as_movement=interaction_as_movement,
            exclude_stuff=exclude_stuff,
            mode=split,
        )
        coords, colors, normals, sem_gt, inst_gt, inter_gt, articulation_gt = \
            scene_parser.get_data(ignore_index=ignore_index)
    except Exception as e:
        logger.warning(f"Failed to parse scene {scan_id}: {e}")
        return False

    num_points = coords.shape[0]

    # Create output directory
    out_dir = os.path.join(save_dir, split, scan_id)
    os.makedirs(out_dir, exist_ok=True)

    # --- Save base data ---
    np.save(os.path.join(out_dir, "coord.npy"), coords.astype(np.float32))
    np.save(os.path.join(out_dir, "color.npy"), colors.astype(np.float32))
    np.save(os.path.join(out_dir, "normal.npy"), normals.astype(np.float32))
    np.save(os.path.join(out_dir, "segment.npy"), sem_gt.astype(np.int32))
    np.save(os.path.join(out_dir, "instance.npy"), inst_gt.astype(np.int32))

    # --- Build per-point articulation arrays ---
    artic_axis = np.zeros((num_points, 3), dtype=np.float32)
    artic_origin = np.zeros((num_points, 3), dtype=np.float32)
    artic_type = sem_gt.copy().astype(np.int32)  # 0=static, 1=rotation, 2=translation
    artic_range = np.zeros((num_points, 2), dtype=np.float32)

    for mov_id, params in articulation_gt.items():
        mask = inst_gt == mov_id
        if mask.sum() == 0:
            continue
        artic_axis[mask] = params['axis']
        artic_origin[mask] = params['origin']
        # Note: range_min/range_max not available in current preprocessing
        # They can be added later when the data becomes available

    np.save(os.path.join(out_dir, "artic_axis.npy"), artic_axis)
    np.save(os.path.join(out_dir, "artic_origin.npy"), artic_origin)
    np.save(os.path.join(out_dir, "artic_type.npy"), artic_type)
    np.save(os.path.join(out_dir, "artic_range.npy"), artic_range)

    logger.info(
        f"Scene {scan_id}: {num_points} points, "
        f"{sem_gt[sem_gt > 0].shape[0]} articulable "
        f"({(sem_gt == 1).sum()} rot, {(sem_gt == 2).sum()} trans)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess Articulate3D for DITR")
    parser.add_argument("--data_dir", type=str, default="./data/raw/articulate3d",
                        help="Path to raw Articulate3D data")
    parser.add_argument("--save_dir", type=str, default="./data/articulate3d",
                        help="Output directory for Pointcept-format data")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Which splits to process")
    parser.add_argument("--interaction_as_movement", action="store_true",
                        help="Treat interaction parts as movement parts")
    parser.add_argument("--exclude_stuff", action="store_true",
                        help="Exclude non-hierarchy stuff objects")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    total_scenes = 0
    success_scenes = 0

    for split in args.splits:
        logger.info(f"Processing split: {split}")
        split_file = os.path.join(args.data_dir, "splits", splits.get(split, f"{split}.txt"))

        if not os.path.exists(split_file):
            logger.warning(f"Split file not found: {split_file}, skipping")
            continue

        with open(split_file, "r") as f:
            scan_ids = [line.strip() for line in f if line.strip()]

        for scan_id in scan_ids:
            scene_folder = os.path.join(args.data_dir, "scans", scan_id)
            if not os.path.isdir(scene_folder):
                logger.warning(f"Scene folder not found: {scene_folder}")
                continue

            total_scenes += 1
            ok = process_scene(
                scene_folder, args.save_dir, split,
                interaction_as_movement=args.interaction_as_movement,
                exclude_stuff=args.exclude_stuff,
            )
            if ok:
                success_scenes += 1

    logger.info(f"Done: {success_scenes}/{total_scenes} scenes processed successfully")


if __name__ == "__main__":
    main()
