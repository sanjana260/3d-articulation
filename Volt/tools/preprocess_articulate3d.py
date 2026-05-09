"""
Preprocessing script for Articulate3D dataset.

Converts Articulate3D annotations (parts.json, artic.json) to per-point labels.
Produces three outputs per scene:
  - {scene_id}_movable_label.npy       : (V,) int64, values {0,1,2}
  - {scene_id}_interactable_label.npy  : (V,) int64, values {0,1}
  - {scene_id}_artic_instance_label.npy: (V,) int64, 0=background, N=instance N
  - {scene_id}_artic_instances.pkl     : list of per-instance dicts with axis/origin

Usage:
    python tools/preprocess_articulate3d.py \
        --articulate_root /path/to/articulate3d/data \
        --scannetpp_root /path/to/scannetpp/data \
        --output_root /path/to/output/labels

Author: Sanjana Mohan
"""

import os
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import open3d as o3d


def load_scene_annotations(scene_id, articulate_root, scannetpp_root):
    """Load mesh and annotations for a scene."""
    mesh_path = Path(scannetpp_root) / scene_id / "scans" / "mesh_aligned_0.05.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    parts_path = Path(articulate_root) / f"{scene_id}_parts.json"
    parts_json = json.load(open(parts_path)) if parts_path.exists() else {}

    artic_path = Path(articulate_root) / f"{scene_id}_artic.json"
    artic_dict = json.load(open(artic_path)) if artic_path.exists() else {}

    return mesh, parts_json, artic_dict


def build_triangle_to_part(parts_json):
    """Build a mapping from triangle index -> annotation dict.

    Uses vertIndices when available (faster, no face voting needed),
    otherwise falls back to triIndices.
    """
    tri_to_ann = {}
    vert_to_ann = {}

    if "data" not in parts_json or "annotations" not in parts_json["data"]:
        return tri_to_ann, vert_to_ann

    for ann in parts_json["data"]["annotations"]:
        # Prefer vertIndices — direct vertex assignment, no voting needed
        for v_idx in ann.get("vertIndices", []):
            vert_to_ann[v_idx] = ann
        for t_idx in ann.get("triIndices", []):
            tri_to_ann[t_idx] = ann

    return tri_to_ann, vert_to_ann


def compute_per_vertex_labels(mesh, parts_json, artic_dict):
    """Compute per-vertex movable, interactable, and instance ID labels.

    Args:
        mesh: open3d mesh
        parts_json: loaded parts.json (structure: {data: {annotations: [...]}})
        artic_dict: loaded artic.json (structure: {label: {motion_type, axis, origin}})

    Returns:
        movable_labels:         (V,) int64, {0=fixed, 1=rotation, 2=translation}
        interactable_labels:    (V,) int64, {0, 1}
        artic_instance_labels:  (V,) int64, 0=background, N=Nth articulated part
        instances:              list of dicts with instance-level metadata
    """
    num_vertices = len(mesh.vertices)
    movable_labels = np.zeros(num_vertices, dtype=np.int64)
    interactable_labels = np.zeros(num_vertices, dtype=np.int64)
    artic_instance_labels = np.zeros(num_vertices, dtype=np.int64)

    tri_to_ann, vert_to_ann = build_triangle_to_part(parts_json)

    # Fall back to face voting for vertices not covered by vertIndices
    if tri_to_ann:
        triangles = np.asarray(mesh.triangles)
        vertex_to_faces = [[] for _ in range(num_vertices)]
        for face_idx, face in enumerate(triangles):
            for v_idx in face:
                vertex_to_faces[v_idx].append(face_idx)

    # Assign per-annotation instance IDs (only articulated parts get an ID)
    # Build label -> instance_id map from artic_dict entries
    label_to_instance_id = {}
    instances = []
    for label, artic_info in artic_dict.items():
        motion_type_str = artic_info.get("motion_type", "rotation").lower()
        motion_type = 1 if "rotation" in motion_type_str else 2

        # MODIFIED: Parse axis and origin from artic.json
        # Expected artic.json fields: 'axis' (list[3]) and 'origin' (list[3])
        raw_axis = artic_info.get("axis", None)
        raw_origin = artic_info.get("origin", None)

        if raw_axis is None or raw_origin is None:
            # Skip instance if axis/origin are missing
            continue

        axis = np.array(raw_axis, dtype=np.float32)
        origin = np.array(raw_origin, dtype=np.float32)

        # Normalise axis to unit vector
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis = axis / norm

        instance_id = len(instances) + 1  # 1-indexed (0 = background)
        label_to_instance_id[label] = instance_id
        instances.append({
            "instance_id": instance_id,
            "label": label,
            "motion_type": motion_type,
            "axis": axis,
            "origin": origin,
            "vertex_mask": None,  # filled in below
        })

    # Per-vertex label assignment
    instance_vertex_sets = {inst["instance_id"]: [] for inst in instances}

    for v_idx in range(num_vertices):
        # Prefer direct vertIndices mapping
        if v_idx in vert_to_ann:
            ann = vert_to_ann[v_idx]
        elif tri_to_ann:
            # Fall back to majority face vote
            face_anns = [
                tri_to_ann[f]
                for f in vertex_to_faces[v_idx]
                if f in tri_to_ann
            ]
            if not face_anns:
                continue
            counter = {}
            for a in face_anns:
                key = a.get("label", "")
                counter[key] = counter.get(key, 0) + 1
            best_label = max(counter, key=counter.get)
            ann = next(a for a in face_anns if a.get("label") == best_label)
        else:
            continue

        label = ann.get("label", "")

        # Movable label
        if label in artic_dict:
            artic_info = artic_dict[label]
            motion_type_str = artic_info.get("motion_type", "rotation").lower()
            movable_labels[v_idx] = 1 if "rotation" in motion_type_str else 2

        # Interactable label
        part_name = label.lower()
        if "handle" in part_name or "knob" in part_name or "switch" in part_name:
            interactable_labels[v_idx] = 1

        # Instance ID label (only for articulated parts with axis/origin)
        if label in label_to_instance_id:
            inst_id = label_to_instance_id[label]
            artic_instance_labels[v_idx] = inst_id
            instance_vertex_sets[inst_id].append(v_idx)

    # Build per-instance vertex masks
    for inst in instances:
        v_set = instance_vertex_sets[inst["instance_id"]]
        mask = np.zeros(num_vertices, dtype=bool)
        if v_set:
            mask[v_set] = True
        inst["vertex_mask"] = mask

    # Drop instances with no assigned vertices
    instances = [inst for inst in instances if inst["vertex_mask"].any()]

    return movable_labels, interactable_labels, artic_instance_labels, instances


def preprocess_scene(scene_id, articulate_root, scannetpp_root, output_root):
    """Preprocess a single scene and save all label files."""
    try:
        mesh, parts_json, artic_dict = load_scene_annotations(
            scene_id, articulate_root, scannetpp_root
        )

        movable_labels, interactable_labels, artic_instance_labels, instances = \
            compute_per_vertex_labels(mesh, parts_json, artic_dict)

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        np.save(output_root / f"{scene_id}_movable_label.npy", movable_labels)
        np.save(output_root / f"{scene_id}_interactable_label.npy", interactable_labels)
        np.save(output_root / f"{scene_id}_artic_instance_label.npy", artic_instance_labels)

        with open(output_root / f"{scene_id}_artic_instances.pkl", "wb") as f:
            pickle.dump(instances, f)

        return True
    except Exception as e:
        print(f"Error processing {scene_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Articulate3D annotations to per-point labels"
    )
    parser.add_argument("--articulate_root", type=str, required=True)
    parser.add_argument("--scannetpp_root", type=str, required=True)
    parser.add_argument(
        "--output_root", type=str, default="data/articulate3d_labels"
    )
    args = parser.parse_args()

    articulate_root = Path(args.articulate_root)
    scene_ids = list(
        {f.stem.rsplit("_", 1)[0] for f in articulate_root.glob("*_parts.json")}
    )
    print(f"Processing {len(scene_ids)} scenes...")

    successful = 0
    for scene_id in tqdm(scene_ids):
        if preprocess_scene(
            scene_id, args.articulate_root, args.scannetpp_root, args.output_root
        ):
            successful += 1

    print(f"\nSuccessfully processed {successful}/{len(scene_ids)} scenes")
    print(f"Labels saved to {args.output_root}")


if __name__ == "__main__":
    main()
