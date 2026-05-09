"""
Preprocessing script for Articulate3D dataset.

Converts Articulate3D annotations (parts.json, artic.json) to per-vertex labels
and saves them alongside the scene's coord.npy for use during training.

Outputs per scene (all at vertex resolution, same N as coord.npy):
    {scene_id}_movable_label.npy        — (V,) int64  {0=fixed, 1=rotation, 2=translation}
    {scene_id}_interactable_label.npy   — (V,) int64  {0=not, 1=interactable}
    {scene_id}_instance_artic_label.npy — (V,) int64  {0=background, k=instance k}  [NEW]
    {scene_id}_axis_label.npy           — (V, 3) float32  unit axis direction         [NEW]
    {scene_id}_origin_label.npy         — (V, 3) float32  axis origin point           [NEW]

The three new files enable per-instance axis/origin regression at training time.
They go through the normal Pointcept transform pipeline (GridSample, SphereCrop, etc.)
alongside the other per-vertex arrays.

Usage:
    python tools/preprocess_articulate3d.py \
        --articulate_root /path/to/articulate3d/data \
        --scannetpp_root  /path/to/scannetpp/data \
        --output_root     data/articulate3d_labels

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
    parts_dict = json.load(open(parts_path)) if parts_path.exists() else {}

    artic_path = Path(articulate_root) / f"{scene_id}_artic.json"
    artic_dict = json.load(open(artic_path)) if artic_path.exists() else {}

    return mesh, parts_dict, artic_dict


def _build_vertex_to_faces(triangles, num_vertices):
    """Return a list mapping each vertex index to its adjacent face indices."""
    vertex_to_faces = [[] for _ in range(num_vertices)]
    for face_idx, face in enumerate(triangles):
        for v_idx in face:
            vertex_to_faces[v_idx].append(face_idx)
    return vertex_to_faces


def _parse_annotations(parts_dict):
    """
    Parse parts.json into two mappings:
        triangle_to_part: face_idx -> part_label (str)
        part_to_tris:     part_label -> list of face_idx
    """
    triangle_to_part = {}
    part_to_tris = {}
    if "data" in parts_dict and "annotations" in parts_dict["data"]:
        for ann in parts_dict["data"]["annotations"]:
            label = ann.get("label", "")
            for tri_idx in ann.get("triIndices", []):
                triangle_to_part[tri_idx] = label
                part_to_tris.setdefault(label, []).append(tri_idx)
    return triangle_to_part, part_to_tris


def compute_per_vertex_labels(mesh, parts_dict, artic_dict):
    """
    Compute binary movable/interactable labels for each mesh vertex.

    Uses majority-vote over adjacent faces: each vertex inherits the most
    common part label among its neighbouring triangles.

    Returns:
        movable_labels      (V,) int64 — 0=fixed, 1=rotation, 2=translation
        interactable_labels (V,) int64 — 0=not interactable, 1=interactable
    """
    num_vertices = len(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    movable_labels = np.zeros(num_vertices, dtype=np.int64)
    interactable_labels = np.zeros(num_vertices, dtype=np.int64)

    vertex_to_faces = _build_vertex_to_faces(triangles, num_vertices)
    triangle_to_part, _ = _parse_annotations(parts_dict)

    for v_idx in range(num_vertices):
        adjacent_faces = vertex_to_faces[v_idx]
        part_labels = [triangle_to_part[f] for f in adjacent_faces if f in triangle_to_part]
        if not part_labels:
            continue

        # Majority vote
        label_counter = {}
        for lbl in part_labels:
            label_counter[lbl] = label_counter.get(lbl, 0) + 1
        label = max(label_counter, key=label_counter.get)

        if label in artic_dict:
            motion = artic_dict[label].get("motion_type", "")
            if "rotation" in motion.lower():
                movable_labels[v_idx] = 1
            elif "translation" in motion.lower():
                movable_labels[v_idx] = 2
            else:
                movable_labels[v_idx] = 1  # default to rotation if type unclear

        part_name = label.lower()
        if "handle" in part_name or "knob" in part_name or "switch" in part_name:
            interactable_labels[v_idx] = 1

    return movable_labels, interactable_labels


# ── NEW: per-instance regression labels ────────────────────────────────────────

def compute_instance_labels(mesh, parts_dict, artic_dict):
    """
    Compute per-vertex instance IDs, axis directions, and axis origins.

    For each articulated part that has both 'axis' and 'origin' fields in
    artic.json, assigns a unique instance ID (1, 2, …) to all of its vertices
    and stores the unit axis and origin point.  Background vertices get 0 / zeros.

    Returns:
        instance_artic_label (V,)   int64  — 0=background, k=instance k
        axis_label           (V, 3) float32 — unit axis direction (zero for background)
        origin_label         (V, 3) float32 — axis origin point   (zero for background)

    The axis/origin values are the SAME for all vertices that belong to the same
    instance (redundant but makes per-vertex storage uniform, so GridSample and
    other transforms can subsample them like any other array).
    """
    num_vertices = len(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    instance_artic_label = np.zeros(num_vertices, dtype=np.int64)
    axis_label = np.zeros((num_vertices, 3), dtype=np.float32)
    origin_label = np.zeros((num_vertices, 3), dtype=np.float32)

    vertex_to_faces = _build_vertex_to_faces(triangles, num_vertices)
    triangle_to_part, part_to_tris = _parse_annotations(parts_dict)

    instance_id = 0  # incremented for each valid articulated part

    for part_label, artic_info in artic_dict.items():
        # Only process parts that have both axis and origin annotations
        if "axis" not in artic_info or "origin" not in artic_info:
            continue

        raw_axis = np.array(artic_info["axis"], dtype=np.float32)
        raw_origin = np.array(artic_info["origin"], dtype=np.float32)

        # Normalise axis to unit length
        axis_norm = np.linalg.norm(raw_axis)
        if axis_norm < 1e-6:
            continue  # degenerate axis — skip
        unit_axis = raw_axis / axis_norm

        # Collect triangles that belong to this part
        tri_indices = set(part_to_tris.get(part_label, []))
        if not tri_indices:
            continue

        # Find vertices adjacent to those triangles
        part_vertices = set()
        for v_idx in range(num_vertices):
            for f_idx in vertex_to_faces[v_idx]:
                if f_idx in tri_indices:
                    part_vertices.add(v_idx)
                    break

        if not part_vertices:
            continue

        instance_id += 1
        for v_idx in part_vertices:
            instance_artic_label[v_idx] = instance_id
            axis_label[v_idx] = unit_axis
            origin_label[v_idx] = raw_origin

    return instance_artic_label, axis_label, origin_label


def preprocess_scene(scene_id, articulate_root, scannetpp_root, output_root):
    """Preprocess one scene: compute and save all per-vertex label arrays."""
    try:
        mesh, parts_dict, artic_dict = load_scene_annotations(
            scene_id, articulate_root, scannetpp_root
        )

        # Existing: binary movable/interactable masks
        movable_labels, interactable_labels = compute_per_vertex_labels(
            mesh, parts_dict, artic_dict
        )

        # NEW: per-instance regression labels
        instance_artic_label, axis_label, origin_label = compute_instance_labels(
            mesh, parts_dict, artic_dict
        )

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        np.save(output_root / f"{scene_id}_movable_label.npy", movable_labels)
        np.save(output_root / f"{scene_id}_interactable_label.npy", interactable_labels)

        # NEW saves
        np.save(output_root / f"{scene_id}_instance_artic_label.npy", instance_artic_label)
        np.save(output_root / f"{scene_id}_axis_label.npy", axis_label)
        np.save(output_root / f"{scene_id}_origin_label.npy", origin_label)

        num_instances = int(instance_artic_label.max())
        print(
            f"  {scene_id}: {num_instances} articulated instances with axis/origin"
        )
        return True

    except Exception as e:
        print(f"Error processing {scene_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Articulate3D annotations to per-vertex label arrays"
    )
    parser.add_argument("--articulate_root", required=True,
                        help="Root directory of Articulate3D dataset")
    parser.add_argument("--scannetpp_root", required=True,
                        help="Root directory of ScanNet++ data")
    parser.add_argument("--output_root", default="data/articulate3d_labels",
                        help="Output directory for label arrays")
    args = parser.parse_args()

    articulate_root = Path(args.articulate_root)
    scene_ids = sorted({
        f.stem.rsplit("_", 1)[0]
        for f in articulate_root.glob("*_parts.json")
    })
    print(f"Processing {len(scene_ids)} scenes...")

    successful = sum(
        preprocess_scene(sid, args.articulate_root, args.scannetpp_root, args.output_root)
        for sid in tqdm(scene_ids)
    )
    print(f"\nSuccessfully processed {successful}/{len(scene_ids)} scenes")
    print(f"Labels saved to {args.output_root}")


if __name__ == "__main__":
    main()
