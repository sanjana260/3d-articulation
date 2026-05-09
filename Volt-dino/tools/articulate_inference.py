"""
Inference utilities for Articulate3D predictions.

This module provides functions to:
1. Convert per-point predictions to instance masks
2. Format predictions for Articulate3D submission
3. Evaluate articulation predictions

Author: Sanjana Mohan
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import pickle
from pathlib import Path


# ===== ADDED: Instance extraction from movable points =====
def extract_movable_instances(
    xyz, movable_logits, interactable_logits, threshold=0.5, min_points=20
):
    """Extract movable object instances from point predictions.

    Args:
        xyz: (N, 3) point coordinates
        movable_logits: (N, 3) movable class logits or (N,) binary logits
        interactable_logits: (N, 1) or (N,) interactable logits
        threshold: probability threshold for movable detection
        min_points: minimum points per instance

    Returns:
        movable_instances: (num_instances, N) binary mask array
        instance_motion_types: (num_instances,) motion type per instance (1=rotation, 2=translation)
        interactable_mask: (N,) interactable probability
    """
    # Get movable probability
    if movable_logits.shape[-1] == 3:
        # 3-class output: softmax and take rotation + translation probability
        movable_probs = torch.softmax(movable_logits, dim=-1)
        movable_prob = (movable_probs[:, 1] + movable_probs[:, 2]).cpu().numpy()
        motion_class = movable_probs[:, 1:].argmax(dim=-1).cpu().numpy() + 1  # 1 or 2
    else:
        # Binary output
        movable_prob = torch.sigmoid(movable_logits.squeeze(-1)).cpu().numpy()
        motion_class = np.ones_like(movable_prob, dtype=np.int32)  # default to rotation

    movable_mask = movable_prob > threshold

    if not movable_mask.any():
        return np.zeros((0, len(movable_prob)), dtype=bool), np.array(
            [], dtype=np.int32
        ), torch.sigmoid(interactable_logits).cpu().squeeze().numpy()

    # Cluster movable points using DBSCAN
    xyz_np = xyz.cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
    movable_points = xyz_np[movable_mask]
    movable_indices = np.where(movable_mask)[0]

    # DBSCAN clustering with spatial distance
    if len(movable_points) > 0:
        clustering = DBSCAN(eps=0.05, min_samples=min_points).fit(movable_points)
        cluster_labels = clustering.labels_

        # Extract instance masks (ignore noise label -1)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        instances = []
        instance_motions = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points = movable_indices[cluster_mask]
            instance_mask = np.zeros(len(movable_prob), dtype=bool)
            instance_mask[cluster_points] = True
            instances.append(instance_mask)
            # Use majority motion type in the cluster
            instance_motions.append(np.bincount(motion_class[cluster_points]).argmax() + 1)

        if instances:
            movable_instances = np.stack(instances, axis=0)
            instance_motion_types = np.array(instance_motions, dtype=np.int32)
        else:
            movable_instances = np.zeros((0, len(movable_prob)), dtype=bool)
            instance_motion_types = np.array([], dtype=np.int32)
    else:
        movable_instances = np.zeros((0, len(movable_prob)), dtype=bool)
        instance_motion_types = np.array([], dtype=np.int32)

    # Interactable probability
    interactable_prob = (
        torch.sigmoid(interactable_logits).cpu().squeeze().numpy()
    )

    return movable_instances, instance_motion_types, interactable_prob


# ── Regression: axis and origin prediction from clustered instances ─────────────

def predict_instance_axes_origins(features, instance_masks, axis_head, origin_head):
    """Predict articulation axis and origin for each detected instance.

    For each instance (identified by DBSCAN in extract_movable_instances), we
    mean-pool the backbone features over the instance's point set and pass them
    through the regression heads.

    Args:
        features:       (N, D) backbone voxel features as a torch.Tensor
        instance_masks: (K, N) bool numpy array — one row per instance
        axis_head:      AxisHead module (from VoltArticulate)
        origin_head:    OriginHead module (from VoltArticulate)

    Returns:
        pred_axes:    (K, 3) numpy array — unit axis directions
        pred_origins: (K, 3) numpy array — predicted origin points
    """
    import torch.nn.functional as F

    if instance_masks.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    device = features.device
    pooled = []
    for mask in instance_masks:
        idx = torch.from_numpy(mask).to(device)
        pooled.append(features[idx].mean(dim=0))

    pooled = torch.stack(pooled, dim=0)  # (K, D)

    was_training = axis_head.training
    axis_head.eval()
    origin_head.eval()
    with torch.no_grad():
        axis_raw  = axis_head(pooled)
        origins   = origin_head(pooled)
        axes      = F.normalize(axis_raw, dim=1)

    if was_training:
        axis_head.train()
        origin_head.train()

    return axes.cpu().numpy(), origins.cpu().numpy()


def evaluate_axis_origin(pred_axes, pred_origins, gt_axes, gt_origins):
    """Compute axis angular error (degrees) and origin point-to-line distance (metres).

    Args:
        pred_axes:    (K, 3) predicted unit axes
        pred_origins: (K, 3) predicted origin points
        gt_axes:      (K, 3) GT unit axes
        gt_origins:   (K, 3) GT origin points

    Returns:
        metrics dict with 'mean_axis_err_deg' and 'mean_origin_dist'
    """
    if len(pred_axes) == 0:
        return {"mean_axis_err_deg": float("nan"), "mean_origin_dist": float("nan")}

    # Angular error: min(angle, 180-angle) to handle ± ambiguity
    cos_sim = np.clip(np.abs((pred_axes * gt_axes).sum(axis=1)), -1.0, 1.0)
    axis_err_deg = np.degrees(np.arccos(cos_sim))

    # Point-to-line distance: perpendicular component of (pred_origin - gt_origin)
    delta    = pred_origins - gt_origins
    parallel = (delta * gt_axes).sum(axis=1, keepdims=True) * gt_axes
    perp     = delta - parallel
    origin_dist = np.linalg.norm(perp, axis=1)

    return {
        "mean_axis_err_deg": float(axis_err_deg.mean()),
        "mean_origin_dist":  float(origin_dist.mean()),
    }


# ===== ADDED: Prepare submission format =====
def prepare_articulate_submission(
    scene_predictions,
    output_dir="submissions",
    format="pickle",
):
    """Prepare predictions in Articulate3D submission format.

    Args:
        scene_predictions: dict mapping scene_id to predictions dict with:
            - 'movable_instances': (num_instances, N) binary masks
            - 'instance_motions': (num_instances,) motion types
            - 'interactable_prob': (N,) interactable probabilities
        output_dir: directory to save submissions
        format: 'pickle' or 'numpy'

    Returns:
        submission_paths: dict of scene_id -> file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_paths = {}

    for scene_id, pred_dict in scene_predictions.items():
        # Prepare submission dict
        submission = {
            "movable_instances":    pred_dict.get("movable_instances", np.zeros((0, 1))),
            "instance_motion_types": pred_dict.get("instance_motions", np.array([], dtype=np.int32)),
            "interactable_prob":    pred_dict.get("interactable_prob", np.array([])),
            # Regression outputs (present when regression heads were run)
            "instance_axes":        pred_dict.get("instance_axes",   np.zeros((0, 3), dtype=np.float32)),
            "instance_origins":     pred_dict.get("instance_origins", np.zeros((0, 3), dtype=np.float32)),
        }

        # Save to file
        if format == "pickle":
            save_path = output_dir / f"{scene_id}_predictions.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(submission, f)
        elif format == "numpy":
            save_path = output_dir / f"{scene_id}_predictions.npz"
            np.savez_compressed(save_path, **submission)
        else:
            raise ValueError(f"Unknown format: {format}")

        submission_paths[scene_id] = str(save_path)

    return submission_paths


# ===== ADDED: Evaluation metrics =====
def evaluate_articulation(
    pred_movable_instances,
    pred_motion_types,
    pred_interactable,
    gt_movable_labels,
    gt_interactable_labels,
    pred_axes=None,
    pred_origins=None,
    gt_axes=None,
    gt_origins=None,
):
    """Compute evaluation metrics for articulation prediction.

    Args:
        pred_movable_instances: (num_pred_instances, N) binary masks
        pred_motion_types:      (num_pred_instances,) predicted motion types
        pred_interactable:      (N,) predicted interactable probability
        gt_movable_labels:      (N,) ground truth movable labels (0=fixed, 1=rotation, 2=translation)
        gt_interactable_labels: (N,) ground truth interactable (0 or 1)
        pred_axes:    (K, 3) predicted unit axes — optional
        pred_origins: (K, 3) predicted origins  — optional
        gt_axes:      (K, 3) GT unit axes        — optional
        gt_origins:   (K, 3) GT origins          — optional

    Returns:
        metrics: dict with 'movable_recall', 'movable_precision', 'interactable_iou',
                 and (if regression args provided) 'mean_axis_err_deg', 'mean_origin_dist'
    """
    metrics = {}

    # Movable detection metrics
    if pred_movable_instances is not None and pred_movable_instances.size > 0:
        pred_movable_mask = pred_movable_instances.any(axis=0)
    else:
        pred_movable_mask = np.zeros(len(gt_movable_labels), dtype=bool)

    gt_movable_mask = gt_movable_labels > 0

    true_positives = (pred_movable_mask & gt_movable_mask).sum()
    false_positives = (pred_movable_mask & ~gt_movable_mask).sum()
    false_negatives = (~pred_movable_mask & gt_movable_mask).sum()

    movable_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    movable_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    metrics["movable_recall"] = movable_recall
    metrics["movable_precision"] = movable_precision

    # Interactable metrics (binary classification)
    pred_interactable_binary = (pred_interactable > 0.5).astype(int)
    iou_interactable = (
        (pred_interactable_binary & gt_interactable_labels).sum()
        / (pred_interactable_binary | gt_interactable_labels).sum()
    )
    metrics["interactable_iou"] = iou_interactable

    # Axis / origin regression metrics (optional)
    if pred_axes is not None and gt_axes is not None:
        reg_metrics = evaluate_axis_origin(pred_axes, pred_origins, gt_axes, gt_origins)
        metrics.update(reg_metrics)

    return metrics
