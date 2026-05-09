"""
Volt with Articulation Heads for Articulate3D Dataset

Four output heads:
1. Semantic segmentation   (N, num_seg_classes)  — existing Volt task
2. Movable part            (N, 3)                — {bg, rotation, translation}
3. Interactable part       (N, 1)                — binary {handle, knob, switch}
4. Axis direction          (N_inst, 3)            — unit vector, instance-level
5. Axis origin             (N_inst, 3)            — point on axis, instance-level

Author: Sanjana Mohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS


# ===== ADDED: Instance feature pooling =====
def pool_instance_features(voxel_features, voxel_coords, instance_label, max_instance_id):
    """Mean-pool voxel features for each articulated instance.

    Uses artic_instance_label to group voxels — no need for separate mask tensors.

    Args:
        voxel_features:  (M, D) — per-voxel features from decoder
        voxel_coords:    (M, 3) — world-space coordinates
        instance_label:  (M,)   — int64, 0=background, N=Nth instance
        max_instance_id: int    — number of instances in this scene

    Returns:
        inst_features:  (N_inst, D)
        inst_centroids: (N_inst, 3)
        valid:          (N_inst,) bool — False for instances absent after voxelization
    """
    D = voxel_features.shape[1]
    inst_feats, centroids, valid = [], [], []

    for inst_id in range(1, max_instance_id + 1):
        mask = instance_label == inst_id
        if mask.sum() == 0:
            inst_feats.append(torch.zeros(D, device=voxel_features.device))
            centroids.append(torch.zeros(3, device=voxel_features.device))
            valid.append(False)
        else:
            inst_feats.append(voxel_features[mask].mean(0))
            centroids.append(voxel_coords[mask].mean(0))
            valid.append(True)

    return (
        torch.stack(inst_feats),
        torch.stack(centroids),
        torch.tensor(valid, dtype=torch.bool, device=voxel_features.device),
    )


# ===== ADDED: Axis direction head =====
class AxisHead(nn.Module):
    """Predicts a unit-vector axis direction per instance.

    Uses LayerNorm instead of BatchNorm — instance batches are tiny (1-5 per scene).
    Output is L2-normalised so the loss can use cosine similarity directly.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 3),
        )

    def forward(self, inst_feat):
        # inst_feat: (N_inst, D)
        raw = self.mlp(inst_feat)           # (N_inst, 3)
        return F.normalize(raw, dim=-1)     # unit vectors


# ===== ADDED: Axis origin head =====
class OriginHead(nn.Module):
    """Predicts a point on the motion axis per instance.

    Conditioned on the instance centroid: network predicts an offset,
    which is more stable than predicting absolute world coordinates.
    """

    def __init__(self, feat_dim):
        super().__init__()
        # +3 for centroid conditioning
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 3),
        )

    def forward(self, inst_feat, inst_centroid):
        # inst_feat:     (N_inst, D)
        # inst_centroid: (N_inst, 3)
        inp = torch.cat([inst_feat, inst_centroid], dim=-1)  # (N_inst, D+3)
        offset = self.mlp(inp)                               # (N_inst, 3)
        return inst_centroid + offset                        # absolute origin


@MODELS.register_module()
class VoltArticulate(nn.Module):
    """Volt backbone with segmentation and articulation regression heads."""

    def __init__(
        self,
        backbone_out_channels=128,
        num_seg_classes=100,
        backbone=None,
        freeze_backbone=False,
        freeze_seg_head=False,
    ):
        super().__init__()
        self.backbone_out_channels = backbone_out_channels

        from pointcept.models.builder import build_model
        self.backbone = build_model(backbone)

        # Semantic segmentation head (existing task)
        self.seg_head = nn.Linear(backbone_out_channels, num_seg_classes)

        # Per-point classification heads
        self.movable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 3),
        )
        self.interactable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 1),
        )

        # MODIFIED: Instance-level regression heads
        self.axis_head = AxisHead(feat_dim=backbone_out_channels)
        self.origin_head = OriginHead(feat_dim=backbone_out_channels)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if freeze_seg_head:
            for p in self.seg_head.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        features = self.backbone(data_dict)  # (M, D)

        seg_logits = self.seg_head(features)
        movable_logits = self.movable_head(features)
        interactable_logits = self.interactable_head(features)

        # Regression heads are run by ArticulateSegmentor (needs coord + instance labels)
        return features, seg_logits, movable_logits, interactable_logits


@MODELS.register_module()
class ArticulateSegmentor(nn.Module):
    """Training and inference wrapper for VoltArticulate.

    Handles multi-task loss computation and exposes axis/origin predictions.
    """

    def __init__(
        self,
        backbone=None,
        num_classes=100,
        backbone_out_channels=128,
        criteria=None,
        articulation_criteria=None,
        articulation_weight=0.5,
        # MODIFIED: Regression loss weights
        lambda_axis=1.0,
        lambda_origin=0.5,
        freeze_backbone=False,
        freeze_seg_head=False,
    ):
        super().__init__()
        self.articulation_weight = articulation_weight
        self.lambda_axis = lambda_axis
        self.lambda_origin = lambda_origin

        self.model = VoltArticulate(
            backbone_out_channels=backbone_out_channels,
            num_seg_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            freeze_seg_head=freeze_seg_head,
        )

        from pointcept.models.losses import build_criteria
        from pointcept.models.losses.articulation import artic_regression_loss

        self.seg_criteria = build_criteria(criteria)
        self.articulation_criteria = (
            build_criteria([articulation_criteria])
            if articulation_criteria is not None
            else None
        )
        # Keep a reference so forward can call it directly
        self._artic_regression_loss = artic_regression_loss

    def forward(self, input_dict):
        features, seg_logits, movable_logits, interactable_logits = self.model(
            input_dict
        )

        return_dict = {
            "seg_logits": seg_logits,
            "movable_logits": movable_logits,
            "interactable_logits": interactable_logits,
        }

        if self.training:
            total_loss = self.seg_criteria(seg_logits, input_dict["segment"])

            # Per-point segmentation loss (movable + interactable)
            if (
                self.articulation_criteria is not None
                and "movable_label" in input_dict
            ):
                pred_dict = {
                    "movable_logits": movable_logits,
                    "interactable_logits": interactable_logits,
                }
                artic_seg_loss = self.articulation_criteria(pred_dict, input_dict)
                if isinstance(artic_seg_loss, torch.Tensor) and artic_seg_loss.item() > 0:
                    total_loss = total_loss + self.articulation_weight * artic_seg_loss

            # MODIFIED: Instance-level regression loss (axis + origin)
            if (
                input_dict.get("has_articulation", False)
                and "artic_instance_label" in input_dict
                and "artic_instances" in input_dict
                and len(input_dict["artic_instances"]) > 0
            ):
                reg_loss = self._run_regression(
                    features, input_dict, return_preds=False
                )
                if reg_loss is not None:
                    total_loss = total_loss + reg_loss

            return_dict["loss"] = total_loss

        else:
            if "segment" in input_dict:
                return_dict["loss"] = self.seg_criteria(
                    seg_logits, input_dict["segment"]
                )

            # MODIFIED: Run regression at inference too, return axis/origin
            if (
                "artic_instance_label" in input_dict
                and "artic_instances" in input_dict
            ):
                axis_preds, origin_preds = self._run_regression(
                    features, input_dict, return_preds=True
                )
                return_dict["axis_preds"] = axis_preds
                return_dict["origin_preds"] = origin_preds

        return return_dict

    def _run_regression(self, features, input_dict, return_preds):
        """Pool instance features and compute axis/origin predictions.

        At training time (return_preds=False): compute and return the regression loss.
        At inference time  (return_preds=True): return (axis_preds, origin_preds) lists.
        """
        instance_label = input_dict["artic_instance_label"]  # (M,)
        artic_instances = input_dict["artic_instances"]       # list of dicts
        coord = input_dict["coord"].float()                   # (M, 3)

        # artic_instances is a list-of-lists when batched; flatten if needed
        if artic_instances and isinstance(artic_instances[0], list):
            artic_instances = [inst for scene in artic_instances for inst in scene]

        max_instance_id = int(instance_label.max().item())
        if max_instance_id == 0 or len(artic_instances) == 0:
            return (None, None) if return_preds else None

        inst_feats, inst_centroids, valid = pool_instance_features(
            features, coord, instance_label, max_instance_id
        )

        if not valid.any():
            return (None, None) if return_preds else None

        axis_preds = self.model.axis_head(inst_feats[valid])      # (N_valid, 3)
        origin_preds = self.model.origin_head(                    # (N_valid, 3)
            inst_feats[valid], inst_centroids[valid]
        )

        if return_preds:
            return axis_preds, origin_preds

        # Compute regression loss against GT
        loss = self._artic_regression_loss(
            axis_preds,
            origin_preds,
            artic_instances,
            valid,
            lambda_axis=self.lambda_axis,
            lambda_origin=self.lambda_origin,
        )
        return loss
