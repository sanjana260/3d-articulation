"""
Volt with Articulation Heads for Articulate3D Dataset

Extended Volt model with:
  1. Movable part segmentation head   (3 classes: fixed, rotation, translation)
  2. Interactable part segmentation head (binary)
  3. Axis regression head             (predicts unit axis direction per instance)
  4. Origin regression head           (predicts a point on the axis line per instance)

Heads 3 & 4 operate at instance level: backbone voxel features are mean-pooled
within each ground-truth instance mask before being passed to AxisHead / OriginHead.
This pooling uses the per-vertex instance_artic_label produced by
tools/preprocess_articulate3d.py.

Author: Sanjana Mohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS


# ── Instance feature pooling ────────────────────────────────────────────────────

def pool_instance_features(features, instance_artic_label, batch, axis_label=None, origin_label=None):
    """Mean-pool voxel features per articulated instance.

    Instance IDs are local to each scene (scene 0 and scene 1 can both have
    instance ID 1).  We iterate over scenes using the batch index to avoid
    inter-scene collisions.

    Args:
        features:              (N, D) — per-voxel backbone features
        instance_artic_label:  (N,)   int64 — per-voxel instance ID; 0 = background
        batch:                 (N,)   int64 — scene index for each voxel
        axis_label:            (N, 3) float32 or None — GT axis (same for all voxels in an instance)
        origin_label:          (N, 3) float32 or None — GT origin (same for all voxels in an instance)

    Returns:
        inst_features: (K, D)   — pooled feature per instance (K = total instances)
        inst_axes:     (K, 3)   or None
        inst_origins:  (K, 3)   or None
    """
    inst_features_list = []
    inst_axes_list     = []
    inst_origins_list  = []

    num_scenes = int(batch.max().item()) + 1
    for scene_idx in range(num_scenes):
        scene_mask  = batch == scene_idx
        scene_feats = features[scene_mask]
        scene_inst  = instance_artic_label[scene_mask]

        unique_ids = scene_inst.unique()
        unique_ids = unique_ids[unique_ids > 0]  # exclude background (0)

        for inst_id in unique_ids:
            inst_mask = scene_inst == inst_id
            inst_features_list.append(scene_feats[inst_mask].mean(dim=0))

            if axis_label is not None:
                inst_axes_list.append(axis_label[scene_mask][inst_mask][0])
            if origin_label is not None:
                inst_origins_list.append(origin_label[scene_mask][inst_mask][0])

    if not inst_features_list:
        D      = features.shape[1]
        empty  = features.new_zeros(0, D)
        zeros3 = features.new_zeros(0, 3)
        return empty, (zeros3 if axis_label is not None else None), \
                      (zeros3 if origin_label is not None else None)

    inst_features = torch.stack(inst_features_list, dim=0)
    inst_axes     = torch.stack(inst_axes_list,     dim=0) if inst_axes_list     else None
    inst_origins  = torch.stack(inst_origins_list,  dim=0) if inst_origins_list  else None
    return inst_features, inst_axes, inst_origins


# ── Regression heads ────────────────────────────────────────────────────────────

class AxisHead(nn.Module):
    """Predict axis direction for each instance.

    Input:  (K, D) pooled instance features
    Output: (K, 3) unnormalised axis vectors (normalised to unit length in loss / inference)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 3),
        )

    def forward(self, x):
        return self.net(x)


class OriginHead(nn.Module):
    """Predict a point on the axis line for each instance.

    Input:  (K, D) pooled instance features
    Output: (K, 3) 3-D coordinates in the scene's coordinate frame
    """

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 3),
        )

    def forward(self, x):
        return self.net(x)


# ── Model ───────────────────────────────────────────────────────────────────────

@MODELS.register_module()
class VoltArticulate(nn.Module):
    """Volt backbone with articulation heads."""

    def __init__(
        self,
        backbone_out_channels=128,
        num_seg_classes=100,
        backbone=None,
        freeze_backbone=False,
        freeze_seg_head=False,
        use_regression=True,
    ):
        """
        Args:
            backbone_out_channels: output feature dimension from Volt's decoder
            num_seg_classes:       number of semantic segmentation classes
            backbone:              dict config for the Volt backbone
            freeze_backbone:       whether to freeze backbone weights
            freeze_seg_head:       whether to freeze the semantic seg head (transfer learning)
            use_regression:        add AxisHead + OriginHead for per-instance regression
        """
        super().__init__()
        self.backbone_out_channels = backbone_out_channels
        self.num_seg_classes       = num_seg_classes
        self.use_regression        = use_regression

        from pointcept.models.builder import build_model
        self.backbone = build_model(backbone)

        # Per-voxel classification heads
        self.seg_head = nn.Linear(backbone_out_channels, num_seg_classes)

        self.movable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 3),  # fixed / rotation / translation
        )

        self.interactable_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, 1),  # binary
        )

        # Per-instance regression heads (optional)
        if use_regression:
            self.axis_head   = AxisHead(backbone_out_channels)
            self.origin_head = OriginHead(backbone_out_channels)
        else:
            self.axis_head   = None
            self.origin_head = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_seg_head:
            for p in self.seg_head.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        """
        Returns:
            features:              (N, D)
            seg_logits:            (N, num_seg_classes)
            movable_logits:        (N, 3)
            interactable_logits:   (N, 1)
        """
        features = self.backbone(data_dict)
        return (
            features,
            self.seg_head(features),
            self.movable_head(features),
            self.interactable_head(features),
        )


@MODELS.register_module()
class ArticulateSegmentor(nn.Module):
    """Training / inference wrapper for VoltArticulate.

    Computes all losses and returns them during training.
    Returns logits and (optionally) per-instance regression outputs during eval.
    """

    def __init__(
        self,
        backbone=None,
        num_classes=100,
        backbone_out_channels=128,
        criteria=None,
        articulation_criteria=None,
        articulation_weight=0.5,
        regression_criteria=None,
        regression_weight=0.5,
        freeze_backbone=False,
        freeze_seg_head=False,
        use_regression=True,
    ):
        """
        Args:
            backbone:              Volt backbone config
            num_classes:           number of semantic segmentation classes
            backbone_out_channels: Volt decoder output channels
            criteria:              loss configs for semantic segmentation
            articulation_criteria: loss config for movable / interactable heads
            articulation_weight:   relative weight of articulation loss
            regression_criteria:   loss config for axis + origin regression
            regression_weight:     relative weight of regression loss
            freeze_backbone:       freeze backbone weights
            freeze_seg_head:       freeze semantic seg head (transfer learning)
            use_regression:        enable AxisHead + OriginHead
        """
        super().__init__()
        self.articulation_weight = articulation_weight
        self.regression_weight   = regression_weight

        self.model = VoltArticulate(
            backbone_out_channels=backbone_out_channels,
            num_seg_classes=num_classes,
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            freeze_seg_head=freeze_seg_head,
            use_regression=use_regression,
        )

        from pointcept.models.losses import build_criteria

        self.seg_criteria = build_criteria(criteria)

        if articulation_criteria is not None:
            self.articulation_criteria = build_criteria([articulation_criteria])
        else:
            self.articulation_criteria = None

        if regression_criteria is not None:
            self.regression_criteria = build_criteria([regression_criteria])
        else:
            self.regression_criteria = None

    def forward(self, input_dict):
        """
        Args:
            input_dict: batch dict containing geometry/features and (during training)
                        segment, movable_label, interactable_label,
                        instance_artic_label, axis_label, origin_label.

        Returns:
            Training: dict with 'loss' scalar.
            Eval:     dict with seg_logits, movable_logits, interactable_logits,
                      and optionally axis_pred / origin_pred.
        """
        features, seg_logits, movable_logits, interactable_logits = self.model(input_dict)
        return_dict = {}

        # ── Per-instance regression ─────────────────────────────────────────────
        # Pooling requires GT instance masks; skip when they're absent.
        regression_preds = {}
        if (
            self.model.use_regression
            and "instance_artic_label" in input_dict
        ):
            inst_feats, inst_axes_gt, inst_origins_gt = pool_instance_features(
                features,
                input_dict["instance_artic_label"],
                input_dict["batch"],
                axis_label=input_dict.get("axis_label"),
                origin_label=input_dict.get("origin_label"),
            )

            if inst_feats.shape[0] > 0:
                regression_preds["axis_pred"]   = self.model.axis_head(inst_feats)
                regression_preds["origin_pred"] = self.model.origin_head(inst_feats)
                regression_preds["axis_gt"]     = inst_axes_gt
                regression_preds["origin_gt"]   = inst_origins_gt

        # ── Training ────────────────────────────────────────────────────────────
        if self.training:
            total_loss = self.seg_criteria(seg_logits, input_dict["segment"])

            if (
                self.articulation_criteria is not None
                and "movable_label" in input_dict
                and "interactable_label" in input_dict
            ):
                pred_dict = {
                    "movable_logits":      movable_logits,
                    "interactable_logits": interactable_logits,
                }
                artic_loss = self.articulation_criteria(pred_dict, input_dict)
                if artic_loss.item() != 0.0:
                    total_loss = total_loss + self.articulation_weight * artic_loss

            if self.regression_criteria is not None and regression_preds:
                reg_loss = self.regression_criteria(regression_preds, input_dict)
                if reg_loss.item() != 0.0:
                    total_loss = total_loss + self.regression_weight * reg_loss

            return_dict["loss"] = total_loss
            return return_dict

        # ── Eval / Test ─────────────────────────────────────────────────────────
        return_dict["seg_logits"]          = seg_logits
        return_dict["movable_logits"]      = movable_logits
        return_dict["interactable_logits"] = interactable_logits

        if regression_preds:
            return_dict["axis_pred"]   = F.normalize(regression_preds["axis_pred"],   dim=1)
            return_dict["origin_pred"] = regression_preds["origin_pred"]

        if "segment" in input_dict:
            return_dict["loss"] = self.seg_criteria(seg_logits, input_dict["segment"])

        return return_dict
