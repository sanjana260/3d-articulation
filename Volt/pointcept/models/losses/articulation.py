"""
Articulation Losses for Articulate3D Dataset

Author: Sanjana Mohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


# ===== ADDED: Dice loss for binary/multiclass segmentation =====
def dice_loss(pred_logits, target, eps=1e-6, reduction="mean"):
    """Compute Dice loss for binary or multiclass segmentation.

    Args:
        pred_logits: (N,) or (N, num_classes) logits
        target: (N,) binary labels
        eps: numerical stability
        reduction: 'mean' or 'none'
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = 1 - (2 * intersection + eps) / (union + eps)

    if reduction == "mean":
        return dice.mean()
    else:
        return dice


@LOSSES.register_module()
class ArticulationLoss(nn.Module):
    """Combined loss for movable and interactable segmentation.

    Follows the USDNet approach: Lseg = λ_dice * L_dice + λ_ce * L_ce
    """

    def __init__(
        self,
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(ArticulationLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred_dict, batch_dict):
        """
        Args:
            pred_dict: dict with keys 'movable_logits', 'interactable_logits'
            batch_dict: dict with keys 'movable_label', 'interactable_label', 'has_articulation'

        Returns:
            loss: scalar tensor
        """
        movable_logits = pred_dict.get("movable_logits", None)
        interactable_logits = pred_dict.get("interactable_logits", None)
        has_articulation = batch_dict.get("has_articulation", None)

        if movable_logits is None or interactable_logits is None:
            return torch.tensor(0.0, device=movable_logits.device if movable_logits is not None else None)

        # Only compute loss on scenes with articulation annotations
        if has_articulation is not None and not has_articulation.any():
            return torch.tensor(0.0, device=movable_logits.device)

        if has_articulation is not None:
            movable_logits = movable_logits[has_articulation]
            interactable_logits = interactable_logits[has_articulation]
            movable_labels = batch_dict["movable_label"][has_articulation]
            interactable_labels = batch_dict["interactable_label"][has_articulation]
        else:
            movable_labels = batch_dict["movable_label"]
            interactable_labels = batch_dict["interactable_label"]

        if movable_logits.shape[0] == 0:
            return torch.tensor(0.0, device=movable_logits.device)

        total_loss = 0.0

        # MODIFIED: Movable segmentation loss (3 classes: background, rotation, translation)
        # Convert to binary for loss computation (movable vs non-movable)
        movable_binary = (movable_labels > 0).float()

        # Use the 'movable' logits (typically index 1 in 3-class output)
        if movable_logits.shape[-1] == 3:
            # If 3-class output, use the combined probability of rotation/translation
            movable_pred = torch.logsumexp(movable_logits[:, 1:], dim=1)
        else:
            # If binary output
            movable_pred = movable_logits.squeeze(-1)

        bce_loss_mov = F.binary_cross_entropy_with_logits(
            movable_pred, movable_binary, reduction="mean"
        )
        dice_loss_mov = dice_loss(movable_pred, movable_binary, reduction="mean")
        total_loss += self.lambda_ce * bce_loss_mov + self.lambda_dice * dice_loss_mov

        # MODIFIED: Interactable segmentation loss (binary)
        interactable_labels_f = interactable_labels.float()
        bce_loss_int = F.binary_cross_entropy_with_logits(
            interactable_logits.squeeze(-1), interactable_labels_f, reduction="mean"
        )
        dice_loss_int = dice_loss(
            interactable_logits.squeeze(-1), interactable_labels_f, reduction="mean"
        )
        total_loss += self.lambda_ce * bce_loss_int + self.lambda_dice * dice_loss_int

        return total_loss * self.loss_weight


# ===== ADDED: Instance-level regression losses =====

def axis_loss(pred_axes, gt_axes):
    """Cosine loss with ±axis ambiguity handling.

    A rotation axis of [0,1,0] is identical to [0,-1,0], so we take
    the absolute value of cosine similarity before computing the loss.

    Args:
        pred_axes: (N, 3) unit vectors (already normalised by AxisHead)
        gt_axes:   (N, 3) unit vectors
    """
    cos_sim = F.cosine_similarity(pred_axes, gt_axes, dim=-1)  # (N,)
    return (1.0 - cos_sim.abs()).mean()


def origin_loss(pred_origins, gt_origins, gt_axes):
    """Point-to-line distance loss.

    Any point on the axis line is an equally valid GT origin, so
    point-to-point distance would penalise correct predictions that
    lie on the axis but far from the stored GT point.
    Point-to-line distance measures the perpendicular offset only.

    Distance from point P to line (Q, d):
        perp = (P - Q) - ((P - Q) · d) * d

    Args:
        pred_origins: (N, 3)
        gt_origins:   (N, 3)
        gt_axes:      (N, 3) unit vectors
    """
    diff = pred_origins - gt_origins                              # (N, 3)
    proj = (diff * gt_axes).sum(-1, keepdim=True) * gt_axes      # (N, 3)
    perp = diff - proj                                           # (N, 3)
    return perp.norm(dim=-1).mean()


def artic_regression_loss(
    pred_axes,
    pred_origins,
    artic_instances,
    valid,
    lambda_axis=1.0,
    lambda_origin=0.5,
):
    """Compute combined axis + origin regression loss.

    Args:
        pred_axes:       (N_valid, 3) predicted unit axes
        pred_origins:    (N_valid, 3) predicted origin points
        artic_instances: list of instance dicts (motion_type, axis, origin, ...)
        valid:           (N_inst,) bool tensor — which instances survived voxelization
        lambda_axis:     weight for axis loss
        lambda_origin:   weight for origin loss
    """
    if pred_axes is None or len(pred_axes) == 0:
        return torch.tensor(0.0, device=pred_axes.device if pred_axes is not None else None)

    device = pred_axes.device

    gt_axes = torch.stack(
        [torch.tensor(inst["axis"], dtype=torch.float32) for inst in artic_instances]
    ).to(device)[valid]  # (N_valid, 3)

    gt_origins = torch.stack(
        [torch.tensor(inst["origin"], dtype=torch.float32) for inst in artic_instances]
    ).to(device)[valid]  # (N_valid, 3)

    L_axis = axis_loss(pred_axes, gt_axes)
    L_origin = origin_loss(pred_origins, gt_origins, gt_axes)

    return lambda_axis * L_axis + lambda_origin * L_origin


@LOSSES.register_module()
class BinaryCrossEntropyWithDiceLoss(nn.Module):
    """Binary cross-entropy + Dice loss for part segmentation."""

    def __init__(
        self,
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(BinaryCrossEntropyWithDiceLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (N,) logits
            target: (N,) binary labels
        """
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="mean")
        dice = dice_loss(pred, target.float(), reduction="mean")
        loss = self.lambda_ce * bce + self.lambda_dice * dice
        return loss * self.loss_weight
