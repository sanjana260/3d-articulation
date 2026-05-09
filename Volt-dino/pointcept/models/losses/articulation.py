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


# ── Regression losses ──────────────────────────────────────────────────────────

def axis_loss(pred_axes, gt_axes):
    """Angular distance between predicted and GT axes, invariant to sign.

    L = 1 - |cos_sim(pred, gt)|

    Args:
        pred_axes: (K, 3) — unnormalised; normalised internally
        gt_axes:   (K, 3) — unit vectors (from preprocessing)

    Returns:
        scalar loss
    """
    pred_norm = F.normalize(pred_axes, dim=1, eps=1e-6)
    cos_sim = (pred_norm * gt_axes).sum(dim=1)
    return (1.0 - cos_sim.abs()).mean()


def origin_loss(pred_origins, gt_origins, gt_axes):
    """Perpendicular distance from predicted origin to the GT axis line.

    Any point on the axis line (not just gt_origin) has zero loss.
    This removes the ill-posed component along the axis direction.

    delta     = pred_origin - gt_origin
    perp(d,a) = d - (d · a) * a
    L         = ||perp||

    Args:
        pred_origins: (K, 3)
        gt_origins:   (K, 3) — a known point on the GT axis line
        gt_axes:      (K, 3) — unit vectors

    Returns:
        scalar loss
    """
    delta = pred_origins - gt_origins
    parallel = (delta * gt_axes).sum(dim=1, keepdim=True) * gt_axes
    perp = delta - parallel
    return perp.norm(dim=1).mean()


@LOSSES.register_module()
class ArticulationRegressionLoss(nn.Module):
    """Combined axis + origin regression loss for articulated parts.

    Both losses are averaged over all instances present in the batch.
    Skips gracefully when no instances are present (returns 0).
    """

    def __init__(
        self,
        axis_weight=1.0,
        origin_weight=1.0,
        loss_weight=1.0,
    ):
        super().__init__()
        self.axis_weight = axis_weight
        self.origin_weight = origin_weight
        self.loss_weight = loss_weight

    def forward(self, pred_dict, batch_dict):
        """
        Args:
            pred_dict:  must contain 'axis_pred' (K,3), 'origin_pred' (K,3),
                        'axis_gt' (K,3), 'origin_gt' (K,3)
            batch_dict: unused (GT is already extracted per-instance in pred_dict)

        Returns:
            scalar loss tensor
        """
        axis_pred   = pred_dict.get("axis_pred")
        origin_pred = pred_dict.get("origin_pred")
        axis_gt     = pred_dict.get("axis_gt")
        origin_gt   = pred_dict.get("origin_gt")

        if axis_pred is None or axis_pred.shape[0] == 0:
            # No instances in this batch — return zero loss cleanly
            device = next(iter(pred_dict.values())).device if pred_dict else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True)

        l_axis   = axis_loss(axis_pred, axis_gt)
        l_origin = origin_loss(origin_pred, origin_gt, axis_gt)
        return (self.axis_weight * l_axis + self.origin_weight * l_origin) * self.loss_weight


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
