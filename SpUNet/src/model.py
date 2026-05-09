"""
ArticulateSPUNet — SPUNet backbone + DINOv2 fusion + Articulation heads
========================================================================

Fixes vs broken version:
  1. NUM_PARTS = 500  (dataset part IDs go up to ~456, NOT 80)
  2. Seg loss = CrossEntropy(ignore_index=0) + Dice over foreground only
     - ignore_index=0 is CORRECT: class 0 = scene background (16k pts),
       not a real part. USDNet used Dice+BCE on instance crops where class 0
       IS a part — different setup. Dice is kept but foreground-only.
  3. LR = 5e-4, no DDP scaling (2e-3 caused geometry head divergence)
  4. w_axis = 2.0 (axis head was starved of gradient)
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

POINTCEPT_PATH = "/scratch/ky2751/articulate3d_project/Pointcept"

# Dataset part IDs go up to ~456. 500 gives safe headroom.
# Class 0 = background/unknown — always ignored in seg loss.
NUM_PARTS = 500


# ─────────────────────────────────────────────────────────────────────────────
# SPUNet backbone
# ─────────────────────────────────────────────────────────────────────────────

def build_spunet_backbone(in_channels=6, base_channels=32):
    try:
        if POINTCEPT_PATH not in sys.path:
            sys.path.insert(0, POINTCEPT_PATH)
        from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase
        backbone = SpUNetBase(
            in_channels=in_channels,
            num_classes=128,
            base_channels=base_channels,
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
        )
        print("[SPUNet] Loaded from Pointcept ✓")
        return backbone, 128
    except Exception as e:
        print(f"[SPUNet] Pointcept import failed ({e}), using MLP placeholder")
        return _MLPPlaceholder(in_channels, 128), 128


class _MLPPlaceholder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 64), nn.ReLU(),
            nn.Linear(64, 128),   nn.ReLU(),
            nn.Linear(128, out_ch),
        )
    def forward(self, data_dict):
        return self.net(data_dict["feat"])


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 image encoder (frozen)
# ─────────────────────────────────────────────────────────────────────────────

class DINOv3Encoder(nn.Module):
    DINO_DIM = 768

    def __init__(self, out_dim=128, model_name="facebook/dinov2-base"):
        super().__init__()
        self.out_dim = out_dim
        try:
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained(model_name)
            for p in self.dino.parameters():
                p.requires_grad = False
            self.loaded = True
            print(f"[DINOv3] Loaded {model_name} ✓  (frozen)")
        except Exception as e:
            print(f"[DINOv3] Load failed ({e}). Running without image features.")
            self.dino   = None
            self.loaded = False

        self.proj = nn.Sequential(
            nn.Linear(self.DINO_DIM, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, images):
        if not self.loaded or images is None:
            return None
        with torch.no_grad():
            out = self.dino(pixel_values=images)
            patch_feat = out.last_hidden_state[:, 1:, :]
        return self.proj(patch_feat)


# ─────────────────────────────────────────────────────────────────────────────
# 2D → 3D feature fusion
# ─────────────────────────────────────────────────────────────────────────────

class ImageTo3DFusion(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, point_feat, img_feat_3d):
        if img_feat_3d is None:
            return point_feat
        combined = torch.cat([point_feat, img_feat_3d], dim=-1)
        return point_feat + self.gate(combined)


def project_image_features_to_points(coords, img_patches, patch_size=16,
                                      img_h=518, img_w=518,
                                      intrinsics=None, extrinsics=None):
    N, C   = coords.shape[0], img_patches.shape[-1]
    device = coords.device

    if intrinsics is None or extrinsics is None:
        return torch.zeros(N, C, device=device)

    K       = intrinsics[0]
    E       = extrinsics[0]
    patches = img_patches[0]

    ones   = torch.ones(N, 1, device=device)
    pts_h  = torch.cat([coords, ones], dim=-1)
    pts_c  = (E[:3, :] @ pts_h.T).T
    pts_2d = (K @ pts_c.T).T
    z      = pts_c[:, 2].clamp(min=1e-6)
    u      = pts_2d[:, 0] / z
    v      = pts_2d[:, 1] / z

    ph = img_h // patch_size
    pw = img_w // patch_size
    pi = (v / patch_size).long().clamp(0, ph - 1)
    pj = (u / patch_size).long().clamp(0, pw - 1)
    patch_idx = pi * pw + pj

    valid    = (z > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    feat_out = torch.zeros(N, C, device=device)
    feat_out[valid] = patches[patch_idx[valid]]
    return feat_out


# ─────────────────────────────────────────────────────────────────────────────
# Prediction heads
# ─────────────────────────────────────────────────────────────────────────────

class ArticulationHeads(nn.Module):
    def __init__(self, in_channels=128, num_parts=NUM_PARTS):
        super().__init__()
        C = in_channels
        self.seg_head = nn.Sequential(
            nn.Linear(C, C), nn.BatchNorm1d(C), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(C, num_parts),
        )
        self.mob_head = nn.Sequential(
            nn.Linear(C, C), nn.BatchNorm1d(C), nn.ReLU(),
            nn.Linear(C, 3),
        )
        self.axis_head = nn.Sequential(
            nn.Linear(C, C), nn.BatchNorm1d(C), nn.ReLU(),
            nn.Linear(C, 3),
        )
        self.origin_head = nn.Sequential(
            nn.Linear(C, C), nn.BatchNorm1d(C), nn.ReLU(),
            nn.Linear(C, 3),
        )
        self.range_head = nn.Sequential(
            nn.Linear(C, C), nn.BatchNorm1d(C), nn.ReLU(),
            nn.Linear(C, 2),
        )

    def forward(self, feat):
        seg    = self.seg_head(feat)
        mob    = self.mob_head(feat)
        axis   = F.normalize(self.axis_head(feat), dim=-1)
        origin = self.origin_head(feat)
        rng    = self.range_head(feat)
        return seg, mob, axis, origin, rng


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class ArticulateSPUNet(nn.Module):
    def __init__(self, in_channels=6, base_channels=32,
                 num_parts=NUM_PARTS, use_dino=True):
        super().__init__()
        self.use_dino = use_dino
        self.backbone, feat_dim = build_spunet_backbone(in_channels, base_channels)
        feat_dim = 128
        if use_dino:
            self.dino    = DINOv3Encoder(out_dim=feat_dim)
            self.fusion  = ImageTo3DFusion(feat_dim)
            self.use_dino = self.dino.loaded
        self.heads = ArticulationHeads(feat_dim, num_parts)

    def forward(self, data_dict):
        if "grid_coord" not in data_dict:
            data_dict["grid_coord"] = (data_dict["coord"] / 0.05).floor().int()

        feat = self.backbone(data_dict)

        if self.use_dino and "images" in data_dict and data_dict["images"] is not None:
            img_patches = self.dino(data_dict["images"])
            if img_patches is not None:
                img_feat_3d = project_image_features_to_points(
                    coords      = data_dict["coord"],
                    img_patches = img_patches,
                    intrinsics  = data_dict.get("intrinsics"),
                    extrinsics  = data_dict.get("extrinsics"),
                )
                feat = self.fusion(feat, img_feat_3d)

        seg, mob, axis, origin, rng = self.heads(feat)
        return {
            "seg_logits":  seg,
            "mob_logits":  mob,
            "axis_pred":   axis,
            "origin_pred": origin,
            "range_pred":  rng,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss_foreground(seg_probs, part_gt_safe, num_parts):
    """
    Dice loss over foreground classes only (part_id > 0).
    Keeps USDNet-style dense part supervision without background contamination.

    seg_probs:     (N, num_parts) softmax probabilities
    part_gt_safe:  (N,) integer labels clamped to [0, num_parts-1]
    """
    fg_mask = part_gt_safe > 0
    if fg_mask.sum() == 0:
        return seg_probs.new_tensor(0.0)

    probs_fg = seg_probs[fg_mask]               # (Nfg, num_parts)
    gt_fg    = part_gt_safe[fg_mask]            # (Nfg,)

    oh = F.one_hot(gt_fg, num_parts).float()    # (Nfg, num_parts)
    # drop column 0 (background) from both sides
    oh = oh[:, 1:]
    pr = probs_fg[:, 1:]

    intersection = (pr * oh).sum(dim=0)
    dice = (2.0 * intersection + 1.0) / (
        pr.sum(dim=0) + oh.sum(dim=0) + 1.0
    )
    return (1.0 - dice).mean()


class ArticulationLoss(nn.Module):
    def __init__(self, num_parts=NUM_PARTS,
                 w_seg=1.0, w_mob=1.0, w_axis=2.0, w_origin=1.0, w_range=0.5,
                 lambda_ce=1.0, lambda_dice=0.5):
        super().__init__()
        self.num_parts   = num_parts
        self.w           = dict(seg=w_seg, mob=w_mob, axis=w_axis,
                                origin=w_origin, range=w_range)
        self.lambda_ce   = lambda_ce
        self.lambda_dice = lambda_dice

    def forward(self, preds, batch):
        part_gt = batch["part_id"].long()
        mob_gt  = batch["mobility"].long()
        ax_gt   = batch["axis"].float()
        ori_gt  = batch["origin"].float()
        rmin_gt = batch["range_min"].float()
        rmax_gt = batch["range_max"].float()

        seg_logits   = preds["seg_logits"]
        part_gt_safe = part_gt.clamp(0, self.num_parts - 1)

        # CE with background ignored — this is the key fix
        L_ce = F.cross_entropy(
            seg_logits, part_gt_safe,
            ignore_index=0,
            label_smoothing=0.05,
        )

        # Dice over foreground only — keeps USDNet style, corrected
        seg_probs = torch.softmax(seg_logits, dim=-1)
        L_dice    = dice_loss_foreground(seg_probs, part_gt_safe, self.num_parts)

        L_seg = self.lambda_ce * L_ce + self.lambda_dice * L_dice

        # mobility
        L_mob = F.cross_entropy(preds["mob_logits"], mob_gt)

        # articulation — mobile points only
        mobile = mob_gt > 0
        if mobile.sum() > 0:
            ax_pred  = preds["axis_pred"][mobile]
            cos_sim  = (ax_pred * ax_gt[mobile]).sum(-1).abs().clamp(0, 1)
            L_axis   = (1.0 - cos_sim).mean()
            L_origin = F.smooth_l1_loss(preds["origin_pred"][mobile], ori_gt[mobile])
            rng_tgt  = torch.stack([rmin_gt[mobile], rmax_gt[mobile]], dim=-1)
            L_range  = F.smooth_l1_loss(preds["range_pred"][mobile], rng_tgt)
        else:
            L_axis = L_origin = L_range = seg_logits.new_tensor(0.0)

        total = (self.w["seg"]    * L_seg
               + self.w["mob"]    * L_mob
               + self.w["axis"]   * L_axis
               + self.w["origin"] * L_origin
               + self.w["range"]  * L_range)

        return total, {
            "L_seg":    L_seg.item(),
            "L_ce":     L_ce.item(),
            "L_dice":   L_dice.item(),
            "L_mob":    L_mob.item(),
            "L_axis":   L_axis.item() if mobile.sum() > 0 else 0.0,
            "L_origin": L_origin.item() if mobile.sum() > 0 else 0.0,
            "L_range":  L_range.item() if mobile.sum() > 0 else 0.0,
        }

