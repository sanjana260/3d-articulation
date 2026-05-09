"""
Transfer Learning: Train Only Articulation Heads

Configuration for training ONLY the articulation heads (movable + interactable)
while keeping the Volt backbone and semantic segmentation head frozen.

Best for: Limited Articulate3D data, need fast training

Usage:
    python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer-heads-only.py \
        --options weight=/path/to/pretrained_volt.pth

Training time: ~1-2 days on 4 V100 GPUs
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# MODIFIED: Load pretrained Volt weights
weight = "pretrained_volt.pth"  # CHANGE THIS to your checkpoint path

# Training hyperparameters for head-only training
batch_size = 32  # Can use larger batch (backbone frozen)
num_worker = 24
enable_amp = True
use_ema = False  # Not needed for transfer learning
epoch = 150  # Much shorter training
eval_epoch = 15

# MODIFIED: Model with frozen backbone and seg head
model = dict(
    type="ArticulateSegmentor",
    num_classes=100,
    backbone_out_channels=128,
    # MODIFIED: Freeze backbone (no gradient updates)
    freeze_backbone=True,
    # MODIFIED: Freeze semantic seg head (only train articulation heads)
    freeze_seg_head=True,
    backbone=dict(
        type="Volt",
        in_channels=6,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=128,
    ),
    # Lightweight criteria (backbone is frozen, only for consistency)
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=0.0,  # Don't compute (backbone frozen)
            label_smoothing=0.1,
            ignore_index=-1,
        ),
    ],
    # MODIFIED: Articulation loss is the main objective
    articulation_criteria=dict(
        type="ArticulationLoss",
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
    ),
    # MODIFIED: Full weight on articulation (1.0 instead of 0.5)
    articulation_weight=1.0,
)

# MODIFIED: Higher learning rate for head-only training
# Heads have fewer parameters, can use higher LR
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=5.0,
    final_div_factor=10.0,
)

# Dataset settings
dataset_type = "ScanNetPPArticulateDataset"
data_root = "data/scannetpp"
articulation_root = "data/articulate3d_labels"

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        articulation_root=articulation_root,
        # MODIFIED: Simplified augmentation for transfer learning
        transform=[
            dict(type="SphereCrop", point_max=1000000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            # Reduce augmentation (backbone features are fixed)
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.003, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "movable_label", "interactable_label", "artic_instance_label", "has_articulation"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        articulation_root=articulation_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "movable_label", "interactable_label", "artic_instance_label", "has_articulation"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)
