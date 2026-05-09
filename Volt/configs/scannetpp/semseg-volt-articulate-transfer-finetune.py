"""
Transfer Learning: Fine-tune with Frozen Backbone

Configuration for fine-tuning the semantic segmentation head and
articulation heads while keeping the Volt backbone frozen.

Best for: Moderate Articulate3D data, balance between quality and speed

Usage:
    python tools/train.py configs/scannetpp/semseg-volt-articulate-transfer-finetune.py \
        --options weight=/path/to/pretrained_volt.pth

Training time: ~3-5 days on 4 V100 GPUs
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# MODIFIED: Load pretrained Volt weights
resume = False # To restart training in case of failure
weight = "/scratch/sm13117/my_volt/Volt/models/model_best.pth"  # CHANGE THIS to your checkpoint path

# Training hyperparameters for fine-tuning
batch_size = 16
num_worker = 24
enable_amp = True
use_ema = True  # Can use EMA for fine-tuning
epoch = 150  # Intermediate duration
eval_epoch = 30

# MODIFIED: Model with frozen backbone, trainable heads
model = dict(
    type="ArticulateSegmentor",
    num_classes=100,
    backbone_out_channels=128,
    # MODIFIED: Freeze backbone (encoder decoder frozen)
    freeze_backbone=True,
    # MODIFIED: Allow semantic seg head to train
    freeze_seg_head=False,
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
    # Semantic segmentation loss (head is trainable)
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # Articulation loss
    articulation_criteria=dict(
        type="ArticulationLoss",
        lambda_dice=1.0,
        lambda_ce=1.0,
        loss_weight=1.0,
    ),
    # Balanced weight for both tasks
    articulation_weight=0.5,
    # MODIFIED: Regression loss weights (axis harder to learn; down-weight origin initially)
    lambda_axis=1.0,
    lambda_origin=0.5,
)

# MODIFIED: Lower learning rate for fine-tuning (prevent catastrophic forgetting)
# General rule: LR for fine-tuning ≈ 0.2x of full training LR
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
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
        # MODIFIED: Moderate augmentation for fine-tuning
        transform=[
            dict(type="SphereCrop", point_max=1000000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.1
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.3),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.3),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.004, clip=0.015),
            dict(type="ChromaticAutoContrast", p=0.1, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.9, ratio=0.03),
            dict(type="ChromaticJitter", p=0.9, std=0.03),
            dict(type="InstanceShift", p=0.1, shift_range=[0.05, 0.05, 0.05]),
            dict(type="InstanceRotate", p=0.1, axis="z", angle=[-0.15, 0.15]),
            dict(type="InstanceFlip", p=0.1, flip_prob=0.5),
            dict(type="InstanceScale", p=0.1, scale=[0.95, 1.05]),
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
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
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
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "movable_label", "interactable_label", "artic_instance_label", "has_articulation"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)
