"""
Training config for Volt with DITR-style DINOv2 injection on ScanNet++.

What changed vs semseg-volt-base.py:
  - backbone gains  use_dino=True, dino_dim=1024
  - train/val Collect transforms add "dino_feat" to their keys list

Prerequisites:
  1. Run Phase A to extract per-image DINOv2 feature maps:
       python scripts/precompute_dino_image_features.py \\
           --scannetpp_root /data/scannetpp \\
           --output_dir     /data/dino_image_feats \\
           --model          dinov2_vitl14 \\
           --num_workers    4

  2. Run Phase B to assign per-point features:
       python scripts/precompute_dino_point_features.py \\
           --pointcept_root  data/scannetpp \\
           --scannetpp_root  /data/scannetpp \\
           --dino_image_dir  /data/dino_image_feats \\
           --num_workers     8

     This writes  data/scannetpp/{train,val}/<scene_id>/dino_feat.npy
     alongside coord.npy/color.npy/normal.npy.

  3. Launch training with this config:
       python tools/train.py configs/scannetpp/semseg-volt-dino.py

Notes:
  - dino_dim=1024 matches dinov2_vitl14. Use dino_dim=768 for dinov2_vitb14.
  - "dino_feat" is optional: if dino_feat.npy is absent the model silently
    falls back to pure-3D Volt behaviour (use_dino gate in Volt.forward).
  - Optimizer uses a single LR for all parameters. If you want the DITR paper's
    10× higher LR for the two projection layers, switch to a custom optimizer
    defined in the engine or override via --options.
"""

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# misc custom setting
batch_size = 16
num_worker = 24
mix_prob = 0.85
empty_cache = False
enable_amp = True
use_ema = True

# ── Model ──────────────────────────────────────────────────────────────────────
model = dict(
    type="DefaultSegmentorV2",
    num_classes=100,
    backbone_out_channels=128,
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
        # DITR DINOv2 injection (NEW)
        use_dino=True,
        dino_dim=1024,   # 1024 for dinov2_vitl14; 768 for dinov2_vitb14
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            label_smoothing=0.1,
            ignore_index=-1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ── Scheduler ──────────────────────────────────────────────────────────────────
epoch = 800
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset_type = "ScanNetPPDataset"
data_root = "data/scannetpp"

data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=1000000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="InstanceShift", p=0.2, shift_range=[0.1, 0.1, 0.1]),
            dict(type="InstanceRotate", p=0.2, axis="z", angle=[-0.25, 0.25]),
            dict(type="InstanceFlip", p=0.2, flip_prob=0.5),
            dict(type="InstanceScale", p=0.2, scale=[0.9, 1.1]),
            dict(type="InstanceDropOut", p=0.1, drop_ratio=0.5),
            dict(type="InstanceColorDropout", p=0.2, drop_value=0),
            dict(type="SwapInstances", p=0.2),
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
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
                # DITR: include precomputed DINOv2 features if present (NEW)
                extra_keys=("dino_feat",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
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
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("color", "normal"),
                # DITR: include precomputed DINOv2 features if present (NEW)
                extra_keys=("dino_feat",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
