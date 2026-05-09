"""
ScanNet++ dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ScanNetPPDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "superpoint",
        "segment",
        "instance",
        "dino_feat",  # DITR: per-point DINOv2 features, precomputed offline (optional)
    ]

    def __init__(
        self,
        multilabel=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # DITR: stored as float16 to save disk; cast to float32 for model compatibility
        if "dino_feat" in data_dict.keys():
            data_dict["dino_feat"] = data_dict["dino_feat"].astype(np.float32)

        if "superpoint" in data_dict.keys():
            data_dict["superpoint"] = data_dict["superpoint"].astype(np.int32)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError
        return data_dict


# ===== MODIFIED: Extended dataset with articulation labels =====
@DATASETS.register_module()
class ScanNetPPArticulateDataset(ScanNetPPDataset):
    """Extended ScanNetPPDataset with articulation labels (movable, interactable)."""

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "superpoint",
        "segment",
        "instance",
        "movable_label",        # binary articulation: 0=fixed, 1=rotation, 2=translation
        "interactable_label",   # binary articulation: 0=not interactable, 1=interactable
    ]

    def __init__(
        self,
        multilabel=False,
        articulation_root=None,
        **kwargs,
    ):
        super().__init__(multilabel=multilabel, **kwargs)
        self.articulation_root = articulation_root

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        num_points = data_dict["coord"].shape[0]
        scene_id = data_dict["name"]

        data_dict["has_articulation"] = False
        if self.articulation_root is not None:
            movable_path = os.path.join(
                self.articulation_root, f"{scene_id}_movable_label.npy"
            )
            interactable_path = os.path.join(
                self.articulation_root, f"{scene_id}_interactable_label.npy"
            )
            instance_path = os.path.join(
                self.articulation_root, f"{scene_id}_instance_artic_label.npy"
            )
            axis_path = os.path.join(
                self.articulation_root, f"{scene_id}_axis_label.npy"
            )
            origin_path = os.path.join(
                self.articulation_root, f"{scene_id}_origin_label.npy"
            )

            if os.path.exists(movable_path) and os.path.exists(interactable_path):
                data_dict["movable_label"] = np.load(movable_path).astype(np.int64)
                data_dict["interactable_label"] = np.load(interactable_path).astype(np.int64)
                data_dict["has_articulation"] = True
            else:
                data_dict["movable_label"] = np.zeros(num_points, dtype=np.int64)
                data_dict["interactable_label"] = np.zeros(num_points, dtype=np.int64)

            # Regression targets: per-vertex instance ID, axis direction, axis origin.
            # All three are optional — scenes without axis/origin annotations get zeros.
            if os.path.exists(instance_path):
                data_dict["instance_artic_label"] = np.load(instance_path).astype(np.int64)
                data_dict["axis_label"] = np.load(axis_path).astype(np.float32)
                data_dict["origin_label"] = np.load(origin_path).astype(np.float32)
            else:
                data_dict["instance_artic_label"] = np.zeros(num_points, dtype=np.int64)
                data_dict["axis_label"] = np.zeros((num_points, 3), dtype=np.float32)
                data_dict["origin_label"] = np.zeros((num_points, 3), dtype=np.float32)
        else:
            data_dict["movable_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["interactable_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["instance_artic_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["axis_label"] = np.zeros((num_points, 3), dtype=np.float32)
            data_dict["origin_label"] = np.zeros((num_points, 3), dtype=np.float32)

        return data_dict
