"""
ScanNet++ dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import pickle
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


# ===== MODIFIED: Extended dataset with articulation labels and instance metadata =====
@DATASETS.register_module()
class ScanNetPPArticulateDataset(ScanNetPPDataset):
    """ScanNetPPDataset extended with per-point articulation labels and per-instance
    axis/origin metadata for movable and interactable part segmentation."""

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

        # has_articulation: stored as a 0-dim numpy bool so it survives ToTensor
        data_dict["has_articulation"] = np.bool_(False)

        if self.articulation_root is None:
            data_dict["movable_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["interactable_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["artic_instance_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["artic_instances"] = []
            return data_dict

        root = self.articulation_root
        movable_path = os.path.join(root, f"{scene_id}_movable_label.npy")
        interactable_path = os.path.join(root, f"{scene_id}_interactable_label.npy")
        instance_label_path = os.path.join(root, f"{scene_id}_artic_instance_label.npy")
        instances_pkl_path = os.path.join(root, f"{scene_id}_artic_instances.pkl")

        if os.path.exists(movable_path) and os.path.exists(interactable_path):
            data_dict["movable_label"] = np.load(movable_path).astype(np.int64)
            data_dict["interactable_label"] = np.load(interactable_path).astype(np.int64)
            data_dict["has_articulation"] = np.bool_(True)
        else:
            data_dict["movable_label"] = np.zeros(num_points, dtype=np.int64)
            data_dict["interactable_label"] = np.zeros(num_points, dtype=np.int64)

        # Per-point instance ID — flows through GridSample voxelization like any other label
        if os.path.exists(instance_label_path):
            data_dict["artic_instance_label"] = np.load(instance_label_path).astype(np.int64)
        else:
            data_dict["artic_instance_label"] = np.zeros(num_points, dtype=np.int64)

        # Per-instance metadata (axis/origin): variable-length list of dicts.
        # NOT passed through transforms — re-attached in prepare_train_data.
        if os.path.exists(instances_pkl_path):
            with open(instances_pkl_path, "rb") as f:
                data_dict["artic_instances"] = pickle.load(f)
        else:
            data_dict["artic_instances"] = []

        return data_dict

    def prepare_train_data(self, idx):
        # Save artic_instances before the transform pipeline drops it (it's not a tensor)
        data_dict = self.get_data(idx)
        artic_instances = data_dict.get("artic_instances", [])
        data_dict = self.transform(data_dict)
        # Re-attach after Collect has filtered the dict
        data_dict["artic_instances"] = artic_instances
        return data_dict
