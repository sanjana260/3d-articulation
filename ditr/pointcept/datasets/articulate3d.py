"""
Articulate3D Dataset for Pointcept

Loads per-point data including articulation annotations (axis, origin, type, range).
Expects preprocessed data in the format produced by tools/preprocess_articulate3d.py.
"""

import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from pointcept.utils.cache import shared_dict


@DATASETS.register_module()
class Articulate3DDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "artic_axis",
        "artic_origin",
        "artic_type",
        "artic_range",
    ]

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

        # Standard fields (same as DefaultDataset)
        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict:
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict:
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        # Articulation fields
        num_points = data_dict["coord"].shape[0]

        if "artic_axis" in data_dict:
            data_dict["artic_axis"] = data_dict["artic_axis"].astype(np.float32)
        else:
            data_dict["artic_axis"] = np.zeros((num_points, 3), dtype=np.float32)

        if "artic_origin" in data_dict:
            data_dict["artic_origin"] = data_dict["artic_origin"].astype(np.float32)
        else:
            data_dict["artic_origin"] = np.zeros((num_points, 3), dtype=np.float32)

        if "artic_type" in data_dict:
            data_dict["artic_type"] = data_dict["artic_type"].reshape([-1]).astype(np.int32)
        else:
            data_dict["artic_type"] = np.zeros(num_points, dtype=np.int32)

        if "artic_range" in data_dict:
            data_dict["artic_range"] = data_dict["artic_range"].astype(np.float32)
        else:
            data_dict["artic_range"] = np.zeros((num_points, 2), dtype=np.float32)

        return data_dict
