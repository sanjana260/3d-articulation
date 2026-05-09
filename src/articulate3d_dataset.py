import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

MOBILITY_CLASS = {"static": 0, "rotation": 1, "translation": 2}
GROUP_TO_SEMANTIC = {
    "others":           0,
    "doors/windows":    1,
    "handles movable":  2,
    "drawers":          3,
    "lids":             4,
}



def build_part_id_remap(data_root, anno_root, split_ids):
    """
    Scan all scenes in split_ids, collect all unique foreground part IDs,
    return a dict mapping raw_id → compact_id (1-indexed, 0=background).
    """
    all_ids = set()
    for scene_id in split_ids:
        scene_dir = os.path.join(data_root, scene_id)
        anno_path = os.path.join(anno_root, f"{scene_id}_artic.json")
        if not os.path.exists(anno_path):
            continue
        seg_path = os.path.join(scene_dir, "scans", "segments_anno.json")
        if not os.path.exists(seg_path):
            continue
        with open(seg_path) as f:
            anno_data = json.load(f)
        for grp in anno_data.get("segGroups", []):
            pid = grp.get("partId", 0)
            if pid > 0:
                all_ids.add(pid)

    sorted_ids = sorted(all_ids)
    # 0 → 0 (background), foreground ids → 1..N
    remap = {0: 0}
    for new_id, old_id in enumerate(sorted_ids, start=1):
        remap[old_id] = new_id

    return remap, len(sorted_ids) + 1  # +1 for background class 0


def load_ply_vertices(ply_path):
    import struct
    with open(ply_path, "rb") as f:
        n_verts = 0
        has_rgb = False
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            line_s = line.decode("utf-8", errors="ignore").strip()
            if line_s.startswith("element vertex"):
                n_verts = int(line_s.split()[-1])
            if "red" in line_s:
                has_rgb = True
            if line_s == "end_header":
                break
        stride = 15 if has_rgb else 12
        raw = f.read(n_verts * stride)
    coords = np.zeros((n_verts, 3), dtype=np.float32)
    colors = np.zeros((n_verts, 3), dtype=np.uint8)
    for i in range(n_verts):
        off = i * stride
        x, y, z = struct.unpack_from("<fff", raw, off)
        coords[i] = [x, y, z]
        if has_rgb:
            colors[i] = struct.unpack_from("<BBB", raw, off + 12)
    return coords, colors


def build_vertex_labels(scene_dir, artic_annotation, part_id_remap=None):
    seg_path = os.path.join(scene_dir, "scans", "segments.json")
    with open(seg_path) as f:
        seg_data = json.load(f)
    seg_indices = np.array(seg_data["segIndices"], dtype=np.int32)
    N = len(seg_indices)

    anno_path = os.path.join(scene_dir, "scans", "segments_anno.json")
    with open(anno_path) as f:
        anno_data = json.load(f)

    seg_to_part = {}
    for grp in anno_data["segGroups"]:
        pid = grp.get("partId", 0)
        for sid in grp.get("segments", []):
            seg_to_part[sid] = pid

    parts     = {p["pid"]: p for p in artic_annotation["data"]["parts"]}
    artic_map = {a["pid"]: a for a in artic_annotation["data"]["articulations"]}

    part_ids  = np.zeros(N, dtype=np.int32)
    mobility  = np.zeros(N, dtype=np.int32)
    axis      = np.zeros((N, 3), dtype=np.float32)
    origin    = np.zeros((N, 3), dtype=np.float32)
    range_min = np.zeros(N, dtype=np.float32)
    range_max = np.zeros(N, dtype=np.float32)
    semantic  = np.zeros(N, dtype=np.int32)

    for i in range(N):
        sid = int(seg_indices[i])
        pid = seg_to_part.get(sid, 0)

        # remap to compact id space if remap provided
        if part_id_remap is not None:
            pid_mapped = part_id_remap.get(pid, 0)  # unknown → background
        else:
            pid_mapped = pid

        part_ids[i] = pid_mapped
        part_info   = parts.get(pid, {})
        grp_name    = part_info.get("group", "others")
        semantic[i] = GROUP_TO_SEMANTIC.get(grp_name, 0)

        if pid in artic_map:
            a        = artic_map[pid]
            mob_type = a.get("type", "static")
            mobility[i] = MOBILITY_CLASS.get(mob_type, 0)
            ax = np.array(a.get("axis", [0, 0, 0]), dtype=np.float32)
            norm = np.linalg.norm(ax)
            if norm > 1e-6:
                ax = ax / norm
            axis[i]      = ax
            origin[i]    = np.array(a.get("origin", [0, 0, 0]), dtype=np.float32)
            range_min[i] = float(a.get("rangeMin", 0.0))
            range_max[i] = float(a.get("rangeMax", 0.0))

    return part_ids, mobility, axis, origin, range_min, range_max, semantic


class Articulate3DDataset(Dataset):
    def __init__(
        self,
        data_root,
        anno_root,
        split="train",
        voxel_size=0.05,
        max_points=None,
        augment=False,
        part_id_remap=None,   # pass in pre-built remap dict
    ):
        self.data_root    = data_root
        self.anno_root    = anno_root
        self.voxel_size   = voxel_size
        self.max_points   = max_points
        self.augment      = augment
        self.part_id_remap = part_id_remap

        split_file = os.path.join(anno_root, f"{split}.txt")
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.scene_ids = [
            sid for sid in all_ids
            if os.path.exists(os.path.join(data_root, sid, "scans", "mesh_aligned_0.05.ply"))
        ]
        skipped = len(all_ids) - len(self.scene_ids)
        if skipped:
            print(f"[Articulate3D] WARNING: skipped {skipped} missing scenes")
        print(f"[Articulate3D] {split} split: {len(self.scene_ids)} scenes")

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        scene_id  = self.scene_ids[idx]
        scene_dir = os.path.join(self.data_root, scene_id)
        ply_path  = os.path.join(scene_dir, "scans", "mesh_aligned_0.05.ply")

        coords, colors = load_ply_vertices(ply_path)
        colors_f = colors.astype(np.float32) / 127.5 - 1.0

        anno_path = os.path.join(self.anno_root, f"{scene_id}_artic.json")
        with open(anno_path) as f:
            artic_anno = json.load(f)

        part_ids, mobility, axis, origin, rmin, rmax, semantic = \
            build_vertex_labels(scene_dir, artic_anno, self.part_id_remap)

        if self.voxel_size > 0.05:
            coords, colors_f, part_ids, mobility, axis, origin, rmin, rmax, semantic = \
                self._voxel_downsample(
                    coords, colors_f, part_ids, mobility,
                    axis, origin, rmin, rmax, semantic
                )

        N = len(coords)
        if self.max_points and N > self.max_points:
            idx_sel  = np.random.choice(N, self.max_points, replace=False)
            coords   = coords[idx_sel]
            colors_f = colors_f[idx_sel]
            part_ids = part_ids[idx_sel]
            mobility = mobility[idx_sel]
            axis     = axis[idx_sel]
            origin   = origin[idx_sel]
            rmin     = rmin[idx_sel]
            rmax     = rmax[idx_sel]
            semantic = semantic[idx_sel]

        if self.augment:
            coords = self._augment(coords)

        coords -= coords.mean(axis=0)
        feat = np.concatenate([colors_f, np.zeros((len(coords), 3), np.float32)], axis=1)

        return {
            "coord":     torch.from_numpy(coords),
            "color":     torch.from_numpy(colors_f),
            "feat":      torch.from_numpy(feat),
            "part_id":   torch.from_numpy(part_ids),
            "mobility":  torch.from_numpy(mobility),
            "axis":      torch.from_numpy(axis),
            "origin":    torch.from_numpy(origin),
            "range_min": torch.from_numpy(rmin),
            "range_max": torch.from_numpy(rmax),
            "semantic":  torch.from_numpy(semantic),
            "scene_id":  scene_id,
        }

    def _voxel_downsample(self, coords, colors, part_ids, mobility,
                          axis, origin, rmin, rmax, semantic):
        voxel = np.floor(coords / self.voxel_size).astype(np.int32)
        keys  = voxel[:, 0] * 1_000_000 + voxel[:, 1] * 1_000 + voxel[:, 2]
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        M = len(counts)

        new_coords   = np.zeros((M, 3), np.float32)
        new_colors   = np.zeros((M, 3), np.float32)
        new_axis     = np.zeros((M, 3), np.float32)
        new_origin   = np.zeros((M, 3), np.float32)
        new_rmin     = np.zeros(M, np.float32)
        new_rmax     = np.zeros(M, np.float32)
        new_part     = np.zeros(M, np.int32)
        new_mobility = np.zeros(M, np.int32)
        new_semantic = np.zeros(M, np.int32)

        np.add.at(new_coords, inv, coords)
        np.add.at(new_colors, inv, colors)
        np.add.at(new_axis,   inv, axis)
        np.add.at(new_origin, inv, origin)
        np.add.at(new_rmin,   inv, rmin)
        np.add.at(new_rmax,   inv, rmax)

        new_coords /= counts[:, None]
        new_colors /= counts[:, None]
        new_axis   /= counts[:, None]
        new_origin /= counts[:, None]
        new_rmin   /= counts
        new_rmax   /= counts

        for i in range(M):
            mask = inv == i
            new_part[i]     = np.bincount(part_ids[mask]).argmax()
            new_mobility[i] = np.bincount(mobility[mask]).argmax()
            new_semantic[i] = np.bincount(semantic[mask]).argmax()

        return (new_coords, new_colors, new_part, new_mobility,
                new_axis, new_origin, new_rmin, new_rmax, new_semantic)

    def _augment(self, coords):
        theta = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(theta), np.sin(theta)
        R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        coords = coords @ R.T
        coords += np.random.randn(*coords.shape).astype(np.float32) * 0.005
        return coords


def collate_fn(batch):
    keys   = [k for k in batch[0].keys() if k != "scene_id"]
    out    = {}
    offset = []
    total  = 0
    for sample in batch:
        total += len(sample["coord"])
        offset.append(total)
    for k in keys:
        out[k] = torch.cat([s[k] for s in batch], dim=0)
    out["offset"]   = torch.tensor(offset, dtype=torch.int32)
    out["scene_id"] = [s["scene_id"] for s in batch]
    return out

