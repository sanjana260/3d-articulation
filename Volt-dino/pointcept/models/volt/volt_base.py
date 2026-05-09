import flash_attn
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
from torch.nn.init import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.volt.decoder import Decoder


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float = 100.0,
        freq_split: tuple = (12, 12, 8),
        max_grid_size: tuple = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])

        # Precompute the complex values for the maximum possible grid size
        self.register_buffer(
            "cis_cache_x", self._precompute(freqs_x, max_grid_size[0]), persistent=False
        )
        self.register_buffer(
            "cis_cache_y", self._precompute(freqs_y, max_grid_size[1]), persistent=False
        )
        self.register_buffer(
            "cis_cache_z", self._precompute(freqs_z, max_grid_size[2]), persistent=False
        )

    def _precompute(self, freqs, max_pos):
        # Create a lookup table [Max_Pos, Dim]
        freqs_pos = torch.outer(torch.arange(max_pos).float(), freqs)
        return torch.polar(torch.ones_like(freqs_pos), freqs_pos)

    def compute_axial_cis_efficient(self, indices):
        cis_x = self.cis_cache_x[indices[:, 0]]
        cis_y = self.cis_cache_y[indices[:, 1]]
        cis_z = self.cis_cache_z[indices[:, 2]]
        return torch.cat([cis_x, cis_y, cis_z], dim=-1).unsqueeze(0)


class RoPE_Attention(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.h_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(2)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)

        return q_out.type_as(q), k_out.type_as(k)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        N, C = x.shape
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.h_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q).to(q.dtype), self.k_norm(k).to(k.dtype)
        q, k = self.apply_rotary_emb(q, k, freqs_cis)
        qkv = torch.stack([q, k, v], dim=0).permute(2, 0, 1, 3)

        qkv_dtype = qkv.dtype
        x = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half(),
            cu_seqlens,
            max_seqlen=max_seqlen,
        )

        x = x.reshape(-1, C).to(qkv_dtype)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float | None = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPE_Attention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), freqs_cis, cu_seq_lens, max_seqlen))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ── DITR-style DINOv2 injection helpers ────────────────────────────────────────
# These two functions implement the voxel→patch aggregation described in the DITR
# paper (Algorithm 1). They are module-level utilities used by Volt.forward().

def _compute_voxel_to_patch_idx(
    grid_coord: torch.Tensor,   # (N, 3) int  — input voxel coordinates
    batch: torch.Tensor,        # (N,) int    — batch indices
    patch_indices: torch.Tensor,  # (T, 4) int  — [b, px, py, pz] from tokenizer
    stride: int,
) -> torch.Tensor:
    """
    For each input voxel, find the index of its corresponding patch token in [0, T).

    The tokenizer (SparseConv3d, stride=s) maps input voxel at (gx, gy, gz) to
    patch at (gx//s, gy//s, gz//s). We find, for each voxel, which row in
    patch_indices that patch corresponds to.

    Uses a sort+searchsorted approach: O(N log N), O(N+T) memory — avoids large
    lookup arrays.
    """
    device = grid_coord.device

    # Compute the patch-space coordinate each voxel falls into: (N, 3)
    voxel_patch_xyz = grid_coord.long() // stride

    # Encode (batch, px, py, pz) as a single int64 key.
    # We use the actual max patch coordinate to avoid overflow.
    max_p = int(patch_indices[:, 1:].max().item()) + 1  # upper bound per spatial dim

    def _encode(b, xyz):
        # key = b*P^3 + px*P^2 + py*P + pz   (P = max_p, safe within int64)
        P = max_p
        return (
            b.long() * (P * P * P)
            + xyz[:, 0].long() * (P * P)
            + xyz[:, 1].long() * P
            + xyz[:, 2].long()
        )

    patch_keys = _encode(patch_indices[:, 0], patch_indices[:, 1:])  # (T,)
    voxel_keys = _encode(batch, voxel_patch_xyz)                     # (N,)

    # Sort patch keys and build an inverse-rank map
    T = patch_indices.shape[0]
    sorted_patch_keys, sorted_patch_order = patch_keys.sort()  # (T,)
    inv_patch_order = torch.empty_like(sorted_patch_order)
    inv_patch_order[sorted_patch_order] = torch.arange(T, device=device)

    # For each voxel key, find its rank in the sorted patch keys
    positions = torch.searchsorted(
        sorted_patch_keys.contiguous(), voxel_keys.contiguous()
    )  # (N,) — each value is an index into sorted_patch_keys

    # Map back to the original (unsorted) patch index
    positions = positions.clamp(0, T - 1)          # safety clamp
    voxel_to_patch = inv_patch_order[positions]    # (N,)
    return voxel_to_patch


def _aggregate_to_patch_level(
    dino_feat: torch.Tensor,        # (N, D)  per-voxel DINOv2 features
    voxel_to_patch: torch.Tensor,   # (N,)    patch index for each voxel
    T: int,                          # number of patch tokens
) -> torch.Tensor:
    """
    Max-pool voxel-level DINOv2 features into patch-level features (T, D).

    Voxels that share a patch token contribute via max: the most informative
    2D feature within each 3D patch wins. Matches DITR paper §3.2.
    """
    D = dino_feat.shape[1]
    X2D_patch = torch.zeros(T, D, dtype=dino_feat.dtype, device=dino_feat.device)
    # scatter_reduce_ is available in PyTorch >= 1.12 / 2.0
    X2D_patch.scatter_reduce_(
        dim=0,
        index=voxel_to_patch.unsqueeze(1).expand(-1, D),
        src=dino_feat,
        reduce="amax",
        include_self=True,
    )
    return X2D_patch  # (T, D)


# ── Main Volt model ─────────────────────────────────────────────────────────────

@MODELS.register_module("Volt")
class Volt(nn.Module):
    def __init__(
        self,
        in_channels=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=None,
        qk_norm=False,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        up_mlp_dim=256,
        increase_drop_path=True,
        # ── DITR DINOv2 injection (NEW) ──────────────────────────────────────
        use_dino=False,    # enable 2D→3D DINOv2 feature injection
        dino_dim=1024,     # DINOv2 output dim: 1024 for ViT-L, 768 for ViT-B
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        # Store stride so forward() can compute voxel→patch mapping
        self.stride = stride

        self.tokenizer = spconv.SparseConv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            indice_key="embedding",
        )

        if increase_drop_path:
            drop_path_list = torch.linspace(0, drop_path, depth).tolist()
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=drop_path_list[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=Mlp,
                )
                for i in range(depth)
            ]
        )

        self.pos_enc = RoPE()
        self.decoder = Decoder(
            in_channels=embed_dim,
            out_channels=up_mlp_dim,
            kernel_size=kernel_size,
            indice_key="embedding",
        )

        # ── DITR DINOv2 projection layers (NEW) ──────────────────────────────
        self.use_dino = use_dino
        if use_dino:
            self.dino_dim = dino_dim

            # Level 1: patch-level injection (pre-transformer)
            # Projects DINOv2 features from dino_dim → embed_dim and adds to patch tokens
            self.dino_proj_patch = nn.Sequential(
                nn.Linear(dino_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            )

            # Level 2: voxel-level injection (post-decoder)
            # Projects DINOv2 features from dino_dim → up_mlp_dim and adds to voxel features
            self.dino_proj_voxel = nn.Sequential(
                nn.Linear(dino_dim, up_mlp_dim),
                nn.BatchNorm1d(up_mlp_dim),
                nn.GELU(),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    @staticmethod
    def compute_seqlens(batch_indices):
        # Add 1 to batch_indices such that the first count is 0
        points_per_batch = torch.bincount(batch_indices + 1)
        cu_seqlens = torch.cumsum(points_per_batch, dim=0, dtype=torch.int32)
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = sequence_lengths.max().item()
        return cu_seqlens, max_seqlen

    def forward(self, data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        batch = data_dict["batch"]
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        indices = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()

        x = spconv.SparseConvTensor(
            features=feat,
            indices=indices,
            spatial_shape=sparse_shape,
            batch_size=batch[-1].item() + 1,
        )

        # ── STEP 1: Tokenisation (sparse strided conv) ────────────────────────
        x = self.tokenizer(x)
        # x.features: (T, embed_dim)  — T << N  (one token per 5³ voxel block)
        # x.indices:  (T, 4)          — [batch, patch_x, patch_y, patch_z]

        # ── STEP 2: Level-1 DINOv2 injection (patch level, pre-transformer) ──
        # Aggregate per-voxel DINOv2 features to patch level, project, and add.
        # This gives the transformer access to 2D semantic information from the start.
        if self.use_dino and "dino_feat" in data_dict:
            dino_feat = data_dict["dino_feat"]          # (N, dino_dim)
            T = x.features.shape[0]

            # Map each input voxel to its patch token index in [0, T)
            voxel_to_patch = _compute_voxel_to_patch_idx(
                grid_coord=grid_coord,
                batch=batch,
                patch_indices=x.indices,
                stride=self.stride,
            )  # (N,)

            # Max-pool voxel DINOv2 features → patch-level features (T, dino_dim)
            X2D_patch = _aggregate_to_patch_level(dino_feat, voxel_to_patch, T)

            # Project to embed_dim and inject into patch tokens
            dino_patch_feat = self.dino_proj_patch(X2D_patch)   # (T, embed_dim)
            x = x.replace_feature(x.features + dino_patch_feat)

        # ── STEP 3: Transformer encoder (unchanged) ───────────────────────────
        cu_seqlens, max_seqlen = self.compute_seqlens(x.indices[:, 0])
        freqs_cis = self.pos_enc.compute_axial_cis_efficient(x.indices[:, 1:])

        features = x.features
        for blk in self.blocks:
            features = blk(features, freqs_cis, cu_seqlens, max_seqlen)

        x = x.replace_feature(features)

        # ── STEP 4: Decoder (transposed conv, stride restored) ────────────────
        x = self.decoder(x)
        # x.features: (N, up_mlp_dim)

        # ── STEP 5: Level-2 DINOv2 injection (voxel level, post-decoder) ──────
        # Directly enrich each voxel's feature with its corresponding 2D DINOv2
        # feature. This is the primary injection point per the DITR paper.
        if self.use_dino and "dino_feat" in data_dict:
            dino_voxel_feat = self.dino_proj_voxel(dino_feat)   # (N, up_mlp_dim)
            x = x.replace_feature(x.features + dino_voxel_feat)

        return x.features
