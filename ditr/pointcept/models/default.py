import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from einops import rearrange

from pointcept.models.losses import build_criteria
from pointcept.models.point_transformer_v3.dinov2 import FrozenDINOv2
from pointcept.models.point_transformer_v3.utils import (
    assign_image_feat,
    get_image_feat,
    mix3d_cls_token,
)
from pointcept.models.utils.structure import Point

from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultDistiller(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        dinov2,
        backbone_bottleneck_channels=None,
        with_cls_tokens=None,
        backbone=None,
        criteria=None,
        cls_token_criteria=None,
        context_channels=None,
        conditions=None,
        head_decouple=False,
    ):
        super().__init__()
        self.img_enc = FrozenDINOv2(model=dinov2)

        if head_decouple:
            assert conditions is not None
            self.proj = nn.ModuleList(
                nn.Linear(backbone_out_channels, self.img_enc.output_channels)
                for _ in range(len(conditions))
            )
        else:
            self.proj = nn.Linear(backbone_out_channels, self.img_enc.output_channels)

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        if with_cls_tokens is not None:
            assert cls_token_criteria is not None
            assert backbone_bottleneck_channels is not None
            self.cls_token_criteria = build_criteria(cls_token_criteria)
            if head_decouple:
                assert conditions is not None
                self.proj_cls_token = nn.ModuleList(
                    nn.Linear(
                        backbone_bottleneck_channels,
                        self.img_enc.output_channels * with_cls_tokens[i],
                    )
                    for i in range(len(conditions))
                )
            else:
                self.proj_cls_token = nn.Linear(
                    backbone_bottleneck_channels,
                    self.img_enc.output_channels * with_cls_tokens,
                )
        self.with_cls_tokens = with_cls_tokens
        self.conditions = conditions
        if conditions is not None and context_channels is not None:
            self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.head_decouple = head_decouple

    def forward(self, input_dict):
        proj = self.proj
        if self.with_cls_tokens:
            proj_cls_token = self.proj_cls_token
        if self.conditions is not None:
            condition = input_dict["condition"][0]
            assert condition in self.conditions

            if hasattr(self, "embedding_table"):
                context = self.embedding_table(
                    torch.tensor(
                        [self.conditions.index(condition)],
                        device=input_dict["coord"].device,
                    )
                )
                input_dict["context"] = context

            if self.head_decouple:
                proj = self.proj[self.conditions.index(condition)]
                if self.with_cls_tokens:
                    proj_cls_token = self.proj_cls_token[
                        self.conditions.index(condition)
                    ]
        else:
            assert "condition" not in input_dict.keys()

        image_feat, image_cls_token, patch_size = get_image_feat(
            self.img_enc, input_dict
        )  # (B, CAM, C, H, W)
        image_feat = assign_image_feat(input_dict, image_feat, patch_size)  # (N, C)
        if self.with_cls_tokens:
            image_cls_token = mix3d_cls_token(
                input_dict, image_cls_token
            )  # (B, CAM, C)

        point = Point(input_dict)
        point = self.backbone(point)

        pred = proj(point.feat)
        target = image_feat
        if "image_mask" in point.keys():
            msk = point.image_mask.any(dim=1)  # visible in any view
            pred = pred[msk]
            target = target[msk]

        out_dict = dict()
        out_dict["patch_loss"] = self.criteria(pred, target)
        out_dict["loss"] = out_dict["patch_loss"]

        if self.with_cls_tokens:
            bottleneck = point
            while "unpooling_parent" in bottleneck.keys():
                bottleneck = bottleneck.unpooling_parent

            pooled_feat = torch_scatter.segment_csr(
                src=bottleneck.feat,
                indptr=nn.functional.pad(bottleneck.offset, (1, 0)),
                reduce="mean",
            )

            out_dict["cls_token_loss"] = self.cls_token_criteria(
                rearrange(
                    proj_cls_token(pooled_feat),
                    "b (cam c) -> (b cam) c",
                    cam=image_cls_token.shape[1],
                    c=image_cls_token.shape[2],
                ),
                rearrange(image_cls_token, "b cam c -> (b cam) c"),
            )
            out_dict["loss"] = (
                out_dict["loss"] + out_dict["cls_token_loss"]
            )  # don't use +=

        if self.training:
            return out_dict

        out_dict["feat"] = point.feat
        return out_dict


@MODELS.register_module()
class TeacherStudent(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        levels,
        backbone=None,
        teacher=None,
        ckpt_path=None,
        load_student=False,
        distill_criteria=None,
        seg_criteria=None,
    ):
        super().__init__()
        if seg_criteria is not None:
            self.seg_head = (
                nn.Linear(backbone_out_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.backbone = build_model(backbone)
        self.teacher = build_model(teacher)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)["state_dict"]
            if seg_criteria is not None:
                self.seg_head.weight.data = state_dict["module.seg_head.weight"]
                self.seg_head.bias.data = state_dict["module.seg_head.bias"]

            weight = dict()
            for key, value in state_dict.items():
                if key.startswith("module.backbone"):
                    key = key.replace("module.backbone.", "").replace(
                        "context_qkv", "qkv_cls_token"
                    )
                    weight[key] = value
            self.teacher.load_state_dict(weight, strict=True)
            if load_student:
                weight = {
                    key: value
                    for key, value in weight.items()
                    if ("proj_image_feat" not in key) and ("qkv_cls_token" not in key)
                }
                self.backbone.load_state_dict(weight, strict=True)

        self.distill_criteria = build_criteria(distill_criteria)
        self.seg_criteria = (
            build_criteria(seg_criteria) if seg_criteria is not None else None
        )
        self.levels = levels
        self.distill_decay = 1.0

    def train(self, mode: bool = True):
        super().train(mode)

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward(self, input_dict):
        backbone_point = Point(input_dict)
        teacher_point = Point(input_dict)
        backbone_point = self.backbone(backbone_point)
        teacher_point = self.teacher(teacher_point)

        if self.seg_criteria is not None:
            feat = backbone_point.feat
            seg_logits = self.seg_head(feat)

        backbone_points = [backbone_point]
        while "unpooling_parent" in backbone_points[-1].keys():
            backbone_points.append(backbone_points[-1].unpooling_parent)

        teacher_points = [teacher_point]
        while "unpooling_parent" in teacher_points[-1].keys():
            teacher_points.append(teacher_points[-1].unpooling_parent)

        out_dict = dict()
        if self.training:
            distill_losses = []
            for lvl in self.levels:
                pred = backbone_points[lvl]
                target = teacher_points[lvl]
                distill_losses.append(self.distill_criteria(pred.feat, target.feat))
            for i, loss in enumerate(distill_losses):
                out_dict[f"distill_loss_lvl_{self.levels[i]}"] = loss

            out_dict["distill_decay"] = torch.tensor(self.distill_decay)
            out_dict["distill_loss"] = (
                self.distill_decay * sum(distill_losses) / len(distill_losses)
            )

            if self.seg_criteria is not None:
                out_dict["seg_loss"] = self.seg_criteria(
                    seg_logits, input_dict["segment"]
                )
                out_dict["loss"] = out_dict["seg_loss"] + out_dict["distill_loss"]
            else:
                out_dict["loss"] = out_dict["distill_loss"]
        elif "segment" in input_dict.keys():
            distill_losses = []
            for lvl in self.levels:
                pred = backbone_points[lvl]
                target = teacher_points[lvl]
                distill_losses.append(self.distill_criteria(pred.feat, target.feat))
            for i, loss in enumerate(distill_losses):
                out_dict[f"distill_loss_lvl_{self.levels[i]}"] = loss

            out_dict["distill_loss"] = (
                self.distill_decay * sum(distill_losses) / len(distill_losses)
            )

            if self.seg_criteria is not None:
                out_dict["seg_loss"] = self.seg_criteria(
                    seg_logits, input_dict["segment"]
                )
                out_dict["loss"] = out_dict["seg_loss"] + out_dict["distill_loss"]
                out_dict["seg_logits"] = seg_logits
            else:
                out_dict["loss"] = out_dict["distill_loss"]
        else:
            if self.seg_criteria is not None:
                out_dict["seg_logits"] = seg_logits

        return out_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class ArticulateSegmentor(nn.Module):
    """Per-point multi-head model for Articulate3D challenge.

    Keeps the PT-v3 backbone and adds parallel prediction heads:
    - seg_head: per-point segmentation (static / rotation / translation)
    - axis_head: per-point articulation axis (3D direction)
    - origin_head: per-point articulation origin offset (3D)
    - type_head: per-point motion type (3-class: static/rotation/translation)
    - range_head: per-point motion range (min, max)
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        hidden_channels=128,
        backbone=None,
        seg_criteria=None,
        axis_loss_weight=1.0,
        origin_loss_weight=1.0,
        type_loss_weight=1.0,
        range_loss_weight=1.0,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.seg_criteria = build_criteria(seg_criteria)
        self.axis_loss_weight = axis_loss_weight
        self.origin_loss_weight = origin_loss_weight
        self.type_loss_weight = type_loss_weight
        self.range_loss_weight = range_loss_weight

        # Segmentation head
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Articulation axis head: (N, C) -> (N, 3)
        self.axis_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

        # Articulation origin head: (N, C) -> (N, 3) offset from point coord
        self.origin_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

        # Motion type head: (N, C) -> (N, 3) logits for static/rotation/translation
        self.type_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

        # Motion range head: (N, C) -> (N, 2) for (range_min, range_max)
        self.range_head = nn.Sequential(
            nn.Linear(backbone_out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point

        seg_logits = self.seg_head(feat)
        axis_pred = self.axis_head(feat)
        origin_offset = self.origin_head(feat)
        type_logits = self.type_head(feat)
        range_pred = self.range_head(feat)

        # Absolute origin = point coordinate + predicted offset
        origin_pred = input_dict["coord"] + origin_offset

        if self.training:
            return self._compute_losses(
                seg_logits, axis_pred, origin_pred, type_logits, range_pred, input_dict
            )
        elif "segment" in input_dict.keys():
            out = self._compute_losses(
                seg_logits, axis_pred, origin_pred, type_logits, range_pred, input_dict
            )
            out["seg_logits"] = seg_logits
            out["axis_pred"] = axis_pred
            out["origin_pred"] = origin_pred
            out["type_logits"] = type_logits
            out["range_pred"] = range_pred
            return out
        else:
            return dict(
                seg_logits=seg_logits,
                axis_pred=axis_pred,
                origin_pred=origin_pred,
                type_logits=type_logits,
                range_pred=range_pred,
            )

    def _compute_losses(
        self, seg_logits, axis_pred, origin_pred, type_logits, range_pred, input_dict
    ):
        # Segmentation loss (standard CE + Lovasz)
        seg_loss = self.seg_criteria(seg_logits, input_dict["segment"])

        # Articulation targets
        artic_type = input_dict["artic_type"]  # (N,) 0=static, 1=rotation, 2=translation
        artic_axis = input_dict["artic_axis"]  # (N, 3)
        artic_origin = input_dict["artic_origin"]  # (N, 3)
        artic_range = input_dict["artic_range"]  # (N, 2)

        artic_mask = artic_type > 0  # articulable points
        rot_mask = artic_type == 1
        trans_mask = artic_type == 2

        # Type loss: 3-class CE on ALL points
        type_loss = F.cross_entropy(type_logits, artic_type)

        device = seg_logits.device
        axis_loss = torch.tensor(0.0, device=device)
        origin_loss = torch.tensor(0.0, device=device)
        range_loss = torch.tensor(0.0, device=device)

        if artic_mask.sum() > 0:
            # --- Axis loss ---
            if rot_mask.sum() > 0:
                axis_pred_r = axis_pred[rot_mask]
                axis_gt_r = artic_axis[rot_mask]
                pred_norm = F.normalize(axis_pred_r, dim=-1)
                gt_norm = F.normalize(axis_gt_r, dim=-1)
                cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=-1)
                axis_loss = axis_loss + (1 - cos_sim.abs()).mean()

            if trans_mask.sum() > 0:
                axis_pred_t = axis_pred[trans_mask]
                axis_gt_t = artic_axis[trans_mask]
                pred_norm = F.normalize(axis_pred_t, dim=-1)
                gt_norm = F.normalize(axis_gt_t, dim=-1)
                axis_loss = axis_loss + F.l1_loss(pred_norm, gt_norm)

            # --- Origin loss ---
            if rot_mask.sum() > 0:
                # Point-to-line distance: ||(o_pred - o_gt) x axis_gt|| / ||axis_gt||
                o_pred_r = origin_pred[rot_mask]
                o_gt_r = artic_origin[rot_mask]
                a_gt_r = artic_axis[rot_mask]
                diff = o_pred_r - o_gt_r
                cross = torch.cross(diff, a_gt_r, dim=-1)
                dist = cross.norm(dim=-1) / (a_gt_r.norm(dim=-1) + 1e-8)
                origin_loss = origin_loss + dist.mean()

            if trans_mask.sum() > 0:
                o_pred_t = origin_pred[trans_mask]
                o_gt_t = artic_origin[trans_mask]
                origin_loss = origin_loss + F.l1_loss(o_pred_t, o_gt_t)

            # --- Range loss ---
            range_pred_a = range_pred[artic_mask]
            range_gt_a = artic_range[artic_mask]
            range_loss = F.l1_loss(range_pred_a, range_gt_a)

        loss = (
            seg_loss
            + self.axis_loss_weight * axis_loss
            + self.origin_loss_weight * origin_loss
            + self.type_loss_weight * type_loss
            + self.range_loss_weight * range_loss
        )

        return dict(
            loss=loss,
            seg_loss=seg_loss,
            axis_loss=axis_loss,
            origin_loss=origin_loss,
            type_loss=type_loss,
            range_loss=range_loss,
        )
