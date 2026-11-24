"""Pinned RF-DETR Medium with trainable sensor residuals; one shared detector."""
import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch import nn
import torch.nn.functional as F

from rfdetr.config import RFDETRMediumConfig
from rfdetr.main import populate_args
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.misc import NestedTensor

from src.inference_checkpoint import load_legacy_checkpoint


class SensorBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for width in (32, 64, 128, 256):
            layers += [nn.Conv2d(channels, width, 3, 2, 1, bias=False), nn.GroupNorm(8, width), nn.SiLU()]
            channels = width
        self.body = nn.Sequential(*layers)
        self.project = nn.Conv2d(256, 256, 1, bias=False)
        self.gate = nn.Conv2d(512, 1, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, sensor, rgb):
        feature = self.body(sensor)
        feature = F.interpolate(feature, rgb.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((rgb, feature), 1)).sigmoid()
        return gate * self.project(feature)


def patch_embed_projection(module):
    """定位 DINOv2 patch-embed 的 Conv2d，返回 (拥有者, Conv2d)。

    RF-DETR 的骨干有多层包装（Backbone -> DinoV2 -> WindowedDinov2WithRegistersBackbone
    -> embeddings.patch_embeddings.projection），层数随版本变化，故按属性查找而非写死路径。
    """
    for _, child in module.named_modules():
        embeddings = getattr(child, "embeddings", None)
        if embeddings is None:
            continue
        projection = getattr(getattr(embeddings, "patch_embeddings", None), "projection", None)
        if isinstance(projection, nn.Conv2d):
            return child, projection
    raise AttributeError(f"patch-embed Conv2d not found under {type(module).__name__}")


def extend_patch_embed(backbone, in_channels):
    """把 DINOv2 patch-embed 的输入通道从 3 扩到 in_channels；新增通道零初始化。

    零初始化保证初始前向与 RGB-only 逐元素一致（与 P4 门控残差同一原理），
    预训练权重因此完整保留，红外/深度从第 0 步起只贡献梯度。这样
    scripts/probe_rfdetr.py 的"零残差 == RGB 对照"断言对早融合同样成立。
    """
    owner, projection = patch_embed_projection(backbone)
    if projection.in_channels == in_channels:
        return projection
    if projection.in_channels != 3:
        raise ValueError(f"Expected 3 pretrained input channels, got {projection.in_channels}")
    widened = nn.Conv2d(in_channels, projection.out_channels, projection.kernel_size,
                        stride=projection.stride, padding=projection.padding,
                        bias=projection.bias is not None)
    with torch.no_grad():
        widened.weight.zero_()
        widened.weight[:, :3].copy_(projection.weight)
        if projection.bias is not None:
            widened.bias.copy_(projection.bias)
    owner.embeddings.patch_embeddings.projection = widened
    # Dinov2WithRegistersPatchEmbeddings.forward 用实例属性校验通道数，必须同步，
    # 否则会抛 "Expected 3 but got 6"；config 一并同步以保持一致。
    owner.embeddings.patch_embeddings.num_channels = in_channels
    if hasattr(owner, "config") and hasattr(owner.config, "num_channels"):
        owner.config.num_channels = in_channels
    return widened


class FusionBackbone(nn.Module):
    """晚融合（默认）：传感器特征只在 P4 与 RGB 残差相加。

    早融合（fusion: early）：RGB/IR/Depth/Valid 六通道直接进入 DINOv2 patch-embed，
    使主干从第一层就能看到传感器模态，同时保留 P4 门控残差分支。
    实测晚融合下 IR 分支贡献约 0.06 分（主干全程看不到 IR），故做此对照。
    """

    def __init__(self, original, multimodal, early_fusion=False):
        super().__init__()
        self.encoder = original.encoder
        self.projector = original.projector
        self.multimodal = multimodal
        self.early_fusion = bool(multimodal and early_fusion)
        if self.early_fusion:
            extend_patch_embed(self.encoder, 6)
        if multimodal:
            self.ir_branch = SensorBranch(1)
            self.depth_branch = SensorBranch(2)

    def forward(self, samples):
        x = samples.tensors
        rgb = self.projector(self.encoder(x if self.early_fusion else x[:, :3]))[0]
        fused = rgb
        if self.multimodal:
            if x.shape[1] != 6:
                raise ValueError("Expected [RGB, IR, Depth, Valid]")
            fused = rgb + self.ir_branch(x[:, 3:4], rgb) + self.depth_branch(x[:, 4:6], rgb)
        mask = F.interpolate(samples.mask[:, None].float(), size=fused.shape[-2:], mode="nearest")[:, 0].bool()
        return [NestedTensor(fused, mask)]


def build_detector(config, initialize=True):
    torch.manual_seed(config["seed"])
    model_config = RFDETRMediumConfig().model_dump()
    # Upstream internally adds 1. Public interface remains 12 classes, IDs 0..11.
    model_config.update(num_classes=11, resolution=config["imgsz"],
                        pretrain_weights=str(config["pretrained"]), device="cpu",
                        gradient_checkpointing=True, num_select=100)
    args = populate_args(**model_config)
    model = build_model(args)
    # 先接好融合结构、再加载权重：早融合会把补丁嵌入扩到 6 通道，
    # 若顺序颠倒，6 通道的 checkpoint 会因模型仍是 3 通道而形状不符、无法加载。
    model.backbone[0] = FusionBackbone(model.backbone[0], config["multimodal"],
                                       early_fusion=config.get("fusion", "late") == "early")
    report = {"missing_keys": [], "unexpected_keys": []}
    if initialize:
        checkpoint = load_legacy_checkpoint(config["pretrained"])
        state = checkpoint["model"]
        target_state = model.state_dict()
        filtered, resized, skipped = {}, [], []
        for key, value in state.items():
            target = target_state.get(key)
            if target is None or tuple(target.shape) == tuple(value.shape):
                filtered[key] = value
                continue
            # 特例：补丁嵌入的输入通道被早融合扩宽过（3→6）。把原有权重复制到前若干通道、
            # 新增通道保持零初始化 —— 与 extend_patch_embed 一致，预训练 RGB 能力不丢。
            if (key.endswith("patch_embeddings.projection.weight")
                    and target.shape[0] == value.shape[0]
                    and tuple(target.shape[2:]) == tuple(value.shape[2:])
                    and target.shape[1] > value.shape[1]):
                merged = torch.zeros_like(target)
                merged[:, :value.shape[1]] = value
                filtered[key] = merged
                resized.append({"key": key, "from": list(value.shape), "to": list(target.shape)})
                continue
            skipped.append({"key": key, "checkpoint": list(value.shape), "model": list(target.shape)})
        result = model.load_state_dict(filtered, strict=False)
        # 允许缺失的只有：分类头（公开权重为 91 类、形状不符）与新建的传感器分支。
        allowed = ("class_embed", "ir_branch", "depth_branch")
        disallowed_missing = [k for k in result.missing_keys if not any(a in k for a in allowed)]
        report = {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
                  "loaded_tensors": len(filtered), "resized_tensors": resized,
                  "skipped_tensors": skipped,
                  "source_class_logits": int(state["class_embed.bias"].shape[0]),
                  "class_head_loaded": not any("class_embed" in k for k in result.missing_keys)}
        if result.unexpected_keys or disallowed_missing:
            raise ValueError(f"Unexpected checkpoint incompatibility: {report}")
    assert model.class_embed.out_features == 12
    assert all(head.out_features == 12 for head in model.transformer.enc_out_class_embed)
    if not report.get("class_head_loaded", False):
        # 仅在分类头确实没被载入时，才应等于新初始化的 focal prior。
        assert torch.allclose(model.class_embed.bias,
                              torch.full_like(model.class_embed.bias, -math.log(99)))
    criterion, _ = build_criterion_and_postprocessors(args)
    criterion.losses = ["labels", "boxes"]  # Upstream cardinality assumes a softmax background slot.
    assert criterion.num_classes == 12
    return model, criterion, report


def output_rows(logits, boxes, geometry, size, threshold=.001):
    """Native sigmoid query/class ranking, global top100; inverse letterbox once."""
    if logits.shape[-1] != 12:
        raise ValueError("Wrong class head")
    scores, indexes = logits.float().sigmoid().flatten().topk(min(100, logits.numel()))
    cls = indexes % 12
    selected = boxes.float()[indexes // 12]
    h, w, nh, nw, y, x = geometry
    corners = torch.cat((selected[:, :2] - selected[:, 2:]/2, selected[:, :2] + selected[:, 2:]/2), 1) * size
    corners[:, [0, 2]] = (corners[:, [0, 2]] - x) * w / nw
    corners[:, [1, 3]] = (corners[:, [1, 3]] - y) * h / nh
    corners[:, [0, 2]] = corners[:, [0, 2]].clamp(0, w)
    corners[:, [1, 3]] = corners[:, [1, 3]].clamp(0, h)
    center = (corners[:, :2] + corners[:, 2:]) / 2
    wh = corners[:, 2:] - corners[:, :2]
    normalized = torch.cat((center, wh), 1) / boxes.new_tensor([w, h, w, h])
    rows = torch.cat((cls[:, None], normalized, scores[:, None]), 1).detach().cpu().tolist()
    from src.check_submit import clean_line
    lines = []
    for c, cx, cy, bw, bh, conf in rows:
        line = f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}"
        if conf >= threshold and clean_line(line.split()) is not None:
            lines.append(line)
    return lines
