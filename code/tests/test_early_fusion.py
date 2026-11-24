"""六通道早融合的等价性与回归测试（CPU 可跑，不占用 GPU）。

关键性质：新增的 IR/Depth/Valid 通道在 patch-embed 上零初始化，因此
fusion=early 的模型在初始化时刻必须与 RGB-only 对照逐元素一致；
同时默认的 fusion=late 路径必须保持 3 通道 patch-embed 不变。
"""
import torch
import yaml

from src.trimodal_data import load_config
from src.trimodal_rfdetr import NestedTensor, build_detector, patch_embed_projection

CONFIG = "configs/rf48b-1280.yaml"


def small_config(**overrides):
    config = load_config(CONFIG)
    config.update(imgsz=128, batch=1, accumulation=1, workers=0)
    config.update(overrides)
    return config


def patch_embed(model):
    return patch_embed_projection(model.backbone[0].encoder)[1]


def test_late_fusion_keeps_three_channel_patch_embed():
    """默认配置（正在训练的 1280 晚融合）不得被早融合改动波及。"""
    model, _, _ = build_detector(small_config())
    projection = patch_embed(model)
    assert projection.in_channels == 3, projection.in_channels
    assert model.backbone[0].early_fusion is False


def test_early_fusion_widens_patch_embed_and_zero_inits_new_channels():
    model, _, _ = build_detector(small_config(fusion="early"))
    projection = patch_embed(model)
    assert projection.in_channels == 6, projection.in_channels
    assert model.backbone[0].early_fusion is True
    assert float(projection.weight[:, 3:].abs().max()) == 0.0, "新增通道必须零初始化"
    assert float(projection.weight[:, :3].abs().max()) > 0.0, "预训练 RGB 通道必须保留"


def test_early_fusion_matches_rgb_control_at_initialization():
    """零初始化 ⇒ 初始前向必须与 RGB-only 对照逐元素相等。"""
    config = small_config(fusion="early")
    fused, _, _ = build_detector(config)
    control, _, _ = build_detector(config | {"multimodal": False})
    images = torch.rand(1, 6, 128, 128)
    mask = torch.zeros(1, 128, 128, dtype=torch.bool)
    fused.eval()
    control.eval()
    with torch.no_grad():
        left = fused(NestedTensor(images, mask))
        right = control(NestedTensor(images, mask))
    difference = max(float((left[k] - right[k]).abs().max()) for k in ("pred_logits", "pred_boxes"))
    # 新增通道权重恒为 0，故差异应为 0；不同卷积形状在 CPU 上可能走不同 kernel，
    # 因此留 1e-5 的浮点噪声容限（实测远小于该值）。
    print(f"initialization difference = {difference:.3e}")
    assert difference < 1e-5, difference


def test_public_checkpoint_merges_into_wide_patch_embed():
    """公开 3 通道权重 + 早融合：补丁嵌入自动合并，RGB 通道保留、新增通道零初始化。

    这是"从公开预训练权重出发做早融合训练"的正确性保证：
    若把 6 通道的补丁嵌入整体跳过，预训练 RGB 能力就丢了。
    """
    config = small_config(fusion="early")
    model, _, report = build_detector(config)
    assert report["class_head_loaded"] is False, "公开权重是 91 类头，不应被继承"
    assert len(report["resized_tensors"]) == 1, report["resized_tensors"]
    item = report["resized_tensors"][0]
    assert item["from"][:2] == [384, 3] and item["to"][:2] == [384, 6], item
    projection = patch_embed(model)
    assert projection.in_channels == 6
    assert float(projection.weight[:, 3:].abs().max()) == 0.0, "新增通道必须零初始化"
    assert float(projection.weight[:, :3].abs().max()) > 0.0, "RGB 通道必须保留预训练权重"


def test_public_checkpoint_missing_keys_are_only_heads_and_branches():
    """公开权重加载后，缺失项只能是分类头与新建的传感器分支。"""
    for fusion in ("late", "early"):
        _, _, report = build_detector(small_config(fusion=fusion))
        allowed = ("class_embed", "ir_branch", "depth_branch")
        disallowed = [k for k in report["missing_keys"] if not any(a in k for a in allowed)]
        assert not disallowed, (fusion, disallowed[:5])
        assert not report["unexpected_keys"], report["unexpected_keys"]


def test_gradients_reach_every_modality_path():
    """零初始化不等于零梯度：RGB 主干与两个传感器分支都必须收到非零梯度。

    上一阶段曾出现"损失有限但主干梯度为零"的静默失败（居中补边被误标为 padding），
    故这里对四条路径逐一断言。
    """
    model, _, _ = build_detector(small_config(fusion="early"))
    model.train()
    images = torch.rand(1, 6, 128, 128)
    mask = torch.zeros(1, 128, 128, dtype=torch.bool)
    output = model(NestedTensor(images, mask))
    loss = output["pred_logits"].float().sum() + output["pred_boxes"].float().sum()
    loss.backward()

    backbone = model.backbone[0]
    projection = patch_embed_projection(backbone.encoder)[1]
    assert projection.weight.grad is not None, "patch-embed 未收到梯度"
    rgb_grad = float(projection.weight.grad[:, :3].abs().max())
    new_grad = float(projection.weight.grad[:, 3:].abs().max())
    ir_grad = float(backbone.ir_branch.project.weight.grad.abs().max())
    depth_grad = float(backbone.depth_branch.project.weight.grad.abs().max())
    print(f"grads: rgb={rgb_grad:.3e} new_channels={new_grad:.3e} "
          f"ir={ir_grad:.3e} depth={depth_grad:.3e}")
    assert rgb_grad > 0, "RGB 主干梯度为零"
    assert new_grad > 0, "新增模态通道梯度为零（早融合实际未生效）"
    assert ir_grad > 0, "IR 分支梯度为零"
    assert depth_grad > 0, "Depth 分支梯度为零"
