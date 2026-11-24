"""letterbox 逆变换在真实画布尺寸下的往返测试。

`output_rows` 负责把网络在方形画布上的归一化坐标反变换回原始图像坐标；
这里在项目实际使用的画布尺寸（768 / 1280 / 1536）上，独立按 `transform` 的公式
构造前向映射，再验证 `output_rows` 能精确还原，避免"几何写错但指标看着正常"的静默错误。
"""
import pytest
import torch

from src.trimodal_rfdetr import output_rows

SIZES = [768, 1280, 1536]
# 数据集两种真实分辨率（均为 16:9）
IMAGES = [(1080, 1920), (360, 640)]


def letterbox_geometry(height, width, size):
    """与 src.trimodal_data.transform 完全一致的居中补边几何。"""
    ratio = size / max(height, width)
    nh, nw = max(1, round(height * ratio)), max(1, round(width * ratio))
    y, x = (size - nh) // 2, (size - nw) // 2
    return ratio, (height, width, nh, nw, y, x)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("height,width", IMAGES)
def test_letterbox_inverse_roundtrip(size, height, width):
    ratio, geometry = letterbox_geometry(height, width, size)
    _, _, _, _, offset_y, offset_x = geometry

    # 原始图像像素坐标下的一个目标框（取画面中部，避开边界裁剪）
    cx_px, cy_px, bw_px, bh_px = width / 2, height / 2, width / 10, height / 10

    # 前向映射到画布并归一化（transform 的标签变换）
    cx_n = (cx_px * ratio + offset_x) / size
    cy_n = (cy_px * ratio + offset_y) / size
    w_n = (bw_px * ratio) / size
    h_n = (bh_px * ratio) / size

    logits = torch.full((1, 12), -30.0)
    logits[0, 3] = 10.0
    boxes = torch.tensor([[cx_n, cy_n, w_n, h_n]])
    # threshold 取 0.5：其余 11 类的 sigmoid(-30)≈9.4e-14 会被正确过滤，只留下目标类。
    lines = output_rows(logits, boxes, geometry, size, threshold=0.5)
    assert len(lines) == 1
    fields = [float(v) for v in lines[0].split()]
    assert int(fields[0]) == 3
    assert fields[1] == pytest.approx(cx_px / width, abs=1e-6)
    assert fields[2] == pytest.approx(cy_px / height, abs=1e-6)
    assert fields[3] == pytest.approx(bw_px / width, abs=1e-6)
    assert fields[4] == pytest.approx(bh_px / height, abs=1e-6)


@pytest.mark.parametrize("size", SIZES)
def test_canvas_is_divisible_by_backbone_block(size):
    """骨干要求输入两维能被 patch_size × num_windows = 16 × 2 = 32 整除。"""
    assert size % 32 == 0, f"{size} 不能被 32 整除，骨干前向会触发断言"
