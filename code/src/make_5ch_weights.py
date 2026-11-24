"""把 3 通道 COCO 权重迁移到 5/6 通道、P3/P2 YOLO 模型。

用法: python src/make_5ch_weights.py [--cfg configs/yolov8m-5ch.yaml] [--out runs/pretrain/yolov8m-5ch.pt]
"""
import argparse
from pathlib import Path

import torch

from ultralytics import YOLO  # noqa: F401  触发 yolov8m.pt 按需下载
from ultralytics.nn.tasks import DetectionModel


def surgery_conv1(src_w: torch.Tensor, n_ch: int) -> torch.Tensor:
    """(out, 3, k, k) → (out, n_ch, k, k)：前3通道复制，其余用通道均值填充。"""
    out_c, _, kh, kw = src_w.shape
    dst = torch.zeros(out_c, n_ch, kh, kw, dtype=src_w.dtype)
    dst[:, :3] = src_w
    if n_ch > 3:
        dst[:, 3:] = src_w.mean(dim=1, keepdim=True)
    return dst


def _layer_index(key: str) -> int | None:
    parts = key.split(".")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() else None


def transfer_state(src, model, channels: int, p2: bool) -> tuple[list[str], list[str], int]:
    """按名称和形状迁移；P2 仅迁移语义仍一致的 backbone 与首段 PAN(0..15)。"""
    csd = src.float().state_dict()
    csd["model.0.conv.weight"] = surgery_conv1(csd["model.0.conv.weight"], channels)
    msd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in csd.items():
        layer = _layer_index(key)
        if p2 and layer is not None and layer > 15:
            skipped.append(key)
            continue
        if key not in msd or value.shape != msd[key].shape:
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return list(missing), list(unexpected), len(filtered)


def main(args: argparse.Namespace) -> Path:
    source = str(getattr(args, "source", "yolov8m.pt"))
    channels = int(getattr(args, "channels", 5))
    if channels not in (5, 6):
        raise ValueError("--channels 仅支持 5 或 6")
    src = YOLO(source).model

    model = DetectionModel(args.cfg, ch=channels, nc=12)
    p2 = len(model.model[-1].f) == 4
    missing, unexpected, n_loaded = transfer_state(src, model, channels, p2)
    # 头部(nc变化)属预期缺失；conv1 必须已迁移
    assert "model.0.conv.weight" not in missing, f"conv1 迁移失败: missing={missing}"
    assert torch.allclose(model.model[0].conv.weight[:, :3],
                          src.state_dict()["model.0.conv.weight"][:, :3])
    assert model.model[0].conv.weight.shape[1] == channels
    assert not unexpected

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.half(), "train_args": {}}, out)
    print(f"[surgery] source={source} channels={channels} p2={p2} "
          f"loaded_keys={n_loaded} missing={len(missing)} → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/yolov8m-5ch.yaml", type=Path)
    ap.add_argument("--out", default="runs/pretrain/yolov8m-5ch.pt", type=Path)
    ap.add_argument("--source", default="yolov8m.pt", help="3通道 COCO 权重")
    ap.add_argument("--channels", default=5, choices=[5, 6], type=int)
    main(ap.parse_args())
