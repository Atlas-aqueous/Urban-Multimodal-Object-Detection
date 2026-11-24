"""推理并直接产出比赛格式 TXT（自管图像加载：3ch PNG / 5ch TIFF 经 np.fromfile +
cv2.imdecode/imdecodemulti 解码，兼容中文路径与 OpenCV 5.0 多页 TIFF，
不依赖 ultralytics 数据加载器对多通道的目录支持）。

用法: python src/predict.py --weights runs/exp-a/weights/best.pt \
        --image-dir datasets/test-a/images/test --out-dir submit_txt
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_image(path: Path) -> np.ndarray:
    """HWC uint8。多页 TIFF（方案A）逐页堆叠；普通图像原样读取。

    经 np.fromfile + imdecodemulti 读取（与 ultralytics 训练加载器 monkey-patch 后的
    imread 逐字节一致，且支持非 ASCII 路径）。原生 cv2.imreadmulti 在 OpenCV 5.0 下
    将本数据的每页解成 (H,W,3)，堆叠成 4 维数组导致 letterbox cv2.resize 崩溃，禁用。
    """
    data = np.fromfile(str(path), np.uint8)
    if path.suffix.lower() in {".tiff", ".tif"}:
        ok, frames = cv2.imdecodemulti(data, cv2.IMREAD_UNCHANGED)
        if not ok:
            raise IOError(f"读取失败: {path}")
        if not frames or any(frame.ndim != 2 for frame in frames):
            raise ValueError(f"多页TIFF每页必须是单通道，实际 {[f.shape for f in frames]}")
        if len({frame.shape for frame in frames}) != 1:
            raise ValueError(f"多页TIFF各页尺寸不一致: {path}")
        return np.stack(frames, axis=2)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"读取失败: {path}")
    return im


def main(args: argparse.Namespace) -> Path:
    model = YOLO(args.weights)
    expected_channels = model.model.model[0].conv.in_channels
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(args.image_dir).iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".tif"})
    if not imgs:
        raise RuntimeError(f"目录无图像: {args.image_dir}")

    n_boxes = 0
    skipped = 0
    for i, p in enumerate(imgs, 1):
        prediction_path = out_dir / f"{p.stem}.txt"
        if args.resume and prediction_path.exists():
            n_boxes += sum(bool(line.strip()) for line in prediction_path.read_text(
                encoding="utf-8", errors="ignore").splitlines())
            skipped += 1
            continue
        im = load_image(p)
        actual_channels = 1 if im.ndim == 2 else im.shape[2]
        if actual_channels != expected_channels:
            raise ValueError(f"{p.name} 为{actual_channels}通道，但模型需要{expected_channels}通道")
        r = model(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, device=args.device, half=args.half,
                  augment=args.augment, verbose=False)[0]
        lines = []
        if r.boxes is not None and len(r.boxes):
            xywhn = r.boxes.xywhn.cpu().numpy()          # 已归一化 (cx, cy, w, h)
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for c, (cx, cy, w, h), cf in zip(cls, xywhn, conf):
                lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cf:.6f}")
        n_boxes += len(lines)
        prediction_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if i % 100 == 0 or i == len(imgs):
            print(f"[predict] {i}/{len(imgs)} 张, 累计 {n_boxes} 框")
    print(f"[predict] 完成: {len(imgs)} 张（断点跳过 {skipped} 张）→ {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--conf", default=0.001, type=float, help="低阈值保recall，交给check截断")
    ap.add_argument("--iou", default=0.7, type=float)
    ap.add_argument("--max-det", default=100, type=int)
    ap.add_argument("--device", default=None, help="如 0、cpu；默认自动选择")
    ap.add_argument("--half", action="store_true", help="GPU FP16推理")
    ap.add_argument("--augment", action="store_true", help="单模型TTA；默认关闭，使用前确认赛规")
    ap.add_argument("--resume", action="store_true", help="跳过输出目录中已有的同名TXT")
    main(ap.parse_args())
