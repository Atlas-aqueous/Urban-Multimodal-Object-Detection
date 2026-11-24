"""可视化：把预测框与真值框画回图像，肉眼检查检测效果。

用法（项目根目录执行）:
  .venv/Scripts/python.exe src/visualize.py --pred-dir submit_txt \
      --image-dir datasets/sample-a/images/val \
      --label-dir datasets/sample-a/labels/val --out-dir vis --conf 0.3

约定：绿框=真值(GT)，红框=预测(带置信度)；5通道TIFF取前3通道(BGR)显示；
3通道合成图 [R,IR,Depth] 按伪彩直接显示。输出为 JPG，长边缩到 1280 方便查看。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prepare_data import NAMES  # noqa: E402
from src.predict import load_image  # noqa: E402

MAX_SIDE = 1280
CLASS_COLORS = [  # 按类别着色（BGR），画预测框用
    (60, 76, 231), (74, 195, 139), (0, 242, 255), (200, 172, 255),
    (113, 204, 114), (222, 196, 90), (92, 158, 46), (21, 0, 136),
    (33, 33, 33), (0, 118, 255), (163, 78, 0), (0, 68, 255),
]


def to_display(im: np.ndarray) -> np.ndarray:
    """任意通道数的图转为可显示的 3 通道 uint8。"""
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] >= 5:        # 5通道磁盘为 RGB+IR+D，OpenCV 显示需转回 BGR
        return im[:, :, 2::-1].copy()
    return im[:, :, :3].copy()  # 3通道合成图按伪彩显示


def draw_boxes(im: np.ndarray, lines, color_fn, thickness: int, with_conf: bool):
    """在图上画一组框。lines: [(cls,cx,cy,w,h[,conf])] 归一化坐标。"""
    h, w = im.shape[:2]
    for row in lines:
        cls = int(row[0])
        cx, cy, bw, bh = (float(v) for v in row[1:5])
        conf = float(row[5]) if with_conf and len(row) > 5 else None
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = color_fn(cls)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = NAMES.get(cls, str(cls))
        if conf is not None:
            label += f" {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(im, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def read_lines(txt: Path):
    rows = []
    if txt.exists():
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(x) for x in parts[:6]])
                except ValueError:
                    continue
    return rows


def main(args: argparse.Namespace) -> Path:
    pred_dir, image_dir = Path(args.pred_dir), Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(args.label_dir) if args.label_dir else None

    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    imgs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not imgs:
        raise RuntimeError(f"目录无图像: {image_dir}")

    n_gt = n_pred = 0
    for i, p in enumerate(imgs, 1):
        im = to_display(load_image(p))
        gt = read_lines(label_dir / f"{p.stem}.txt") if label_dir else []
        pred = [r for r in read_lines(pred_dir / f"{p.stem}.txt")
                if len(r) > 5 and r[5] >= args.conf]
        if gt:    # 真值统一绿色、细线
            draw_boxes(im, gt, lambda c: (0, 200, 0), 2, with_conf=False)
        if pred:  # 预测按类别色、粗线
            draw_boxes(im, pred, lambda c: CLASS_COLORS[c % len(CLASS_COLORS)], 2, with_conf=True)
        n_gt += len(gt)
        n_pred += len(pred)
        scale = MAX_SIDE / max(im.shape[:2])
        if scale < 1:
            im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), im)
        if i % 50 == 0 or i == len(imgs):
            print(f"[vis] {i}/{len(imgs)} 张（GT {n_gt} 框 | 预测(conf≥{args.conf}) {n_pred} 框）")
    print(f"[vis] 完成 → {out_dir}（绿=真值 彩=预测）")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, type=Path, help="predict.py 的 TXT 输出目录")
    ap.add_argument("--image-dir", required=True, type=Path, help="图像目录（tiff/png 均可）")
    ap.add_argument("--label-dir", default=None, type=Path, help="真值标签目录（可选，画绿框）")
    ap.add_argument("--out-dir", default="vis", type=Path)
    ap.add_argument("--conf", default=0.3, type=float, help="预测框显示阈值")
    main(ap.parse_args())
