"""按目标尺寸分层计算 AP，用于把剩余误差定位到"小目标分辨率"还是"特征/融合"。

做法：把 GT 按原图像素面积分档，逐档过滤 GT 后重跑匹配与积分（其余档的真实框不计为漏检，
但落在其他档的预测仍算 FP —— 与 COCO 的 areaRng 分层思路一致）。
预测保持原样，不重新推理。

    python -m scripts.stratified_ap --predictions runs/formal/rf48/mm-val-768 --conf 0.5 --nms 0.6
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sweep_rf48_postprocess import load_predictions, nms_classwise  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402

# 以"短边等效像素"为界：面积阈值 = side^2
STRATA = [("side <16px", 0, 16 ** 2),
          ("16-32px", 16 ** 2, 32 ** 2),
          ("32-64px", 32 ** 2, 64 ** 2),
          ("64-128px", 64 ** 2, 128 ** 2),
          ("side >=128px", 128 ** 2, float("inf"))]


def areas(gt, h, w):
    return gt[:, 3] * w * gt[:, 4] * h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--side", default="val")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.6)
    args = ap.parse_args()

    loaded = load_predictions(args.manifest, args.predictions, args.side)
    records = []
    for stem, (gt, pred, h, w) in loaded.items():
        rows = pred[pred[:, 5] >= args.conf]
        records.append((stem, gt, nms_classwise(rows, args.nms), h, w))

    total = sum(len(gt) for _, gt, _, _, _ in records)
    whole = score_records(records, require_all_classes=False)
    print(f"预测目录 {args.predictions}  conf>={args.conf}  nms={args.nms}")
    print(f"全量: ultralytics={whole['ultralytics_map']*100:.4f} "
          f"rule={whole['rule_map']*100:.4f} | mAP50(ul)={whole['ultralytics_map50']*100:.2f} "
          f"| 目标数={total}\n")
    print(f"{'尺寸档':>13} {'目标数':>7} {'占比':>7} {'ul AP':>8} {'ul AP50':>8} {'rule AP':>8} {'框数':>7}")
    for name, low, high in STRATA:
        subset, kept, kept_pred = [], 0, 0
        for stem, gt, pred, h, w in records:
            a_gt = areas(gt, h, w) if len(gt) else np.zeros(0)
            a_pred = areas(pred, h, w) if len(pred) else np.zeros(0)
            # COCO areaRng 语义：GT 与预测框都用同一面积区间过滤，避免跨档误判为 FP。
            gt_mask = (a_gt >= low) & (a_gt < high)
            pred_mask = (a_pred >= low) & (a_pred < high)
            subset.append((stem, gt[gt_mask], pred[pred_mask], h, w))
            kept += int(gt_mask.sum())
            kept_pred += int(pred_mask.sum())
        if kept == 0:
            print(f"{name:>13} {0:>7} {'—':>7}")
            continue
        report = score_records(subset, require_all_classes=False)
        print(f"{name:>13} {kept:>7} {kept/total*100:>6.1f}% "
              f"{report['ultralytics_map']*100:>8.3f} {report['ultralytics_map50']*100:>8.2f} "
              f"{report['rule_map']*100:>8.3f} {kept_pred:>7}")
    print("\n读法：COCO areaRng 语义下各档 AP 可直接横向比较。"
          "若小尺寸档显著低于大尺寸档，瓶颈在分辨率/小目标；若各档接近，瓶颈更可能在特征或融合。")


if __name__ == "__main__":
    main()
