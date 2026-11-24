"""从 YOLO 验证预测计算按目标面积分组的 IoU@0.5 recall。"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_boxes(path: Path, prediction: bool) -> list[tuple[int, np.ndarray, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        values = list(map(float, line.split()))
        if len(values) < 5:
            continue
        cls, cx, cy, width, height = int(values[0]), *values[1:5]
        box = np.array([cx - width / 2, cy - height / 2,
                        cx + width / 2, cy + height / 2], dtype=np.float32)
        conf = values[5] if prediction and len(values) > 5 else 1.0
        rows.append((cls, box, conf))
    return rows


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
            max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def area_bin(box: np.ndarray) -> str:
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    return "tiny" if area < 0.001 else "medium" if area < 0.01 else "large"


def size_recall(label_dir: Path, prediction_dir: Path, threshold: float = 0.5) -> dict:
    total, matched = Counter(), Counter()
    prediction_counts = []
    for label_path in sorted(Path(label_dir).glob("*.txt")):
        ground_truth = read_boxes(label_path, prediction=False)
        predictions = sorted(
            read_boxes(Path(prediction_dir) / label_path.name, prediction=True),
            key=lambda row: row[2], reverse=True,
        )
        prediction_counts.append(len(predictions))
        used = set()
        for cls, gt_box, _ in ground_truth:
            size = area_bin(gt_box)
            total[size] += 1
            candidates = [(iou(gt_box, box), index) for index, (pred_cls, box, _) in enumerate(predictions)
                          if pred_cls == cls and index not in used]
            if candidates:
                best_iou, index = max(candidates)
                if best_iou >= threshold:
                    matched[size] += 1
                    used.add(index)
    return {
        "iou_threshold": threshold,
        "recall": {size: matched[size] / total[size] if total[size] else None
                   for size in ("tiny", "medium", "large")},
        "targets": dict(total),
        "matched": dict(matched),
        "prediction_files": len(prediction_counts),
        "at_100_predictions": sum(count >= 100 for count in prediction_counts),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", required=True, type=Path)
    ap.add_argument("--prediction-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = size_recall(args.label_dir, args.prediction_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
