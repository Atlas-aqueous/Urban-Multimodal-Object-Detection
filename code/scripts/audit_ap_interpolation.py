"""Audit cached validation TXT with confidence-greedy matches and two AP integrators.

This is a diagnostic, not the unpublished competition scorer. It does not rerun
inference or NMS. Thresholds filter the same cached detections, isolating AP tail
interpolation. Normalized xyxy IoU is invariant to image width/height scaling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def boxes(path, columns):
    if not path.exists() or not path.read_text().strip():
        return np.empty((0, columns), dtype=np.float64)
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != columns or not np.isfinite(data).all():
        raise ValueError(f"Invalid TXT: {path}")
    if np.any(data[:, 0] != np.floor(data[:, 0])) or np.any((data[:, 0] < 0) | (data[:, 0] > 11)):
        raise ValueError(f"Invalid class: {path}")
    if columns == 5 and np.any(data[:, 3:5] <= 0):
        raise ValueError(f"Invalid box: {path}")
    center, size = data[:, 1:3].copy(), data[:, 3:5].copy()
    data[:, 1:3], data[:, 3:5] = center - size / 2, center + size / 2
    return data


def integrations(tp, targets):
    """Return rule-style 101 precision mean and Ultralytics-style linear trapz."""
    if len(tp) == 0:
        return 0.0, 0.0
    recall = np.cumsum(tp) / targets
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, grid, side="left")
    sample = np.zeros(101)
    valid = indices < len(envelope)
    sample[valid] = envelope[indices[valid]]
    mrec = np.r_[0, recall, 1]
    mpre = np.maximum.accumulate(np.r_[1, precision, 0][::-1])[::-1]
    return float(sample.mean()), float(np.trapezoid(np.interp(grid, mrec, mpre), grid))


def match_image(gt, pred):
    pred = pred[np.argsort(-pred[:, 5], kind="stable")][:100]
    tp = np.zeros((len(pred), 10), dtype=bool)
    thresholds = np.linspace(0.5, 0.95, 10)
    used = np.zeros((10, len(gt)), dtype=bool)
    for i, row in enumerate(pred):
        if not len(gt):
            continue
        wh = np.maximum(0, np.minimum(row[3:5], gt[:, 3:5]) - np.maximum(row[1:3], gt[:, 1:3]))
        inter = wh.prod(1)
        union = (row[3:5] - row[1:3]).prod() + (gt[:, 3:5] - gt[:, 1:3]).prod(1) - inter
        overlaps = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        for t, threshold in enumerate(thresholds):
            eligible = (gt[:, 0] == row[0]) & ~used[t] & (overlaps >= threshold)
            if eligible.any():
                j = int(np.argmax(np.where(eligible, overlaps, -1)))
                used[t, j] = True
                tp[i, t] = True
    return pred[:, [0, 5]], tp


def audit(label_dir, prediction_dir):
    counts = np.zeros(12, dtype=int)
    meta, matches = [], []
    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        raise ValueError("Empty ground truth")
    fingerprint = hashlib.sha256()
    missing = []
    degenerate = 0
    for path in labels:
        gt = boxes(path, 5)
        prediction = prediction_dir / path.name
        if not prediction.exists():
            missing.append(path.name)
        fingerprint.update(path.name.encode() + b"\0" + path.read_bytes())
        fingerprint.update(prediction.read_bytes() if prediction.exists() else b"<empty>")
        counts += np.bincount(gt[:, 0].astype(int), minlength=12)
        pred = boxes(prediction, 6)
        valid = np.all(pred[:, 3:5] > pred[:, 1:3], axis=1)
        degenerate += int((~valid).sum())
        m, tp = match_image(gt, pred[valid])
        meta.append(m)
        matches.append(tp)
    if (counts == 0).any():
        raise ValueError("All 12 classes are required for comparable macro AP")
    meta, matches = np.concatenate(meta), np.concatenate(matches)
    order = np.argsort(-meta[:, 1], kind="stable")
    meta, matches = meta[order], matches[order]
    results = []
    for threshold in [0.005, 0.05, 0.10, 0.25, 0.35, 0.40, 0.45]:
        ap = np.zeros((12, 10, 2))
        for cls in range(12):
            selected = matches[(meta[:, 0] == cls) & (meta[:, 1] >= threshold)]
            for t in range(10):
                ap[cls, t] = integrations(selected[:, t], counts[cls])
        results.append({
            "conf_filter": threshold,
            "predictions": int((meta[:, 1] >= threshold).sum()),
            "rule_style_map": float(ap[:, :, 0].mean()),
            "linear_trapz_map_same_matches": float(ap[:, :, 1].mean()),
            "per_class_rule_ap": ap[:, :, 0].mean(1).tolist(),
        })
    return {"prediction_dir": str(prediction_dir), "images": len(labels),
            "gt_per_class": counts.tolist(), "missing_prediction_files_treated_as_empty": missing,
            "inputs_sha256": fingerprint.hexdigest(), "degenerate_predictions_dropped": degenerate,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    prefix = integrations(np.array([1]), 2)
    tail = integrations(np.r_[1, np.zeros(20)], 2)
    assert np.isclose(prefix[0], 51 / 101) and np.isclose(prefix[0], tail[0])
    assert prefix[1] > tail[1] + 0.2
    payload = {
        "scope": "cached validation predictions only; no NMS rerun; not official hidden-test scores",
        "matching": "confidence-greedy, highest eligible IoU, stable ties, top100 per image",
        "recall_comparison": ">= (COCO convention); official prose has an ambiguous > symbol",
        "synthetic_1tp_2gt": {"prefix_rule_trapz": prefix, "with_20_tail_fp_rule_trapz": tail},
        "runs": [audit(args.labels, p) for p in args.predictions],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for run in payload["runs"]:
        print(run["prediction_dir"], "images", run["images"], "gt", sum(run["gt_per_class"]))
        for row in run["results"]:
            print(row["conf_filter"], row["predictions"],
                  round(100 * row["rule_style_map"], 4),
                  round(100 * row["linear_trapz_map_same_matches"], 4))


if __name__ == "__main__":
    main()
