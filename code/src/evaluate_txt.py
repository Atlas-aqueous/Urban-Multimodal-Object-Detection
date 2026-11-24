"""Strict exported-TXT scoring. Rule reconstruction is NOT the hidden server score."""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from src.check_submit import clean_line
from src.trimodal_data import load_manifest, read_labels, read_image, write_json

THRESHOLDS = np.linspace(.5, .95, 10)


def xyxy(rows):
    return np.concatenate((rows[:, 1:3] - rows[:, 3:5] / 2,
                           rows[:, 1:3] + rows[:, 3:5] / 2), axis=1)


def overlaps(a, b):
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(gt, pred):
    ious = overlaps(xyxy(gt), xyxy(pred))
    ious *= gt[:, 0, None] == pred[None, :, 0]
    matched = np.full((len(pred), 10), -1, dtype=int)
    old = np.zeros((len(pred), 10), dtype=bool)
    for t, threshold in enumerate(THRESHOLDS):
        used = set()
        for d in range(len(pred)):
            eligible = [g for g in range(len(gt)) if g not in used and ious[g, d] >= threshold]
            if eligible:
                # COCO replaces ties with the last equally-overlapping GT.
                g = max(eligible, key=lambda g: (ious[g, d], g))
                used.add(g)
                matched[d, t] = g
        pairs = np.array(np.nonzero(ious >= float(np.float32(threshold)))).T
        if len(pairs) > 1:
            pairs = pairs[ious[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        if len(pairs):
            old[pairs[:, 1], t] = True
    return matched, old


def integrate(tp, count, legacy=False):
    if not count:
        return None
    if not len(tp):
        return 0.
    recall = tp.cumsum() / count
    precision = tp.cumsum() / np.arange(1, len(tp) + 1)
    grid = np.linspace(0, 1, 101)
    if legacy:
        mrec = np.r_[0, recall, 1]
        mpre = np.maximum.accumulate(np.r_[1, precision, 0][::-1])[::-1]
        return float(np.trapezoid(np.interp(grid, mrec, mpre), grid))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ids = np.searchsorted(recall, grid, side="left")
    return float(np.r_[envelope, 0][ids].mean())


def score_records(records, require_all_classes=True):
    count = np.zeros(12, int)
    meta, correct, old_correct = [], [], []
    tiny_gt = tiny_hit = capped = 0
    for _, gt, pred, h, w in records:
        pred = pred[np.argsort(-pred[:, 5], kind="stable")][:100]
        indices, old = match(gt, pred)
        meta.append(pred[:, [0, 5]])
        correct.append(indices >= 0)
        old_correct.append(old)
        count += np.bincount(gt[:, 0].astype(int), minlength=12)
        tiny = (gt[:, 3] * w * gt[:, 4] * h) < 32**2
        tiny_gt += int(tiny.sum())
        tiny_hit += sum(bool(tiny[g]) for g in set(indices[:, 0]) if g >= 0)
        capped += int(len(pred) == 100)
    if require_all_classes and (count == 0).any():
        raise ValueError("All 12 classes required in the primary validation split")
    m, tp, old_tp = np.concatenate(meta), np.concatenate(correct), np.concatenate(old_correct)
    stable = np.argsort(-m[:, 1], kind="stable")
    legacy_order = np.argsort(-m[:, 1])  # Matches Ultralytics ap_per_class tie ordering.
    ap, old_ap = np.full((12, 10), np.nan), np.full((12, 10), np.nan)
    for c in range(12):
        if not count[c]:
            continue
        for t in range(10):
            ap[c, t] = integrate(tp[stable][m[stable, 0] == c, t], count[c])
            old_ap[c, t] = integrate(old_tp[legacy_order][m[legacy_order, 0] == c, t], count[c], True)
    return {"images": len(records), "gt_per_class": count.tolist(),
            "rule_map": float(np.nanmean(ap)), "rule_map50": float(np.nanmean(ap[:, 0])),
            "rule_per_class": [float(np.mean(a)) if np.isfinite(a).all() else None for a in ap],
            "ultralytics_map": float(np.nanmean(old_ap)),
            "ultralytics_map50": float(np.nanmean(old_ap[:, 0])),
            "ultralytics_per_class": [float(np.mean(a)) if np.isfinite(a).all() else None for a in old_ap],
            "tiny_recall50": tiny_hit / tiny_gt if tiny_gt else None,
            "tiny_gt": tiny_gt, "at_max_det": capped,
            "metric_notice": "TXT replay; rule reconstruction not verified against hidden server"}


def coco_map(records):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, annotations, detections = [], [], []
    for i, (_, gt, pred, h, w) in enumerate(records):
        images.append({"id": i, "height": h, "width": w})
        for row in gt:
            c, cx, cy, bw, bh = map(float, row)
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": int(c),
                                "bbox": [(cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h],
                                "area": bw*w*bh*h, "iscrowd": 0})
        pred = pred[np.argsort(-pred[:, 5], kind="stable")][:100]
        for c, cx, cy, bw, bh, conf in pred:
            detections.append({"image_id": i, "category_id": int(c), "score": float(conf),
                               "bbox": [float((cx-bw/2)*w), float((cy-bh/2)*h), float(bw*w), float(bh*h)]})
    if not detections:
        return 0.
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"info": {}, "images": images, "annotations": annotations,
                        "categories": [{"id": c, "name": str(c)} for c in range(12)]}
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(detections), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
    values = evaluator.eval["precision"][:, :, :, 0, 2]
    return float(values[values >= 0].mean())


def evaluate(manifest_path, prediction_dir, side="val", coco=False, threshold=0.):
    manifest, samples = load_manifest(manifest_path, side)
    prediction_dir = Path(prediction_dir)
    expected = {r["stem"] for r in samples}
    actual = {p.stem for p in prediction_dir.glob("*.txt")}
    if expected != actual:
        raise ValueError(f"Prediction names mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    records, formats = [], {"jpg": [], "png": []}
    for r in samples:
        rows = []
        for line in (prediction_dir / (r["stem"] + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            clean = clean_line(line.split())
            if clean is None:
                raise ValueError(f"Invalid prediction in {r['stem']}: {line}")
            if clean[-1] >= threshold:
                rows.append(clean)
        pred = np.array(rows, dtype=float).reshape(-1, 6)
        gt = read_labels(r["label"]).astype(float)
        h, w = read_image(Path(r["visible"])).shape[:2]
        record = (r["stem"], gt, pred, h, w)
        records.append(record)
        formats["jpg" if Path(r["depth"]).suffix.lower() in {".jpg", ".jpeg"} else "png"].append(record)
    report = score_records(records)
    report["subsets"] = {k: score_records(v, False) for k, v in formats.items() if v}
    if coco:
        report["pycocotools_map"] = coco_map(records)
        report["coco_difference"] = abs(report["rule_map"] - report["pycocotools_map"])
        if report["coco_difference"] > 1e-6:
            raise AssertionError(f"COCO disagreement: {report['coco_difference']}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coco", action="store_true")
    args = p.parse_args()
    report = evaluate(args.manifest, args.predictions, coco=args.coco)
    write_json(args.out, report)
    print(report["rule_map"], report["ultralytics_map"])
