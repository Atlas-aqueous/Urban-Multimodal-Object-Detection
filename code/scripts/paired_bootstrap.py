"""配对自助法：在同一批验证图上比较两个候选，给出差值分布与"更优概率"。

为什么必须配对：`bootstrap_val_noise.py` 测得的 ±1.93 分是**单次测量的边缘标准误**，
主要来自"这 195 张图好不好测"的图像难度方差。比较两个候选时该方差大部分抵消，
真正决定判断的是**差值**的标准误。用边缘标准误去做决策会严重低估置信度。

    python -m scripts.paired_bootstrap \
        --a runs/formal/rf48/mm-val-768 --a-conf 0.5 --a-nms 0.6 \
        --b runs/formal/rf48/baselines/e0-selected --b-conf 0.5 --b-nms 0.4 \
        --iterations 400
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sweep_rf48_postprocess import load_predictions, nms_classwise  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402

CACHE = {}


def prepare(manifest, predictions, side, conf, nms):
    loaded = load_predictions(manifest, predictions, side)
    prepared = {}
    for stem, (gt, pred, h, w) in loaded.items():
        rows = pred[pred[:, 5] >= conf]
        prepared[stem] = (stem, gt, nms_classwise(rows, nms), h, w)
    return prepared


def score(records):
    return score_records(records, require_all_classes=False)


def one(seed):
    rng = np.random.default_rng(seed)
    stems = CACHE["stems"]
    index = rng.integers(0, len(stems), len(stems))
    picked = [stems[i] for i in index]
    left = [CACHE["a"][s] for s in picked]
    right = [CACHE["b"][s] for s in picked]
    a = score(left)
    b = score(right)
    return {"a_ul": a["ultralytics_map"], "b_ul": b["ultralytics_map"],
            "a_rule": a["rule_map"], "b_rule": b["rule_map"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--side", default="val")
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-conf", type=float, default=0.5)
    ap.add_argument("--a-nms", type=float, default=None)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-conf", type=float, default=0.5)
    ap.add_argument("--b-nms", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out")
    args = ap.parse_args()

    CACHE["a"] = prepare(args.manifest, args.a, args.side, args.a_conf, args.a_nms)
    CACHE["b"] = prepare(args.manifest, args.b, args.side, args.b_conf, args.b_nms)
    CACHE["stems"] = sorted(set(CACHE["a"]) & set(CACHE["b"]))
    assert CACHE["stems"], "两个预测目录没有共同的图像"

    point_a = score([CACHE["a"][s] for s in CACHE["stems"]])
    point_b = score([CACHE["b"][s] for s in CACHE["stems"]])
    print(f"[paired] A={args.a} conf={args.a_conf} nms={args.a_nms} -> "
          f"ul {point_a['ultralytics_map']*100:.4f} / rule {point_a['rule_map']*100:.4f}")
    print(f"[paired] B={args.b} conf={args.b_conf} nms={args.b_nms} -> "
          f"ul {point_b['ultralytics_map']*100:.4f} / rule {point_b['rule_map']*100:.4f}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, range(args.iterations), chunksize=4))
    diff_ul = (np.array([r["a_ul"] for r in rows]) - np.array([r["b_ul"] for r in rows])) * 100
    diff_rule = (np.array([r["a_rule"] for r in rows]) - np.array([r["b_rule"] for r in rows])) * 100
    summary = {
        "images": len(CACHE["stems"]), "iterations": args.iterations,
        "a": {"dir": args.a, "conf": args.a_conf, "nms": args.a_nms,
              "ultralytics_map": point_a["ultralytics_map"], "rule_map": point_a["rule_map"]},
        "b": {"dir": args.b, "conf": args.b_conf, "nms": args.b_nms,
              "ultralytics_map": point_b["ultralytics_map"], "rule_map": point_b["rule_map"]},
        "paired_difference_ultralytics": {
            "mean": float(diff_ul.mean()), "std": float(diff_ul.std(ddof=1)),
            "p2.5": float(np.percentile(diff_ul, 2.5)),
            "p97.5": float(np.percentile(diff_ul, 97.5)),
            "probability_a_better": float((diff_ul > 0).mean())},
        "paired_difference_rule": {
            "mean": float(diff_rule.mean()), "std": float(diff_rule.std(ddof=1)),
            "probability_a_better": float((diff_rule > 0).mean())},
    }
    print(f"[paired] ultralytics 差值 = {diff_ul.mean():+.4f} ± {diff_ul.std(ddof=1):.4f} 分，"
          f"P(A 更优) = {(diff_ul > 0).mean()*100:.1f}%")
    print(f"[paired] rule        差值 = {diff_rule.mean():+.4f} ± {diff_rule.std(ddof=1):.4f} 分，"
          f"P(A 更优) = {(diff_rule > 0).mean()*100:.1f}%")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
