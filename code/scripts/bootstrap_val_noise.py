"""自助法估计 A 验证集（195 张）上的指标抽样噪声。

动机：我们一直在 195 张图上看 0.1~0.5 分的差异并据此选参，但从未量化这些差异是否
超出抽样噪声。本脚本固定同一批预测，只对图像做有放回重采样，得到指标的标准误，
从而给出"多大增量才算真信号"的判据。

    python -m scripts.bootstrap_val_noise --predictions runs/formal/rf48/mm-val-960 \
        --manifest runs/formal/rf48/train-manifest.json --conf 0.5 --iterations 400
"""
import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sweep_rf48_postprocess import load_predictions  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402

CACHE = {}


def one(seed):
    rng = np.random.default_rng(seed)
    records = list(CACHE["records"])
    index = rng.integers(0, len(records), len(records))
    sample = [records[i] for i in index]
    report = score_records(sample, require_all_classes=False)
    return {"ultralytics_map": report["ultralytics_map"], "rule_map": report["rule_map"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--side", default="val")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = load_predictions(args.manifest, args.predictions, args.side)
    CACHE["records"] = [(stem, gt, pred[pred[:, 5] >= args.conf], h, w)
                        for stem, (gt, pred, h, w) in loaded.items()]
    print(f"[bootstrap] {len(CACHE['records'])} 张图，conf>={args.conf}，"
          f"{args.iterations} 次重采样", flush=True)
    full = score_records(CACHE["records"], require_all_classes=False)
    print(f"[bootstrap] 全量点估计: ultralytics={full['ultralytics_map']*100:.4f} "
          f"rule={full['rule_map']*100:.4f}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, range(args.iterations), chunksize=4))
    ul = np.array([r["ultralytics_map"] for r in rows]) * 100
    rule = np.array([r["rule_map"] for r in rows]) * 100
    summary = {
        "predictions": args.predictions, "conf": args.conf, "images": len(CACHE["records"]),
        "iterations": args.iterations,
        "point_estimate": {"ultralytics_map": full["ultralytics_map"],
                           "rule_map": full["rule_map"]},
        "bootstrap": {
            "ultralytics": {"mean": float(ul.mean()), "std": float(ul.std(ddof=1)),
                            "p2.5": float(np.percentile(ul, 2.5)),
                            "p97.5": float(np.percentile(ul, 97.5))},
            "rule": {"mean": float(rule.mean()), "std": float(rule.std(ddof=1)),
                     "p2.5": float(np.percentile(rule, 2.5)),
                     "p97.5": float(np.percentile(rule, 97.5))},
        },
        "interpretation": "std 即 195 张 A 验证集上的单次测量标准误；"
                          "两个候选的差值需显著大于该量级才可判为真实提升。",
    }
    print(json.dumps(summary["bootstrap"], ensure_ascii=False, indent=1))
    print(f"[bootstrap] ultralytics 标准误 = {ul.std(ddof=1):.4f} 分；"
          f"95% 区间宽 = {np.percentile(ul, 97.5) - np.percentile(ul, 2.5):.4f} 分", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
