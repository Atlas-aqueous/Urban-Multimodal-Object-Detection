"""量化「不过滤置信度、直接交满 100 框」在这个评测口径下会掉多少分。

背景：对照版本 C 的 1000 张图里有 955 张恰好 100 框，
最低置信度 0.0052 —— 典型的「原始候选未过滤」。本脚本在已知真值的 A 验证集上
复现同样的做法，给出确切的分差。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trimodal_data import load_manifest  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402
from scripts.sweep_rf48_postprocess import load_predictions, nms_classwise  # noqa: E402

MANIFEST = "runs/formal/rf48/train-manifest.json"
VAL_RAW = "runs/formal/rf48b/final-run-early1280/val-1280"
loaded = load_predictions(ROOT / MANIFEST, ROOT / VAL_RAW, "val")

print(f"A 验证集 {len(loaded)} 图\n")
print(f"{'做法':<34}{'mAP':>9}{'相对0.35':>10}{'框/图':>8}")

CONFIGS = [
    ("conf>=0.35，不设上限（我们的正式做法）", 0.35, None),
    ("conf>=0.40", 0.40, None),
    ("conf>=0.05", 0.05, None),
    ("conf>=0.005，每图上限100框（对照版本 C）", 0.005, 100),
    ("conf>=0.005，不设上限", 0.005, None),
]

base = None
for label, conf, cap in CONFIGS:
    recs, nb = [], 0
    for stem, (gt, pred, h, w) in loaded.items():
        keep = pred[pred[:, 5] >= conf]
        kept = nms_classwise(keep, 0.6)
        if cap is not None and len(kept) > cap:
            kept = kept[np.argsort(-kept[:, 5], kind="stable")[:cap]]
        nb += len(kept)
        recs.append((stem, gt, kept, h, w))
    m = score_records(recs, require_all_classes=False)["ultralytics_map"]
    if base is None:
        base = m
    print(f"{label:<34}{m*100:>9.4f}{(m-base)*100:>+10.4f}{nb/len(loaded):>8.1f}")

print()
print("参考：正式提交的 ep67 模型（conf 0.35）基线评测分 60.8060、评测分 54.819")
