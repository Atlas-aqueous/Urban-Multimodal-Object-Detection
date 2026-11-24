"""零训练成本后处理扫描：在已导出的 TXT 上重扫 conf x NMS，同时给出双口径分数。

背景
----
`src/trimodal_rfdetr.output_rows` 先做「全局 top-100 按分数截断」，再按 conf 阈值过滤。
因此提高 conf 阈值的离线重过滤与「用更高 conf 重新推理」得到的集合完全相同；
NMS 只是删除框，同样是纯后处理。所以本扫描不需要 GPU、不需要模型、不需要重新推理。

口径
----
同时报告两个指标，均由 `src.evaluate_txt.score_records` 原样计算，避免任何指标漂移：
- `rule_map`        ：细则复刻（COCO/pycocotools 式 101 点包络），**未经官方分标定**
- `ultralytics_map` ：旧工具算法（梯形积分 + Ultralytics 并列排序），**经官方分标定**

标定证据（用户反馈的真实官榜分数）：
- E0：本地 55.282 -> 官方 55.1850（偏差 -0.10）
- E9：本地 54.5820 -> 官方 55.5370（偏差 +0.96）
两个提交包均在一个点以内命中，故以 `ultralytics_map` 作为选参主指标。

本脚本只读既有预测与标注，只写指定的输出目录。
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.check_submit import clean_line  # noqa: E402
from src.evaluate_txt import overlaps, score_records, xyxy  # noqa: E402
from src.trimodal_data import load_manifest, read_image, read_labels  # noqa: E402

CACHE = {}


def load_predictions(manifest_path, prediction_dir, side="val"):
    """读入一个预测目录，返回 stem -> (gt, pred, h, w)。只读。"""
    _, samples = load_manifest(manifest_path, side)
    prediction_dir = Path(prediction_dir)
    expected = {r["stem"] for r in samples}
    actual = {p.stem for p in prediction_dir.glob("*.txt")}
    if expected != actual:
        raise ValueError(f"{prediction_dir}: 预测名不匹配 missing={len(expected - actual)} "
                         f"extra={len(actual - expected)}")
    out = {}
    invalid = 0
    for r in samples:
        rows = []
        for line in (prediction_dir / (r["stem"] + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            clean = clean_line(line.split())
            if clean is None:
                # 细则：非法类别/坐标/置信度的预测无效，不参与结果运算 → 丢弃而非报错。
                invalid += 1
                continue
            rows.append(clean)
        pred = np.array(rows, dtype=float).reshape(-1, 6)
        # 按分数降序，与官方式选框一致；后续所有阈值都从这个有序集合里取子集。
        pred = pred[np.argsort(-pred[:, 5], kind="stable")]
        gt = read_labels(r["label"]).astype(float)
        h, w = read_image(Path(r["visible"])).shape[:2]
        out[r["stem"]] = (gt, pred, h, w)
    if invalid:
        print(f"[load] {prediction_dir}: 丢弃 {invalid} 行非法预测（细则：无效不参与运算）", flush=True)
    return out


def nms_classwise(pred, iou_threshold):
    """类别独立贪心 NMS，与旧路线 agnostic_nms=False 一致。纯 numpy。"""
    if iou_threshold is None or len(pred) == 0:
        return pred
    kept = []
    for cls in np.unique(pred[:, 0]):
        rows = pred[pred[:, 0] == cls]
        boxes = xyxy(rows)
        suppressed = np.zeros(len(rows), dtype=bool)
        selection = []
        for i in range(len(rows)):
            if suppressed[i]:
                continue
            selection.append(i)
            if i + 1 < len(rows):
                iou = overlaps(boxes[i:i + 1], boxes[i + 1:])[0]
                suppressed[i + 1:] |= iou > iou_threshold
        kept.append(rows[selection])
    return np.concatenate(kept) if kept else np.zeros((0, 6))


def score_one(task):
    """在指定目录上评估一个 (conf, nms) 组合。CACHE 由 fork 继承。"""
    name, conf, nms_iou = task
    records = []
    for stem, (gt, pred, h, w) in CACHE[name].items():
        rows = pred[pred[:, 5] >= conf]
        rows = nms_classwise(rows, nms_iou)
        records.append((stem, gt, rows, h, w))
    report = score_records(records)
    return {"dir": name, "conf": conf, "nms_iou": nms_iou,
            "rule_map": report["rule_map"], "ultralytics_map": report["ultralytics_map"],
            "rule_map50": report["rule_map50"], "ultralytics_map50": report["ultralytics_map50"],
            "tiny_recall50": report["tiny_recall50"], "at_max_det": report["at_max_det"],
            "rule_per_class": report["rule_per_class"],
            "ultralytics_per_class": report["ultralytics_per_class"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--out", default="runs/formal/rf48/postprocess-sweep")
    ap.add_argument("--conf", default="0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7")
    ap.add_argument("--iou", default="none,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dirs", default="mm-960=runs/formal/rf48/mm-val-960,"
                                     "mm-768=runs/formal/rf48/mm-val-768,"
                                     "e0-selected=runs/formal/rf48/baselines/e0-selected,"
                                     "e9-selected=runs/formal/rf48/baselines/e9-selected,"
                                     "e9-low=runs/formal/rf48/baselines/e9-low,"
                                     "e12-selected=runs/formal/rf48/baselines/e12-selected")
    args = ap.parse_args()
    os.chdir(ROOT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in args.dirs.split(","):
        name, _, rel = item.partition("=")
        path = Path(rel)
        if not path.exists():
            print(f"[skip] {name}: {path} 不存在", flush=True)
            continue
        CACHE[name] = load_predictions(args.manifest, path)
        peak = max((len(v[1]) for v in CACHE[name].values()), default=0)
        print(f"[load] {name:16} images={len(CACHE[name]):4d} 单图最多框数={peak}", flush=True)

    confs = [float(x) for x in args.conf.split(",") if x.strip()]
    ious = [None if x.strip() == "none" else float(x) for x in args.iou.split(",") if x.strip()]
    tasks = [(name, conf, iou) for name in CACHE for conf in confs for iou in ious]
    print(f"[grid] {len(CACHE)} 个目录 x {len(confs)} conf x {len(ious)} NMS = {len(tasks)} 次评估", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(score_one, tasks, chunksize=1), 1):
            results.append(row)
            if i % 50 == 0 or i == len(tasks):
                print(f"  ...{i}/{len(tasks)}", flush=True)

    payload = {"manifest": args.manifest, "confs": confs, "nms_ious": ious,
               "metric_calibration": {"note": "ultralytics_map 经过官方分标定，rule_map 未标定",
                                      "e0": {"local": 55.282, "official": 55.1850},
                                      "e9": {"local": 54.5820, "official": 55.5370}},
               "results": results}
    (out_dir / "sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

    print("\n=== 每个目录的最优配置（按已标定的 ultralytics_map 排序）===")
    print(f"{'dir':16} {'best_ul':>8} {'conf':>6} {'nms':>6} {'rule@best':>10} "
          f"{'rule_best':>10} {'conf':>6} {'nms':>6}")
    summary = {}
    for name in CACHE:
        rows = [r for r in results if r["dir"] == name]
        best_ul = max(rows, key=lambda r: r["ultralytics_map"])
        best_rule = max(rows, key=lambda r: r["rule_map"])
        summary[name] = {"best_ultralytics": best_ul, "best_rule": best_rule}
        print(f"{name:16} {best_ul['ultralytics_map'] * 100:>8.4f} {best_ul['conf']:>6} "
              f"{str(best_ul['nms_iou']):>6} {best_ul['rule_map'] * 100:>10.4f} "
              f"{best_rule['rule_map'] * 100:>10.4f} {best_rule['conf']:>6} "
              f"{str(best_rule['nms_iou']):>6}")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(f"\n[out] {out_dir}/sweep.json, summary.json")


if __name__ == "__main__":
    main()
