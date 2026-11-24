"""把已导出的 TXT 预测做「conf 过滤 + 类别独立 NMS」后再落盘，可选复算分数。

为什么不需要重新推理
--------------------
`src/trimodal_rfdetr.output_rows` 的语义是「先按分数取全局 top-100，再按 conf 过滤」。
因此 `top100 ∩ {conf >= t}` 与「直接以阈值 t 推理」得到的集合完全相同；
NMS 只是删除框，不引入新框。两者都是纯后处理，不需要模型与 GPU。

NMS 逻辑与 `scripts/sweep_rf48_postprocess.nms_classwise` 共用同一实现，
保证扫描出的分数与最终成包内容一致。

用法：
    python -m scripts.apply_nms --source <预测目录> --dest <输出目录> \
        --conf 0.5 --nms 0.6 [--manifest ... --side val --score]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from src.check_submit import clean_line
from src.evaluate_txt import evaluate
from scripts.sweep_rf48_postprocess import nms_classwise


def apply(source, dest, conf, nms_iou, drop_empty_ok=True):
    source, dest = Path(source), Path(dest)
    if not source.is_dir():
        raise FileNotFoundError(f"源预测目录不存在: {source}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    txts = sorted(source.glob("*.txt"))
    if not txts:
        raise RuntimeError(f"源目录没有 TXT: {source}")
    stats = {"files": 0, "rows_in": 0, "rows_out": 0, "empty_after": 0, "dropped_rows": 0}
    for txt in txts:
        rows = []
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            clean = clean_line(line.split())
            if clean is None:
                stats["dropped_rows"] += 1
                continue
            rows.append(clean)
        stats["rows_in"] += len(rows)
        arr = np.array(rows, dtype=float).reshape(-1, 6)
        arr = arr[arr[:, 5] >= conf]
        arr = nms_classwise(arr, nms_iou)
        arr = arr[np.argsort(-arr[:, 5], kind="stable")]
        if len(arr) > 100:
            raise ValueError(f"{txt.name}: NMS 后仍有 {len(arr)} 框，超过 100")
        if not len(arr):
            stats["empty_after"] += 1
        stats["rows_out"] += len(arr)
        stats["files"] += 1
        dest.joinpath(txt.name).write_text("".join(
            f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.6f}\n"
            for c, cx, cy, w, h, s in arr))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--manifest")
    ap.add_argument("--side", default="val")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    stats = apply(args.source, args.dest, args.conf, args.nms)
    print(f"[apply] conf>={args.conf} nms_iou={args.nms} -> {args.dest}")
    print(f"[stats] {stats}")
    if args.score:
        if not args.manifest:
            raise SystemExit("--score 需要 --manifest")
        report = evaluate(args.manifest, args.dest, args.side, coco=True)
        print(f"[score] rule_map={report['rule_map']*100:.4f}  "
              f"ultralytics_map={report['ultralytics_map']*100:.4f}  "
              f"rule50={report['rule_map50']*100:.2f}  ul50={report['ultralytics_map50']*100:.2f}  "
              f"pycoco={report['pycocotools_map']*100:.4f}  tiny={report['tiny_recall50']:.4f}  "
              f"at_max_det={report['at_max_det']}")


if __name__ == "__main__":
    main()
