"""评估一个运行的多个检查点，用于「无验证集」实验的事后选点。

背景：实验 A 用全部 2000 张训练，其中包含 A 验证集的 195 张，
因此在 A 验证集上评估会因**记忆化**而虚高，且越晚的轮次虚高越多。
因此本脚本输出的是**虚高曲线**，用途是看形状、与干净曲线对比，而不是当成绩。

    python -m scripts.eval_checkpoints --run-dir runs/formal/rf48c/run-all2000 \
        --manifest runs/formal/rf48/train-manifest.json --conf 0.35 --nms 0.6 \
        --reference runs/formal/rf48b/run-early1280/history.jsonl \
        --out runs/formal/rf48c/run-all2000/checkpoint-curve.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sweep_rf48_postprocess import load_predictions, nms_classwise  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402


def score_at(pred_dir, manifest, conf, nms):
    loaded = load_predictions(manifest, pred_dir, "val")
    records = []
    for stem, (gt, pred, h, w) in loaded.items():
        rows = pred[pred[:, 5] >= conf]
        records.append((stem, gt, nms_classwise(rows, nms), h, w))
    report = score_records(records, require_all_classes=False)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--side", default="val")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--reference", default=None, help="干净曲线的 history.jsonl（可选）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    identity = json.loads((run_dir / "identity.json").read_text())
    size = args.imgsz or int(identity["config"]["imgsz"])
    checkpoints = sorted(run_dir.glob("epoch-*.pt")) + sorted(run_dir.glob("best.pt"))
    assert checkpoints, f"{run_dir} 下没有检查点"

    from src.trimodal_data import sha256
    import subprocess
    rows = []
    for ckpt in checkpoints:
        pred_dir = run_dir / f"val-{ckpt.stem}"
        tag = ckpt.stem
        if not (pred_dir / "identity.json").exists() or args.force:
            print(f"[eval] 推理 {ckpt.name} ...", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "src.predict_rfdetr",
                            "--weights", str(ckpt), "--sample-manifest", args.manifest,
                            "--out-dir", str(pred_dir), "--side", args.side,
                            "--imgsz", str(size), "--conf", "0.001"],
                           cwd=ROOT, check=True)
        else:
            print(f"[eval] 复用已有预测 {pred_dir.name}", flush=True)
        report = score_at(pred_dir, args.manifest, args.conf, args.nms)
        epoch = None
        if ckpt.stem.startswith("epoch-"):
            epoch = int(ckpt.stem.split("-")[1])
        rows.append({"checkpoint": ckpt.name, "epoch": epoch,
                     "weights_sha256": sha256(ckpt)[:16],
                     "ultralytics_map": report["ultralytics_map"],
                     "rule_map": report["rule_map"]})

    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"] or 0))
    print(f"\n=== {args.run_dir} 检查点曲线（conf={args.conf} nms={args.nms}，**含记忆化虚高**）===")
    print(f"{'检查点':>16} {'ul 口径':>10} {'rule 口径':>10}   与前一档差")
    prev = None
    for r in rows:
        delta = "" if prev is None else f"{r['ultralytics_map'] - prev:+.4f}"
        print(f"{r['checkpoint']:>16} {r['ultralytics_map']*100:>10.4f} "
              f"{r['rule_map']*100:>10.4f}   {delta}")
        prev = r["ultralytics_map"]

    reference = None
    if args.reference:
        ref_rows = [json.loads(l) for l in (ROOT / args.reference).read_text().splitlines() if l.strip()]
        reference = [{"epoch": r["epoch"], "selection_score": r.get("selection_score"),
                      "ultralytics_map": r.get("ultralytics_map")} for r in ref_rows]
        print(f"\n=== 干净参考曲线 {args.reference}（该运行**未见过**这 195 张）===")
        best = max(reference, key=lambda r: r["selection_score"] or 0)
        print(f"  峰值在第 {best['epoch']} 轮，选中分 {best['selection_score']*100:.4f}")
        for r in reference:
            if r["epoch"] % 10 == 0:
                print(f"    第 {r['epoch']:>3} 轮: 选中分 {(r['selection_score'] or 0)*100:.4f}")

    payload = {"run_dir": args.run_dir, "imgsz": size, "conf": args.conf, "nms": args.nms,
               "note": "本曲线的分数含记忆化虚高（验证集图像已参与训练），仅用于看形状与选点",
               "checkpoints": rows, "reference": reference}
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
