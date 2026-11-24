"""汇总 60+ 实验的官方口径指标、类别 AP、小目标诊断和资源消耗。"""
import argparse
import csv
import json
from pathlib import Path


def best_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    key = "metrics/mAP50-95(B)"
    return int(float(max(rows, key=lambda row: float(row[key]))["epoch"]))


def read_candidate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("official_validation") or payload.get("best") or payload
    diagnostics = metrics.get("size_diagnostics", {})
    return {
        "id": path.parent.name,
        "record": str(path),
        "map50_95": metrics.get("map50_95"),
        "map50": metrics.get("map50"),
        "map75": metrics.get("map75"),
        "per_class_ap50_95": metrics.get("per_class_ap50_95", {}),
        "tiny_recall_iou50": diagnostics.get("recall", {}).get("tiny"),
        "at_100_predictions": diagnostics.get("at_100_predictions"),
        "speed_ms_per_image": metrics.get("speed_ms_per_image"),
        "peak_gpu_memory_gib": payload.get("peak_gpu_memory_gib"),
        "peak_gpu_reserved_gib": payload.get("peak_gpu_reserved_gib"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "best_epoch": best_epoch(path.parent / "results.csv"),
        "weights_sha256": payload.get("weights_sha256"),
    }


def fmt(value, digits=4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(rows: list[dict]) -> str:
    lines = [
        "# 60+ 实验汇总", "",
        "| 实验 | mAP50-95 | mAP50 | mAP75 | ball AP | tricycle AP | tiny recall@0.5 | 100框图 | 最佳epoch | 显存GiB | 耗时h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pc = row["per_class_ap50_95"]
        elapsed = row["elapsed_seconds"] / 3600 if row["elapsed_seconds"] else None
        lines.append(
            f"| {row['id']} | {fmt(row['map50_95'])} | {fmt(row['map50'])} | "
            f"{fmt(row['map75'])} | {fmt(pc.get('ball'))} | {fmt(pc.get('tricycle'))} | "
            f"{fmt(row['tiny_recall_iou50'])} | {row['at_100_predictions'] if row['at_100_predictions'] is not None else '—'} | "
            f"{row['best_epoch'] if row['best_epoch'] is not None else '—'} | "
            f"{fmt(row['peak_gpu_memory_gib'], 2)} | {fmt(elapsed, 2)} |"
        )
    classes = sorted({name for row in rows for name in row["per_class_ap50_95"]})
    if classes:
        lines.extend(["", "## 逐类 AP50-95", "",
                      "| 实验 | " + " | ".join(classes) + " |",
                      "|---|" + "---:|" * len(classes)])
        for row in rows:
            pc = row["per_class_ap50_95"]
            lines.append("| " + row["id"] + " | " + " | ".join(fmt(pc.get(c)) for c in classes) + " |")
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    records = sorted(Path(args.root).glob("*/experiment.json"))
    rows = [read_candidate(path) for path in records]
    if args.baseline:
        baseline = read_candidate(Path(args.baseline))
        baseline["id"] = "E0-optimized-inference"
        rows.insert(0, baseline)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_markdown.write_text(markdown(rows), encoding="utf-8")
    print(f"[report] {len(rows)} experiments -> {args.out_markdown}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/formal", type=Path)
    ap.add_argument("--baseline", type=Path, help="如 inference-grid/results.json")
    ap.add_argument("--out-json", default="runs/formal/experiment-summary.json", type=Path)
    ap.add_argument("--out-markdown", default="runs/formal/experiment-summary.md", type=Path)
    main(ap.parse_args())
