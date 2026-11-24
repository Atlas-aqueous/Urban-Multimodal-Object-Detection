"""在固定 max_det=100、类别独立 NMS、无 TTA 下搜索验证推理参数。"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

try:
    from src.train import metric_payload
except ModuleNotFoundError:  # 兼容 `python src/search_inference.py`
    from train import metric_payload


def parse_values(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def select_best(rows: list[dict], tolerance: float = 0.002) -> dict:
    highest = max(row["map50_95"] for row in rows)
    tied = [row for row in rows if highest - row["map50_95"] <= tolerance]
    # 先用较低分辨率控制成本；同分辨率仍以精度为先，耗时仅作最终平局项。
    return min(tied, key=lambda row: (row["imgsz"], -row["map50_95"],
                                      row["elapsed_seconds"]))


def main(args: argparse.Namespace) -> Path:
    if args.reuse:
        rows = json.loads(Path(args.reuse).read_text(encoding="utf-8"))["results"]
    else:
        if not args.weights or not args.data:
            raise ValueError("新搜索必须提供 --weights 和 --data")
        model = YOLO(args.weights)
        rows = []
        for imgsz in parse_values(args.imgsz, int):
            for conf in parse_values(args.conf, float):
                for iou in parse_values(args.iou, float):
                    started = time.time()
                    result = model.val(
                        data=str(Path(args.data).resolve()), imgsz=imgsz, conf=conf,
                        iou=iou, max_det=100, device=args.device, half=args.half,
                        augment=False, agnostic_nms=False, plots=False, verbose=False,
                        project=str(Path(args.project).resolve()),
                        name=f"i{imgsz}-c{conf:g}-n{iou:g}", exist_ok=True,
                    )
                    metrics = metric_payload(result)
                    row = {
                        "imgsz": imgsz, "conf": conf, "iou": iou,
                        "elapsed_seconds": time.time() - started, **metrics,
                    }
                    rows.append(row)
                    print(f"[grid] {imgsz=} {conf=} {iou=} map={row['map50_95']:.6f}")
    best = select_best(rows)
    payload = {
        "fixed": {"max_det": 100, "agnostic_nms": False, "tta": False},
        "tie_tolerance": 0.002, "best": best, "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[grid] best={best} -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--reuse", type=Path, help="复用已有 results JSON，仅重算选择规则")
    ap.add_argument("--imgsz", default="960,1152,1280")
    ap.add_argument("--conf", default="0.0005,0.001,0.002,0.005")
    ap.add_argument("--iou", default="0.55,0.60,0.65,0.70,0.75")
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--project", default="runs/formal/inference-grid", type=Path)
    ap.add_argument("--out", default="runs/formal/inference-grid/results.json", type=Path)
    main(ap.parse_args())
