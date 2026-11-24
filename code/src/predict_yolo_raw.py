"""Fingerprint-bound legacy5 YOLO inference from raw modality files."""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO
from src.trimodal_data import load_manifest, decode, sha256, write_json
from scripts.prepare_rf_phase import write_yolo_result
from src.evaluate_txt import evaluate


def run(weights, manifest, out, side, size=1280, conf=.001, iou=.4):
    _, samples = load_manifest(manifest, side)
    output = Path(out)
    identity = {"weights_sha256": sha256(weights), "manifest_sha256": sha256(manifest),
                "imgsz": size, "conf": conf, "iou": iou, "side": side, "preprocess": "legacy5"}
    if output.exists() and any(output.iterdir()):
        if not (output / "identity.json").exists() or json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Prediction output identity differs")
    write_json(output / "identity.json", identity)
    model = YOLO(weights)
    started = time.monotonic()
    for index, row in enumerate(samples, 1):
        result = model(decode(row, "legacy5"), imgsz=size, conf=conf, iou=iou,
                       max_det=100, device=0, half=True, augment=False, verbose=False)[0]
        write_yolo_result(output / (row["stem"] + ".txt"), result)
        if index % 100 == 0:
            print(index, "/", len(samples), flush=True)
    report = identity | {"inference_seconds": time.monotonic() - started}
    if side == "val":
        report.update(evaluate(manifest, output, coco=True))
    write_json(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--side", choices=["val", "test"], required=True)
    args = p.parse_args()
    run(args.weights, args.manifest, args.out, args.side)
