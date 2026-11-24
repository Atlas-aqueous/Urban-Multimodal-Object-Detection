"""Guarded raw-loader parity, baseline replay and recoverable TIFF retirement."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.trimodal_data import make_manifest, write_json, sha256, decode, load_manifest
from src.predict import load_image
from src.evaluate_txt import evaluate
from src.check_submit import clean_line

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/formal/rf48"
ACTIVE = ROOT / "datasets/formal-train-active"
SPLIT = ROOT / "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
BASELINES = {
    "e0": ("yolov8m-5ch-revised-960", 960, .4, .55),
    "e9": ("e9-yolov8l-p2-1280-aggressive-b4", 1280, .4, .4),
    "e12": ("e12-yolov8x-p2-1280-aggressive-b4", 960, .35, .4),
}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    train = make_manifest(ROOT / "formal_data/train_revised_labels", SPLIT, ACTIVE / "labels")
    test = make_manifest(ROOT / "formal_data/phase1_test")
    assert len(train["samples"]) == 2000 and len(test["samples"]) == 1000
    write_json(OUT / "train-manifest.json", train)
    write_json(OUT / "test-manifest.json", test)
    protected = sorted(set((ROOT / "runs/formal").glob("*/weights/best.pt")) |
                       set((ROOT / "runs/formal/submissions").glob("*.zip")))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    write_json(OUT / "protected-assets.json", hashes)
    result = {"manifest_sha256": sha256(OUT / "train-manifest.json"), "files": [], "passed": False}
    for n, row in enumerate(train["samples"], 1):
        path = ACTIVE / "images" / row["split"] / (row["stem"] + ".tiff")
        raw, old = decode(row, "legacy5"), load_image(path)
        if not np.array_equal(raw, old):
            raise AssertionError(f"Raw/TIFF mismatch: {row['stem']}")
        result["files"].append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "sha256": sha256(path)})
        if n % 100 == 0:
            print(f"raw/TIFF exact: {n}/2000", flush=True)
    result["passed"] = True
    result["rebuild"] = "python -m src.prepare_data --data-root formal_data/train_revised_labels --out datasets/formal-train-active --mode 5ch --split-manifest " + str(SPLIT.relative_to(ROOT))
    write_json(OUT / "input-parity.json", result)


def write_yolo_result(path, result):
    rows = []
    for c, box, conf in zip(result.boxes.cls.cpu().tolist(), result.boxes.xywhn.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        line = f"{int(c)} " + " ".join(f"{x:.6f}" for x in [*box, conf])
        if clean_line(line.split()) is not None:
            rows.append(line)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def baselines():
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    _, samples = load_manifest(OUT / "train-manifest.json", "val")
    all_results = {}
    for name, (run, size, conf, iou) in BASELINES.items():
        weights = ROOT / "runs/formal" / run / "weights/best.pt"
        model = YOLO(weights)
        variants = [("selected", conf)] + ([("low", .001)] if name == "e9" else [])
        for variant, threshold in variants:
            output = OUT / "baselines" / f"{name}-{variant}"
            output.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            parity_equal = True
            for n, row in enumerate(samples, 1):
                raw = decode(row, "legacy5")
                kwargs = dict(imgsz=size, conf=threshold, iou=iou, max_det=100,
                              device=0, half=True, augment=False, agnostic_nms=False, verbose=False)
                result = model(raw, **kwargs)[0]
                if name == "e9" and variant == "selected":
                    old = model(load_image(ACTIVE / "images/val" / (row["stem"] + ".tiff")), **kwargs)[0]
                    parity_equal &= torch.equal(result.boxes.data, old.boxes.data)
                    if not parity_equal:
                        raise AssertionError(f"Prediction mismatch {row['stem']}")
                write_yolo_result(output / (row["stem"] + ".txt"), result)
                if n % 50 == 0:
                    print(name, variant, n, flush=True)
            elapsed = time.monotonic() - started
            report = evaluate(OUT / "train-manifest.json", output, coco=True)
            report.update(weights_sha256=sha256(weights), imgsz=size, conf=threshold, iou=iou,
                          elapsed_seconds=elapsed, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            all_results[f"{name}-{variant}"] = report
            write_json(output / "metrics.json", report)
            if name == "e9" and variant == "selected":
                write_json(OUT / "prediction-parity.json", {"passed": parity_equal, "images": len(samples),
                           "ap_difference": 0., "manifest_sha256": sha256(OUT / "train-manifest.json")})
        del model
        torch.cuda.empty_cache()
    write_json(OUT / "baseline-summary.json", all_results)


def retire():
    source = json.loads((OUT / "input-parity.json").read_text())
    prediction = json.loads((OUT / "prediction-parity.json").read_text())
    assert source["passed"] and prediction["passed"] and prediction["ap_difference"] <= .0001
    assert source["manifest_sha256"] == prediction["manifest_sha256"] == sha256(OUT / "train-manifest.json")
    protected = json.loads((OUT / "protected-assets.json").read_text())
    for path, digest in protected.items():
        assert sha256(ROOT / path) == digest, path
    # Validate every exact target BEFORE the first deletion; never recursive.
    paths = []
    for item in source["files"]:
        path = ROOT / item["path"]
        assert not path.is_symlink() and path.resolve().is_relative_to((ACTIVE / "images").resolve())
        assert path.suffix == ".tiff" and sha256(path) == item["sha256"]
        paths.append(path)
    assert len(paths) == 2000
    record = {"status": "validated_pending_delete", "files": source["files"], "rebuild": source["rebuild"],
              "bytes": sum(i["bytes"] for i in source["files"]), "protected": protected}
    write_json(OUT / "tiff-retirement.json", record)
    for path in paths:
        path.unlink()
    for path, digest in protected.items():
        assert sha256(ROOT / path) == digest, path
    record["status"] = "complete_rebuildable_cache_only"
    write_json(OUT / "tiff-retirement.json", record)
    print("Retired only validated TIFF cache:", record["bytes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "baselines", "retire"])
    args = parser.parse_args()
    {"prepare": prepare, "baselines": baselines, "retire": retire}[args.action]()
