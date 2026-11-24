"""Offline trimodal checkpoint -> strict competition TXT predictions."""
import argparse
from pathlib import Path

import torch

from src.trimodal_data import sha256, write_json
from src.trimodal_rfdetr import build_detector
from src.train_rfdetr import loader, predict
from src.evaluate_txt import evaluate


def run(weights, manifest, out_dir, side="test", size=768, threshold=.001, zero=None):
    torch.set_num_threads(4)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    config = checkpoint["identity"]["config"].copy()
    if side == "test" and not config["multimodal"]:
        raise ValueError("RGB diagnostic is not a competition submission model")
    config["manifest"] = str(manifest)
    model, _, _ = build_detector(config, initialize=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    fingerprint = {"weights_sha256": sha256(weights), "manifest_sha256": sha256(manifest),
                   "side": side, "imgsz": size, "conf": threshold, "zero_modality": zero,
                   "classes": checkpoint["identity"]["classes"], "preprocess": checkpoint["identity"]["preprocess"]}
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        import json
        if not (out_dir / "identity.json").exists() or json.loads((out_dir / "identity.json").read_text()) != fingerprint:
            raise ValueError("Refusing predictions with mismatched identity")
    write_json(out_dir / "identity.json", fingerprint)
    seconds = predict(model, loader(config, side, size), out_dir, size, threshold, zero)
    report = {"inference_seconds": seconds, "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30, **fingerprint}
    if side == "val":
        report.update(evaluate(manifest, out_dir, coco=True))
    write_json(out_dir / "metrics.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--sample-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--side", choices=["val", "test"], default="test")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--conf", type=float, default=.001)
    p.add_argument("--zero-modality", choices=["ir", "depth"])
    args = p.parse_args()
    print(run(args.weights, args.sample_manifest, args.out_dir, args.side, args.imgsz, args.conf, args.zero_modality))
