"""Legacy 5ch YOLO dataset without TIFF caches; bounded E9 fallback."""
import argparse
import math
from copy import copy
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from src.trimodal_data import load_manifest, read_labels, read_image, decode, write_json, sha256

MANIFEST = "runs/formal/rf48/train-manifest.json"


class RawDataset(YOLODataset):
    def __init__(self, *args, side="train", **kwargs):
        _, self.samples = load_manifest(MANIFEST, side)
        self.by_path = {row["visible"]: row for row in self.samples}
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        return [row["visible"] for row in self.samples]

    def get_labels(self):
        result = []
        for path in self.im_files:
            row = self.by_path[path]
            values = read_labels(row["label"])
            if sha256(row["label"]) != row["label_sha256"]:
                raise ValueError("Label fingerprint changed")
            result.append(dict(im_file=path, shape=read_image(Path(path)).shape[:2], cls=values[:, :1],
                               bboxes=values[:, 1:], segments=[], keypoints=None, normalized=True, bbox_format="xywh"))
        return result

    def load_image(self, i, rect_mode=True):
        image = decode(self.by_path[self.im_files[i]], "legacy5")
        h, w = image.shape[:2]
        if rect_mode:
            ratio = self.imgsz / max(h, w)
            if ratio != 1:
                image = cv2.resize(image, (min(math.ceil(w * ratio), self.imgsz),
                                          min(math.ceil(h * ratio), self.imgsz)), interpolation=cv2.INTER_LINEAR)
        elif (h, w) != (self.imgsz, self.imgsz):
            image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return image, (h, w), image.shape[:2]


def dataset(args, data, mode, batch, stride=32):
    return RawDataset(img_path=data[mode], side=mode, imgsz=args.imgsz, batch_size=batch,
                      augment=mode == "train", hyp=args, rect=mode == "val", cache=False,
                      single_cls=False, stride=stride, pad=0.5 if mode == "val" else 0.,
                      prefix=f"raw-{mode}: ", task="detect", data=data)


class RawValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return dataset(self.args, self.data, mode, batch, int(self.stride))


class RawTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return dataset(self.args, self.data, mode, batch)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return RawValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def fallback(out, weights, split_manifest):
    root = Path.cwd()
    manifest, samples = load_manifest(MANIFEST)
    assert sha256(split_manifest) == manifest["split_sha256"]
    data = {"path": str(root), "train": str(root / "formal_data/train_revised_labels/visible"),
            "val": str(root / "formal_data/train_revised_labels/visible"),
            "names": dict(enumerate(manifest["classes"])), "channels": 5}
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "raw-data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True))
    write_json(out / "start-provenance.json", {"manifest_sha256": sha256(MANIFEST),
                "initial_weights_sha256": sha256(weights), "split_sha256": sha256(split_manifest),
                "note": "Fallback only; existing E9 not overwritten"})
    model = YOLO(weights)
    def evaluate_saved_epoch(trainer):
        import shutil
        import torch
        from scripts.prepare_rf_phase import write_yolo_result
        from src.evaluate_txt import evaluate
        saved = YOLO(trainer.last)
        directory = out / "val-latest"
        directory.mkdir(exist_ok=True)
        for row in [r for r in samples if r["split"] == "val"]:
            prediction = saved(decode(row, "legacy5"), imgsz=1280, conf=.001, iou=.4,
                               max_det=100, device=0, half=True, augment=False, verbose=False)[0]
            write_yolo_result(directory / (row["stem"] + ".txt"), prediction)
        report = evaluate(MANIFEST, directory, coco=trainer.epoch == 0)
        report["epoch"] = trainer.epoch + 1
        import json
        with (out / "rule-history.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        if not (out / "best-rule.json").exists() or report["rule_map"] > json.loads((out / "best-rule.json").read_text())["rule_map"]:
            shutil.copy2(trainer.last, out / "best-rule.pt")
            write_json(out / "best-rule.json", report)
        del saved
        torch.cuda.empty_cache()
    model.add_callback("on_model_save", evaluate_saved_epoch)
    model.train(trainer=RawTrainer, data=str(yaml_path), project=str(out.parent), name=out.name,
                exist_ok=True, imgsz=1280, batch=4, epochs=20, patience=20, optimizer="AdamW",
                lr0=1e-5, lrf=.01, warmup_epochs=1, cos_lr=True, weight_decay=1e-4,
                mosaic=0., mixup=0., hsv_h=0., hsv_s=0., hsv_v=0., workers=4,
                seed=42, deterministic=True, device=0, amp=True, max_det=100,
                conf=.001, iou=.4, save_period=-1, plots=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split-manifest", required=True)
    args = p.parse_args()
    fallback(args.out, args.weights, args.split_manifest)
