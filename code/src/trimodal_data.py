"""Versioned raw RGB/IR/depth decoding shared by training and inference."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

from src.prepare_data import (NAMES, IMAGE_EXTS, read_image, load_ir_u8,
                              load_depth_u8, build_5ch, _load_split_manifest)

PREPROCESS_VERSION = "native6-linear20m-v1"

# 训练增强参数。默认值与历史行为逐项一致（bc=0.1 对应 uniform(.9,1.1) 与 ±.05），
# 因此不传 augmentation 时行为完全不变。
DEFAULT_AUG = {"flip": .5, "scale_lo": .8, "scale_hi": 1.0, "bc": .1, "color": 0.0}


def load_config(path):
    # YAML 1.1 treats JSON exponents such as 1e-05 as strings; parse JSON as JSON.
    import yaml
    path = Path(path)
    return json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_manifest(root, split_path=None, label_root=None):
    root = Path(root).resolve()
    indexes = {}
    for mod in ("visible", "infrared", "depth"):
        index = {}
        for path in sorted((root / mod).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                continue
            if path.stem in index:
                raise ValueError(f"Duplicate image: {path}")
            index[path.stem] = str(path.resolve())
        indexes[mod] = index
    stems = sorted(indexes["visible"])
    if not stems or any(set(index) != set(stems) for index in indexes.values()):
        raise ValueError("Missing/extra modality images")
    splits = _load_split_manifest(Path(split_path), stems) if split_path else {"test": stems}
    sides = {s: side for side, items in splits.items() for s in items}
    samples = []
    for stem in stems:
        row = {"stem": stem, "split": sides[stem], **{m: ix[stem] for m, ix in indexes.items()}}
        if split_path:
            # Existing derivative labels contain the five documented repairs.
            label = Path(label_root) / sides[stem] / f"{stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            row.update(label=str(label.resolve()), label_sha256=sha256(label))
        samples.append(row)
    return {"version": 1, "preprocess": PREPROCESS_VERSION,
            "classes": list(NAMES.values()), "root": str(root),
            "split_sha256": sha256(split_path) if split_path else None,
            "split_manifest": json.loads(Path(split_path).read_text()) if split_path else None,
            "samples": samples}


def load_manifest(path, side=None):
    manifest = json.loads(Path(path).read_text())
    samples = [r for r in manifest["samples"] if side is None or r["split"] == side]
    if not samples or len({r["stem"] for r in samples}) != len(samples):
        raise ValueError("Empty or duplicate sample manifest")
    return manifest, samples


def decode(row, mode="native6"):
    vis = read_image(Path(row["visible"]))
    ir = load_ir_u8(Path(row["infrared"]))
    depth_path = Path(row["depth"])
    if mode == "legacy5":
        return build_5ch(vis, ir, load_depth_u8(depth_path))
    if mode != "native6":
        raise ValueError(mode)
    raw = read_image(depth_path)
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    valid = (raw > 0).astype(np.float32)
    depth = np.clip(raw.astype(np.float32), 0, 20000) / 20000 if raw.dtype == np.uint16 else raw.astype(np.float32) / 255
    if vis.shape[:2] != ir.shape or ir.shape != raw.shape:
        raise ValueError(f"Unregistered dimensions: {row['stem']}")
    return np.dstack((vis[:, :, ::-1].astype(np.float32) / 255,
                      ir.astype(np.float32) / 255, depth, valid))


def read_labels(path):
    text = Path(path).read_text().strip()
    arr = np.array([list(map(float, l.split())) for l in text.splitlines()], dtype=np.float32).reshape(-1, 5)
    if (not np.isfinite(arr).all() or (arr[:, 0] != arr[:, 0].astype(int)).any()
            or (arr[:, 0] < 0).any() or (arr[:, 0] > 11).any() or (arr[:, 3:] <= 0).any()):
        raise ValueError(f"Invalid labels: {path}")
    return arr


def transform(image, size, labels=None, train=False, aug=None):
    """Centered isotropic letterbox; return inverse geometry in original pixels."""
    aug = {**DEFAULT_AUG, **(aug or {})}
    image = image.copy()
    labels = np.empty((0, 5), np.float32) if labels is None else labels.copy()
    h, w = image.shape[:2]
    if train and np.random.random() < aug["flip"]:
        image = image[:, ::-1].copy()
        labels[:, 1] = 1 - labels[:, 1]
    if train and aug["bc"] > 0:
        # RGB-only brightness/contrast, never applied to sensor channels.
        j = aug["bc"]
        image[:, :, :3] = np.clip((image[:, :, :3] - .5) * np.random.uniform(1 - j, 1 + j)
                                  + .5 + np.random.uniform(-j / 2, j / 2), 0, 1)
    if train and aug["color"] > 0:
        # 仅 RGB 的逐通道增益（模拟成像差异），绝不施加到红外/深度。
        gain = np.random.uniform(1 - aug["color"], 1 + aug["color"], size=3).astype(np.float32)
        image[:, :, :3] = np.clip(image[:, :, :3] * gain, 0, 1)
    ratio = size / max(h, w)
    if train:
        ratio *= np.random.uniform(aug["scale_lo"], aug["scale_hi"])
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    rgbir = cv2.resize(image[:, :, :4], (nw, nh), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(image[:, :, 5], (nw, nh), interpolation=cv2.INTER_NEAREST)
    mass = cv2.resize(image[:, :, 5], (nw, nh), interpolation=cv2.INTER_LINEAR)
    numerator = cv2.resize(image[:, :, 4] * image[:, :, 5], (nw, nh), interpolation=cv2.INTER_LINEAR)
    depth = np.divide(numerator, mass, out=np.zeros_like(numerator), where=mass > 1e-6) * valid
    resized = np.dstack((rgbir, depth, valid))
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 6), np.float32)
    canvas[:, :, :3] = 114 / 255
    canvas[y:y + nh, x:x + nw] = resized
    # The square letterbox is the detector's image coordinate system, NOT
    # variable-size NestedTensor padding. RF-DETR's valid-ratio/proposal code
    # assumes top-left valid regions; marking centered bars as tensor padding
    # makes a valid ratio zero and silently kills encoder gradients. Treat the
    # entire fixed canvas as input; sensor validity remains a separate channel.
    padding = np.zeros((size, size), dtype=bool)
    labels[:, 1] = (labels[:, 1] * nw + x) / size
    labels[:, 2] = (labels[:, 2] * nh + y) / size
    labels[:, 3] *= nw / size
    labels[:, 4] *= nh / size
    canvas[:, :, :3] = (canvas[:, :, :3] - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    return torch.from_numpy(canvas.transpose(2, 0, 1).copy()), torch.from_numpy(padding), labels, (h, w, nh, nw, y, x)


class TrimodalDataset(Dataset):
    def __init__(self, manifest, side, size, train=False, aug=None):
        self.manifest, self.samples = load_manifest(manifest, side)
        self.size, self.train, self.aug = size, train, aug
        for row in self.samples:
            if "label" in row and sha256(row["label"]) != row["label_sha256"]:
                raise ValueError(f"Label changed: {row['stem']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        labels = read_labels(row["label"]) if "label" in row else None
        tensor, mask, labels, geometry = transform(decode(row), self.size, labels, self.train, self.aug)
        target = {"labels": torch.from_numpy(labels[:, 0].astype(np.int64)),
                  "boxes": torch.from_numpy(labels[:, 1:].copy())}
        return tensor, mask, target, {"stem": row["stem"], "geometry": geometry}


def collate(batch):
    images, masks, targets, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(targets), list(metadata)
