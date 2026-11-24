"""Rectangular canvas and sensor preprocessing for RF1029-M."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.prepare_data import load_ir_u8, read_image
from src.trimodal_data import decode, load_manifest, read_labels, sha256

# Mean and standard deviation of IR, depth and depth validity.
SENSOR_STATS = ((.4319, .2209), (.2762, .2336), (.6654, .3969))


def canvas_size(size):
    return tuple(map(int, size.split("x")))


def transform(image, size, labels, train, aug, dropout_ir, dropout_depth):
    """Apply sensor normalization and track image content on the fixed canvas."""
    strong = bool(train and aug == "strong")
    image = image.copy()
    labels = np.empty((0, 5), np.float32) if labels is None else labels.copy()
    h, w = image.shape[:2]
    if train and np.random.random() < .5:
        image = image[:, ::-1].copy()
        labels[:, 1] = 1 - labels[:, 1]
    if train:
        image[:, :, :3] = np.clip((image[:, :, :3] - .5) * np.random.uniform(.9, 1.1)
                                        + .5 + np.random.uniform(-.05, .05), 0, 1)
    if strong:
        rgb = image[:, :, :3]
        rgb = np.clip((rgb - .5) * np.random.uniform(.8, 1.25) + .5
                      + np.random.uniform(-.08, .08), 0, 1)
        rgb = np.clip(rgb * np.random.uniform(.9, 1.1, size=3).astype(np.float32), 0, 1)
        if np.random.random() < .3:
            rgb = np.clip(rgb, 0, 1) ** np.random.uniform(.7, 1.4)
        if np.random.random() < .05:
            rgb = np.repeat(rgb.mean(2, keepdims=True), 3, 2)
        image[:, :, :3] = rgb
        if np.random.random() < dropout_ir:
            image[:, :, 3] = 0
        if np.random.random() < dropout_depth:
            image[:, :, 4] = 0
            image[:, :, 5] = 0
    size_w, size_h = size
    ratio = min(size_w / w, size_h / h)
    if train:
        ratio *= np.random.uniform(.6, 1.5) if strong else np.random.uniform(.8, 1.)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    rgbir = cv2.resize(image[:, :, :4], (nw, nh), interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(image[:, :, 5], (nw, nh), interpolation=cv2.INTER_NEAREST)
    mass = cv2.resize(image[:, :, 5], (nw, nh), interpolation=cv2.INTER_LINEAR)
    numerator = cv2.resize(image[:, :, 4] * image[:, :, 5], (nw, nh), interpolation=cv2.INTER_LINEAR)
    depth = np.divide(numerator, mass, out=np.zeros_like(numerator), where=mass > 1e-6) * valid
    resized = np.dstack((rgbir, depth, valid))

    if nh > size_h:
        y0 = np.random.randint(0, nh - size_h + 1) if strong else (nh - size_h) // 2
    else:
        y0 = 0
    if nw > size_w:
        x0 = np.random.randint(0, nw - size_w + 1) if strong else (nw - size_w) // 2
    else:
        x0 = 0
    tile = resized[y0:y0 + min(nh, size_h), x0:x0 + min(nw, size_w)]
    th, tw = tile.shape[:2]
    if strong:
        y, x = np.random.randint(0, size_h - th + 1), np.random.randint(0, size_w - tw + 1)
    else:
        y, x = (size_h - th) // 2, (size_w - tw) // 2
    canvas = np.zeros((size_h, size_w, 6), np.float32)
    canvas[:, :, :3] = 114 / 255
    for ch, (mean, _) in enumerate(SENSOR_STATS, start=3):
        canvas[:, :, ch] = mean
    canvas[y:y + th, x:x + tw] = tile
    # RF-DETR uses the full canvas. Sensor attention uses its own content mask.
    padding = np.zeros((size_h, size_w), dtype=bool)
    labels[:, 1] = (labels[:, 1] * nw - x0 + x) / size_w
    labels[:, 2] = (labels[:, 2] * nh - y0 + y) / size_h
    labels[:, 3] *= nw / size_w
    labels[:, 4] *= nh / size_h
    if strong and len(labels):
        inside = ((labels[:, 1] > 0) & (labels[:, 1] < 1)
                  & (labels[:, 2] > 0) & (labels[:, 2] < 1))
        labels = labels[inside]
        if len(labels):
            hw, hh = labels[:, 3] / 2, labels[:, 4] / 2
            x1 = np.clip(labels[:, 1] - hw, 0, 1)
            x2 = np.clip(labels[:, 1] + hw, 0, 1)
            y1 = np.clip(labels[:, 2] - hh, 0, 1)
            y2 = np.clip(labels[:, 2] + hh, 0, 1)
            labels[:, 1], labels[:, 2] = (x1 + x2) / 2, (y1 + y2) / 2
            labels[:, 3], labels[:, 4] = x2 - x1, y2 - y1
            labels = labels[(labels[:, 3] > 1e-3) & (labels[:, 4] > 1e-3)]
    canvas[:, :, :3] = (canvas[:, :, :3] - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    for ch, (mean, std) in enumerate(SENSOR_STATS, start=3):
        canvas[:, :, ch] = (canvas[:, :, ch] - mean) / std
    content = np.zeros((size_h, size_w), dtype=bool)
    content[y:y + th, x:x + tw] = True
    return (torch.from_numpy(canvas.transpose(2, 0, 1).copy()), torch.from_numpy(padding),
            labels, (h, w, nh, nw, y, x), torch.from_numpy(content))


def raw_cache(row):
    """Exact integer cache: RGB/IR u8, raw depth u8/u16; no quantization."""
    rgb = read_image(Path(row["visible"]))[:, :, ::-1].copy()
    ir = load_ir_u8(Path(row["infrared"]))
    depth = read_image(Path(row["depth"]))
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    if rgb.shape[:2] != ir.shape or ir.shape != depth.shape:
        raise ValueError("Unregistered cache")
    return np.dstack((rgb, ir)), depth


def decode_cache(cache):
    rgbir, raw = cache
    valid = (raw > 0).astype(np.float32)
    depth = (np.clip(raw.astype(np.float32), 0, 20000) / 20000
             if raw.dtype == np.uint16 else raw.astype(np.float32) / 255)
    return np.dstack((rgbir.astype(np.float32) / 255, depth, valid))


class RF1029Dataset(Dataset):
    def __init__(self, config, manifest, side):
        _, self.samples = load_manifest(manifest, side)
        self.config = config
        self.size = canvas_size(config["imgsz"])
        self.train = side == "train"
        self.epoch = 0
        self.cache = None
        for row in self.samples:
            if "label" in row and sha256(row["label"]) != row["label_sha256"]:
                raise ValueError(f"Label checksum differs: {row['stem']}")
        if self.train and config["cache_images"]:
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.cache = list(pool.map(raw_cache, self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        labels = read_labels(row["label"]) if "label" in row else None
        image = decode(row) if self.cache is None else decode_cache(self.cache[index])
        seed_text = f"{self.config['seed']}:{self.epoch}:{row['stem']}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "little")
        prior = np.random.get_state()
        np.random.seed(seed)
        try:
            tensor, mask, labels, geometry, content = transform(
                image, self.size, labels, self.train, self.config["aug"],
                self.config["sensor_dropout_ir"], self.config["sensor_dropout_depth"])
        finally:
            np.random.set_state(prior)
        target = {"labels": torch.from_numpy(labels[:, 0].astype(np.int64)),
                  "boxes": torch.from_numpy(labels[:, 1:].copy())}
        return tensor, mask, target, {"stem": row["stem"], "geometry": geometry,
                                     "content_valid": content}
