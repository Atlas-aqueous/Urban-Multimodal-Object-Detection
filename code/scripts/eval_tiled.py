"""切块推理评测：把小目标放大后再送入同一个模型，最后跨块合并。

动机（来自分尺寸 AP）：
    side<16px     14 个  AP 14.3
    16-32px      196 个  AP 24.8   ← 12% 的目标卡在这里
    32-64px      553 个  AP 45.7
    64-128px     499 个  AP 51.1
    >=128px      430 个  AP 73.5
整图 letterbox 到 1280 后，1920x1080 的缩放比只有 0.667，24px 的目标
在画布上只剩 16px。切块后每块放大到 1280，等效放大 ~1.75 倍。

实现要点：**不改动任何训练/推理代码**。先在原图上裁剪，再把裁剪块交给
原有的 transform()，因此通道顺序、深度质量加权下采样、letterbox、
归一化全部与线上完全一致；只有几何反算需要把「块坐标」平移回「原图坐标」，
做法是给 output_rows() 传一组等价几何参数（见 effective_geometry）。

    python -m scripts.eval_tiled --weights weights/best-early-ep67.pt \
        --grid 2x2 --overlap 0.25 --include-full --conf 0.35 --nms 0.6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference_checkpoint import load_inference_checkpoint  # noqa: E402
from src.trimodal_data import decode, load_manifest, write_json  # noqa: E402
from src.trimodal_data import transform  # noqa: E402
from src.trimodal_rfdetr import build_detector, NestedTensor, output_rows  # noqa: E402
from scripts.sweep_rf48_postprocess import load_predictions, nms_classwise  # noqa: E402
from src.evaluate_txt import score_records  # noqa: E402


def windows(W, H, nx, ny, ov):
    """nx x ny 网格，相邻块重叠比例 ov（0<=ov<1）。返回 [(x0,y0,x1,y1,名字), ...]。"""
    if nx == 1 and ny == 1:
        return [(0, 0, W, H, "full")]
    tw = W / (nx - (nx - 1) * ov)
    th = H / (ny - (ny - 1) * ov)
    out = []
    for j in range(ny):
        for i in range(nx):
            x0 = int(round(i * tw * (1 - ov)))
            y0 = int(round(j * th * (1 - ov)))
            x1 = min(W, int(round(x0 + tw)))
            y1 = min(H, int(round(y0 + th)))
            out.append((x0, y0, x1, y1, f"t{i}{j}"))
    return out


def effective_geometry(geom, x0, y0, W, H):
    """把「块几何」改写成「原图几何」，使 output_rows 直接输出原图归一化框。

    output_rows 用 (canvas - x) * w / nw 反算像素再除以 w。
    目标： (canvas - x) * K + x0 ，其中 K = cw / nw。
    令 w := W、 nw := W * nw / cw、 x := x - x0 / K 即可等价。
    """
    h, w, nh, nw, y, x = geom
    kx = w / nw
    ky = h / nh
    return (H, W, H * nh / h, W * nw / w, y - y0 / ky, x - x0 / kx)


def parse_rows(lines):
    if not lines:
        return np.zeros((0, 6), dtype=float)
    return np.array([[float(v) for v in ln.split()] for ln in lines], dtype=float)


def run(weights, manifest, out_dir, grid, overlap, include_full, size, conf, nms,
        side="val", max_keep=100, tile_max_side=0, *, expected_sha256=None):
    torch.set_num_threads(4)
    ckpt, weights_sha256 = load_inference_checkpoint(weights, expected_sha256)
    config = ckpt["identity"]["config"].copy()
    config["manifest"] = str(manifest)
    model, _, _ = build_detector(config, initialize=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().cuda()
    size = size or int(config["imgsz"])
    _, samples = load_manifest(manifest, side)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny = (int(v) for v in grid.lower().split("x"))
    view_stats = {}
    started = time.monotonic()
    with torch.no_grad():
        for row in samples:
            image = decode(row)
            H, W = image.shape[:2]
            views = windows(W, H, nx, ny, overlap)
            if include_full and not (nx == 1 and ny == 1):
                # 整图这一路放最前，它的框用来兜住大目标。
                views = [(0, 0, W, H, "full")] + views
            tensors, geoms = [], []
            for (x0, y0, x1, y1, name) in views:
                crop = image[y0:y1, x0:x1]
                tensor, mask, _, geom = transform(crop, size)
                tensors.append(tensor)
                geoms.append((effective_geometry(geom, x0, y0, W, H), name))
            batch = torch.stack(tensors).cuda()
            masks = torch.zeros(batch.shape[0], size, size, dtype=torch.bool).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(NestedTensor(batch, masks))
            rows = []
            for logit, box, (geom, name) in zip(out["pred_logits"], out["pred_boxes"], geoms):
                lines = output_rows(logit, box, geom, size, .001)
                n = len(lines)
                view_stats[name] = view_stats.get(name, 0) + n
                arr = parse_rows(lines)
                if tile_max_side > 0 and name != "full" and len(arr):
                    # 只信任切块路的小目标：整图短边超过阈值的框一律丢弃，
                    # 避免切块路在放大尺度下把大目标判错后污染结果。
                    short = np.minimum(arr[:, 3] * W, arr[:, 4] * H)
                    arr = arr[short <= tile_max_side]
                rows.append(arr)
            merged = np.concatenate(rows) if rows else np.zeros((0, 6))
            merged = merged[merged[:, 5] >= conf]
            merged = nms_classwise(merged, nms)
            if len(merged) > max_keep:
                merged = merged[np.argsort(-merged[:, 5], kind="stable")[:max_keep]]
            text = "\n".join(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {s:.6f}"
                             for c, cx, cy, bw, bh, s in merged)
            (out_dir / (row["stem"] + ".txt")).write_text(text + ("\n" if text else ""))
    torch.cuda.synchronize()
    seconds = time.monotonic() - started

    loaded = load_predictions(manifest, out_dir, side)
    records = []
    per_image = []
    for stem, (gt, pred, h, w) in loaded.items():
        records.append((stem, gt, pred, h, w))
        per_image.append(len(pred))
    report = score_records(records, require_all_classes=False)
    payload = {"weights": str(weights), "weights_sha256": weights_sha256[:16],
               "manifest": str(manifest), "grid": grid, "overlap": overlap,
               "include_full": include_full, "size": size, "conf": conf, "nms": nms,
               "tile_max_side": tile_max_side,
               "inference_seconds": seconds, "boxes_per_image_mean": float(np.mean(per_image)),
               "view_box_counts": view_stats,
               "ultralytics_map": report["ultralytics_map"],
               "rule_map": report["rule_map"],
               "ultralytics_per_class": report["ultralytics_per_class"],
               "tiny_recall50": report.get("tiny_recall50")}
    out_dir_parent = out_dir.parent
    write_json(out_dir_parent / (out_dir.name + "-score.json"), payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights/best-early-ep67.pt")
    ap.add_argument("--weights-sha256", help="Expected full SHA-256 of the checkpoint")
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--grid", default="2x2", help="例如 1x1 / 2x2 / 3x2")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--include-full", action="store_true", help="是否同时跑整图这一路")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--side", default="val")
    ap.add_argument("--tile-max-side", type=int, default=0,
                    help=">0 时只保留切块路中短边不超过该像素值的框")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or (f"{args.grid}-ov{args.overlap}"
                       f"{'-full' if args.include_full else ''}"
                       f"{'-max' + str(args.tile_max_side) if args.tile_max_side else ''}")
    out_dir = args.out_dir or f"runs/formal/rf48c/tiled/{tag}"
    payload = run(args.weights, args.manifest, out_dir, args.grid, args.overlap,
                  args.include_full, args.size, args.conf, args.nms, args.side,
                  tile_max_side=args.tile_max_side, expected_sha256=args.weights_sha256)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("ultralytics_per_class",)}, ensure_ascii=False, indent=1))
    print("逐类 AP:", [None if p is None else round(p, 4) for p in payload["ultralytics_per_class"]])


if __name__ == "__main__":
    main()
