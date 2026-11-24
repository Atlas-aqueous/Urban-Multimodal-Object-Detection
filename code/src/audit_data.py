"""赛前数据审计：检查三模态配对、图像规格、标签合法性并输出统计摘要。

用法：python src/audit_data.py --data-root ./data [--strict-official] [--json report.json]
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prepare_data import IMAGE_EXTS, NAMES, read_image


def image_map(folder: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(folder.iterdir() if folder.is_dir() else []):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            if path.stem in result:
                raise RuntimeError(f"{folder} 存在同名不同扩展名: {path.stem}")
            result[path.stem] = path
    return result


def audit(data_root: Path, strict_official: bool = False) -> dict:
    root = Path(data_root)
    errors, warnings = [], []
    maps = {name: image_map(root / name) for name in ("visible", "infrared", "depth")}
    label_dir = root / "labels"
    labels = {p.stem: p for p in sorted(label_dir.glob("*.txt"))} if label_dir.is_dir() else {}
    all_stems = set().union(*(set(items) for items in maps.values()), set(labels))
    for name, items in {**maps, "labels": labels}.items():
        missing = sorted(all_stems - set(items))
        if missing:
            errors.append(f"{name} 缺少 {len(missing)} 个样本: {missing[:10]}")

    class_counts, scene_counts = Counter(), Counter()
    label_rows = 0
    for stem, path in labels.items():
        scene_counts[stem.split("_")[0]] += 1
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
            if not line.strip():
                continue
            label_rows += 1
            parts = line.split()
            try:
                values = [float(x) for x in parts]
            except ValueError:
                errors.append(f"{path.name}:{line_no} 非数字标签")
                continue
            valid = (len(values) == 5 and all(math.isfinite(x) for x in values) and
                     values[0].is_integer() and 0 <= values[0] < len(NAMES) and
                     0 <= values[1] <= 1 and 0 <= values[2] <= 1 and
                     0 < values[3] <= 1 and 0 < values[4] <= 1)
            if not valid:
                errors.append(f"{path.name}:{line_no} 非法标签: {line}")
            else:
                class_counts[int(values[0])] += 1

    ir_spreads, depth_zero_fractions = [], []
    for stem in sorted(set.intersection(*(set(items) for items in maps.values())) if maps else set()):
        vis, ir, depth = (read_image(maps[name][stem]) for name in ("visible", "infrared", "depth"))
        if len({vis.shape[:2], ir.shape[:2], depth.shape[:2]}) != 1:
            errors.append(f"{stem} 三模态尺寸不一致: {vis.shape}, {ir.shape}, {depth.shape}")
        if not (vis.ndim == 3 and vis.shape[2] == 3 and vis.dtype == np.uint8):
            errors.append(f"{stem} visible 应为3通道uint8，实际 {vis.shape}/{vis.dtype}")
        if not (ir.ndim == 3 and ir.shape[2] == 3 and ir.dtype == np.uint8):
            errors.append(f"{stem} infrared 应为3通道uint8，实际 {ir.shape}/{ir.dtype}")
        else:
            ir_spreads.append(int(np.ptp(ir.astype(np.int16), axis=2).max()))
        official_depth = depth.ndim == 2 and depth.dtype == np.uint16
        if not official_depth:
            message = f"{stem} depth 非官方单通道uint16规格，实际 {depth.shape}/{depth.dtype}"
            (errors if strict_official else warnings).append(message)
        depth_plane = depth[:, :, 0] if depth.ndim == 3 else depth
        depth_zero_fractions.append(float((depth_plane == 0).mean()))
        for name in ("visible", "infrared", "depth"):
            if maps[name][stem].suffix.lower() != ".png":
                message = f"{name}/{maps[name][stem].name} 非官方PNG格式"
                (errors if strict_official else warnings).append(message)

    missing_classes = [f"{i}:{NAMES[i]}" for i in NAMES if class_counts[i] == 0]
    if missing_classes:
        warnings.append(f"标签中缺少类别: {missing_classes}")
    stats = {
        "samples": len(all_stems),
        "counts": {**{k: len(v) for k, v in maps.items()}, "labels": len(labels)},
        "scenes": len(scene_counts),
        "scene_image_counts": dict(sorted(scene_counts.items())),
        "label_rows": label_rows,
        "class_counts": {str(i): class_counts[i] for i in NAMES},
        "ir_max_channel_spread": max(ir_spreads, default=None),
        "depth_zero_fraction": {
            "min": min(depth_zero_fractions, default=None),
            "mean": float(np.mean(depth_zero_fractions)) if depth_zero_fractions else None,
            "max": max(depth_zero_fractions, default=None),
        },
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}


def main(args: argparse.Namespace) -> int:
    report = audit(args.data_root, args.strict_official)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data", type=Path)
    parser.add_argument("--strict-official", action="store_true",
                        help="把样例中的JPG/uint8深度等非官方格式也视为错误")
    parser.add_argument("--json", type=Path, help="另存机器可读报告")
    raise SystemExit(main(parser.parse_args()))
