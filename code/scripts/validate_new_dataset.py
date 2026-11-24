"""新数据集到手后的第一道检查：几分钟内确认能不能直接用现有管线。

不训练、不推理，只做数据体检。输出 JSON + 人类可读摘要 + 明确建议。

    python -m scripts.validate_new_dataset --root /path/to/data --out runs/formal/rf48c/data-audit.json

检查项：
  1. 三模态目录存在且文件名一一对应（同一 stem）
  2. 三模态在同一批次内尺寸严格配准
  3. 可见光分辨率与宽高比分布（决定画布尺寸）
  4. 深度图的位深与取值分布（决定用 clip(20000)/20000 还是 /255）
  5. 若有标签：类别直方图、框数与合法性、类别表是否与训练集一致
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
INITIAL_CLASSES = ["person", "boat", "animal", "seat", "sign", "bicycle",
                   "car", "ball", "light", "garbage can", "uav", "tricycle"]


def index_dir(path):
    index = {}
    if not path.is_dir():
        return index
    for entry in sorted(path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            index.setdefault(entry.stem, []).append(entry)
    return index


def read_gray(path):
    import cv2
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise RuntimeError(f"无法读取 {path}")
    return array


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="数据集根目录（含 visible/infrared/depth）")
    ap.add_argument("--labels", default=None, help="标签目录（可选）")
    ap.add_argument("--sample", type=int, default=60, help="抽样检查的样本数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    report = {"root": str(root), "checks": {}, "sampled": 0, "warnings": [], "advice": []}
    dirs = {name: index_dir(root / name) for name in ("visible", "infrared", "depth")}
    for name, index in dirs.items():
        report["checks"][f"{name}_dir_exists"] = (root / name).is_dir()
        report["checks"][f"{name}_files"] = len(index)

    missing = [n for n, i in dirs.items() if not i]
    if missing:
        report["warnings"].append(f"缺少目录或目录为空: {missing}")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return

    stems = sorted(set(dirs["visible"]) & set(dirs["infrared"]) & set(dirs["depth"]))
    report["checks"]["common_stems"] = len(stems)
    for name, index in dirs.items():
        extra = set(index) - set(stems)
        if extra:
            report["warnings"].append(f"{name} 有 {len(extra)} 个 stem 不匹配，例如 {sorted(extra)[:3]}")
        dup = {k: len(v) for k, v in index.items() if len(v) > 1}
        if dup:
            report["warnings"].append(f"{name} 有 {len(dup)} 个重复 stem（多扩展名），例如 {list(dup)[:3]}")

    rng = np.random.default_rng(0)
    picked = [stems[i] for i in rng.choice(len(stems), size=min(args.sample, len(stems)), replace=False)]

    sizes, ratios, ext = Counter(), Counter(), Counter()
    depth_dtypes, depth_max, depth_zero_frac, depth_min_nonzero = Counter(), [], [], []
    ir_dtypes = Counter()
    mismatch = []
    for stem in picked:
        vis_path = dirs["visible"][stem][0]
        ir_path = dirs["infrared"][stem][0]
        dp_path = dirs["depth"][stem][0]
        vis = read_gray(vis_path)
        ir = read_gray(ir_path)
        dep = read_gray(dp_path)
        if vis.shape[:2] != ir.shape[:2] or vis.shape[:2] != dep.shape[:2]:
            mismatch.append({"stem": stem, "visible": vis.shape[:2], "infrared": ir.shape[:2],
                             "depth": dep.shape[:2]})
        sizes[(vis.shape[1], vis.shape[0])] += 1
        ratios[round(vis.shape[1] / vis.shape[0], 3)] += 1
        ext[(vis_path.suffix.lower(), ir_path.suffix.lower(), dp_path.suffix.lower())] += 1
        depth_dtypes[str(dep.dtype)] += 1
        ir_dtypes[str(ir.dtype)] += 1
        depth_max.append(int(dep.max()))
        depth_zero_frac.append(float((dep == 0).mean()))
        nz = dep[dep > 0]
        depth_min_nonzero.append(int(nz.min()) if nz.size else -1)

    report["sampled"] = len(picked)
    report["resolution"] = [{"size": f"{w}x{h}", "count": c} for (w, h), c in sizes.most_common()]
    report["aspect_ratio"] = [{"ratio": r, "count": c} for r, c in ratios.most_common(5)]
    report["extensions"] = [{"visible/infrared/depth": list(k), "count": c} for k, c in ext.most_common(5)]
    report["depth"] = {"dtypes": dict(depth_dtypes), "max_values": sorted(set(depth_max))[:8],
                       "zero_fraction_mean": round(float(np.mean(depth_zero_frac)), 4),
                       "min_nonzero": sorted(set(depth_min_nonzero))[:8]}
    report["infrared_dtypes"] = dict(ir_dtypes)
    report["registration_mismatch"] = mismatch[:5]
    if mismatch:
        report["warnings"].append(f"{len(mismatch)}/{len(picked)} 个样本三模态尺寸不一致")
    else:
        report["checks"]["registration_ok"] = True

    # 深度编码建议
    if depth_dtypes.get("uint16"):
        report["advice"].append(
            "深度为 uint16：沿用 native6-linear20m-v1 的 clip(depth, 0, 20000) / 20000。"
            "若最大值远超 20000，需重新确定裁剪上界。")
    elif depth_dtypes.get("uint8"):
        report["advice"].append(
            "深度为 uint8：按灰度值 /255 处理，不要解释成毫米（与训练集 JPG 组一致）。")
    else:
        report["advice"].append(f"深度 dtype 为 {list(depth_dtypes)}，需确认单位与量纲后再定编码。")

    # 画布建议
    if report["aspect_ratio"]:
        ar = report["aspect_ratio"][0]["ratio"]
        report["advice"].append(
            f"主宽高比 {ar}。训练数据为 16:9 且已验证画布 1280（缩放 0.667）最优、1536 反而更差；"
            f"若新数据集分辨率分布不同，画布需要重新确定，且必须满足被 32 整除。")
    if len(ratios) > 1:
        report["warnings"].append(f"存在 {len(ratios)} 种宽高比，居中补边会浪费画布，需注意。")

    # 标签检查
    if args.labels:
        label_dir = Path(args.labels)
        files = sorted(label_dir.rglob("*.txt"))
        counts, boxes, bad, empty = Counter(), 0, 0, 0
        for path in files:
            rows = [l.split() for l in path.read_text().splitlines() if l.strip()]
            if not rows:
                empty += 1
            boxes += len(rows)
            for row in rows:
                try:
                    c = int(float(row[0]))
                    cx, cy, w, h = map(float, row[1:5])
                    if not (0 <= c < 12 and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        bad += 1
                    else:
                        counts[c] += 1
                except Exception:
                    bad += 1
        report["labels"] = {"files": len(files), "boxes": boxes, "empty_files": empty,
                            "invalid_rows": bad,
                            "per_class": {INITIAL_CLASSES[c] if c < 12 else str(c): n
                                          for c, n in sorted(counts.items())}}
        if bad:
            report["warnings"].append(f"标签中有 {bad} 行非法")
        report["advice"].append("类别表需与训练集核对：若类别数或顺序不同，checkpoint 与提交格式都要改。")
    else:
        report["advice"].append("未提供标签目录；若评测阶段给了新训练集，请用 --labels 再跑一次。")

    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
