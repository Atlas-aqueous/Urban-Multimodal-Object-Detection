"""三模态预处理：合成 3ch PNG / 5或6通道多页TIFF + 场景隔离划分。

通道约定以“进入网络后的顺序”为准：
- 3ch: [R, IR, Depth]。Ultralytics 会对普通 3ch 图做 BGR→RGB 翻转，
  因而磁盘 PNG 必须写成 [Depth, IR, R]；
- 5ch: [R, G, B, IR, Depth]；6ch 额外加入 DepthValid。
  Ultralytics 对非 3ch 图不会翻转通道，
  因而多页 TIFF 直接按该顺序写入。这也与 COCO 首层权重的 RGB 语义一致。

用法:
  训练集: python src/prepare_data.py --data-root ./data --out datasets/train-b --mode 3ch
  测试集: python src/prepare_data.py --data-root ./test_data --out datasets/test-a --mode 5ch --no-labels
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage can", 10: "uav", 11: "tricycle"}
DEPTH_MAX_MM = 20000  # 深度相机可感知上限 20m
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    """兼容中文路径的无损读取。"""
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"图像读取失败: {path}")
    return img


def find_triplet(data_root: Path, stem: str) -> tuple[Path, Path, Path]:
    """按 stem 在三个模态目录中定位文件（png/jpg 任意扩展名）。"""
    trio = []
    for mod in ("visible", "infrared", "depth"):
        hits = sorted(p for p in (data_root / mod).glob(f"{stem}.*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not hits:
            raise FileNotFoundError(f"{mod}/{stem}.* 未找到（data_root={data_root}）")
        if len(hits) > 1:
            raise RuntimeError(f"{mod}/{stem} 存在多个候选文件: {[p.name for p in hits]}")
        trio.append(hits[0])
    return tuple(trio)


def load_ir_u8(path: Path) -> np.ndarray:
    """红外：三通道高度一致但非严格相同 → 取均值融合为单通道。"""
    img = read_image(path)
    if img.ndim == 3:
        return np.mean(img, axis=2).round().clip(0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def load_depth_u8(path: Path) -> np.ndarray:
    """深度归一化到 uint8：
    - 单通道 uint16（毫米）: clip(0, 20000)/20000*255，0=无效区保持 0（训练/推理同规则，网络自行学习）
    - 3通道（jpg 可视化样例）: 取 ch0，已是 uint8
    """
    img = read_image(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint16:
        return (np.clip(img, 0, DEPTH_MAX_MM).astype(np.float32)
                / DEPTH_MAX_MM * 255).round().astype(np.uint8)
    return img.astype(np.uint8)


def load_depth_u8_with_validity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 uint8 深度和 0/255 有效性掩码，掩码由归一化前的原始值计算。"""
    img = read_image(path)
    if img.ndim == 3:
        # 正式 JPG 深度是三通道灰度图，通道间仅有 JPEG 量化造成的微小差异。
        raw = img[:, :, 0]
    else:
        raw = img
    valid = (raw > 0).astype(np.uint8) * 255
    if raw.dtype == np.uint16:
        depth = (np.clip(raw, 0, DEPTH_MAX_MM).astype(np.float32)
                 / DEPTH_MAX_MM * 255).round().astype(np.uint8)
    else:
        depth = raw.astype(np.uint8)
    return depth, valid


def build_3ch(vis_bgr: np.ndarray, ir: np.ndarray, dep: np.ndarray) -> np.ndarray:
    """方案B磁盘顺序 [Depth, IR, R]；Ultralytics 翻转后进入网络为 [R, IR, Depth]。"""
    return np.dstack([dep, ir, vis_bgr[:, :, 2]])


def build_5ch(vis_bgr: np.ndarray, ir: np.ndarray, dep: np.ndarray) -> np.ndarray:
    """方案A: [R, G, B, IR, Depth]，匹配 COCO 首层权重的 RGB 通道语义。"""
    return np.dstack([vis_bgr[:, :, ::-1], ir, dep])


def build_6ch(vis_bgr: np.ndarray, ir: np.ndarray, dep: np.ndarray,
              dep_valid: np.ndarray) -> np.ndarray:
    """[R,G,B,IR,Depth,DepthValid]，有效性掩码取值为 0/255。"""
    return np.dstack([vis_bgr[:, :, ::-1], ir, dep, dep_valid])


def scene_key(stem: str) -> str:
    """返回序列级分组键，避免同一短视频的相邻帧跨 train/val。

    正式集包含三种命名：纯数字单帧、``000005_011_<frame>``，以及
    ``hehe_<index>_000002_058_<frame>``。前两段通常标识采集序列；hehe
    文件的真实序列标识位于第 3、4 段。原实现只取首段，会把全部
    ``shuming_*``/``hehe_*`` 错当成一个超大场景，并把例如
    ``000005_010`` 与 ``000005_036`` 等不同序列不必要地捆绑。
    """
    parts = stem.split("_")
    if parts[0] == "hehe" and len(parts) >= 5:
        return "_".join((parts[0], parts[2], parts[3]))
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def split_by_scene(stems: list, val_ratio: float, seed: int):
    groups = sorted({scene_key(s) for s in stems})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_ratio)) if len(groups) > 1 else (1 if val_ratio > 0 else 0)
    val_groups = set(groups[:n_val])
    train = [s for s in stems if scene_key(s) not in val_groups]
    val = [s for s in stems if scene_key(s) in val_groups]
    return train, val


def _load_split_manifest(path: Path, stems: list[str]) -> dict[str, list[str]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest.get("splits", manifest)
    if set(splits) != {"train", "val"}:
        raise ValueError("划分清单必须且只能包含 train/val")
    train, val = list(splits["train"]), list(splits["val"])
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("划分清单内存在重复样本")
    if set(train) & set(val):
        raise ValueError("划分清单 train/val 存在交集")
    if set(train) | set(val) != set(stems):
        missing = sorted(set(stems) - (set(train) | set(val)))[:5]
        extra = sorted((set(train) | set(val)) - set(stems))[:5]
        raise ValueError(f"划分清单与数据不一致: missing={missing}, extra={extra}")
    if {scene_key(s) for s in train} & {scene_key(s) for s in val}:
        raise ValueError("划分清单存在场景泄漏")
    return {"train": sorted(train), "val": sorted(val)}


def label_classes(path: Path) -> set[int]:
    classes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.add(int(float(line.split()[0])))
    return classes


def write_weighted_train_list(out: Path, train_stems: list[str], ext: str,
                              labels_root: Path) -> dict[str, int]:
    """仅训练集重复 ball×2、tricycle×3；同图取最大倍数。"""
    rows, histogram = [], {"x1": 0, "x2": 0, "x3": 0}
    for stem in train_stems:
        classes = label_classes(labels_root / f"{stem}.txt")
        repeat = 3 if 11 in classes else (2 if 7 in classes else 1)
        histogram[f"x{repeat}"] += 1
        rows.extend([f"./images/train/{stem}{ext}"] * repeat)
    (out / "train_weighted.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {**histogram, "weighted_samples": len(rows)}


def refresh_rare_train_list(out: Path, data_root: Path) -> Path:
    """在不重建图像的前提下启用稀有类重复采样。

    只改写派生数据集中的训练清单与元数据；验证列表保持原样且不重复。
    """
    out, data_root = Path(out), Path(data_root)
    data_yaml, split_path = out / "data.yaml", out / "split.json"
    if not data_yaml.exists() or not split_path.exists():
        raise FileNotFoundError("仅刷新采样清单需要现有 data.yaml 和 split.json")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    if "train" not in splits or "val" not in splits:
        raise ValueError("仅刷新采样清单要求 train/val 划分")
    train, val = list(splits["train"]), list(splits["val"])
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("原始 train/val 划分存在重复")
    if set(train) & set(val):
        raise ValueError("原始 train/val 划分存在交集")
    channels = int(cfg.get("channels", 3))
    ext = ".tiff" if channels > 3 else ".png"
    missing_images = [
        stem for stem in train
        if not (out / "images" / "train" / f"{stem}{ext}").exists()
    ]
    missing_labels = [
        stem for stem in train
        if not (data_root / "labels" / f"{stem}.txt").exists()
    ]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"现有派生数据不完整: missing_images={missing_images[:5]}, "
            f"missing_labels={missing_labels[:5]}"
        )
    stats = write_weighted_train_list(out, train, ext, data_root / "labels")
    cfg["train"] = "train_weighted.txt"
    data_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    manifest["rare_repeat"] = True
    manifest["repeat_stats"] = stats
    split_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[prepare] 仅刷新稀有类采样清单 → {out} ({stats})")
    return data_yaml


def sanitize_label_file(src: Path, dst: Path) -> int:
    """复制 YOLO 标签，并将轻微越界框按图像边界裁回 [0, 1]。

    正式修订标签中存在少量中心或高宽轻微超过 1 的框。Ultralytics 会把
    包含这类行的整张图视为损坏；这里先转成 xyxy、裁边再转回 xywh，既
    保留目标，也不修改官方原始标签。结构性错误仍立即报错。
    """
    output, repaired = [], 0
    for line_no, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            values = [float(value) for value in parts]
        except ValueError as exc:
            raise ValueError(f"{src.name}:{line_no} 含非数字标签: {line}") from exc
        if (len(values) != 5 or not all(math.isfinite(value) for value in values)
                or not values[0].is_integer() or not 0 <= values[0] < len(NAMES)
                or values[3] <= 0 or values[4] <= 0):
            raise ValueError(f"{src.name}:{line_no} 标签结构非法: {line}")
        cls, cx, cy, width, height = values
        # Ultralytics 的标签合法性约束针对归一化的 cx/cy/w/h 本身；官方数据中
        # 一些边缘目标的推导 xyxy 会略出图，但四个原始字段均合法，不应擅自改框。
        if 0 <= cx <= 1 and 0 <= cy <= 1 and width <= 1 and height <= 1:
            output.append(
                f"{int(cls)} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
            )
            continue
        box = (cx - width / 2, cy - height / 2,
               cx + width / 2, cy + height / 2)
        clipped = (max(0.0, box[0]), max(0.0, box[1]),
                   min(1.0, box[2]), min(1.0, box[3]))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"{src.name}:{line_no} 裁边后面积为零: {line}")
        if clipped != box:
            repaired += 1
        x1, y1, x2, y2 = clipped
        output.append(
            f"{int(cls)} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} "
            f"{x2 - x1:.8f} {y2 - y1:.8f}"
        )
    dst.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return repaired


def _stems(data_root: Path, no_labels: bool) -> list:
    if no_labels or not (data_root / "labels").exists():
        return sorted({p.stem for p in (data_root / "visible").iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS})
    return sorted(p.stem for p in (data_root / "labels").glob("*.txt"))


def main(args: argparse.Namespace) -> Path:
    data_root, out = Path(args.data_root), Path(args.out)
    if data_root.resolve() == out.resolve():
        raise ValueError("--out 不能与 --data-root 相同")
    if getattr(args, "refresh_rare_list", False):
        return refresh_rare_train_list(out, data_root)
    managed = [out / "images", out / "labels", out / "data.yaml", out / "split.json",
               out / "train_weighted.txt"]
    if out.exists() and any(out.iterdir()):
        if not getattr(args, "overwrite", False):
            raise FileExistsError(f"输出目录非空: {out}；确认是派生数据后使用 --overwrite")
        for path in managed:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    stems = _stems(data_root, args.no_labels)
    if not stems:
        raise RuntimeError(f"data_root 下未发现数据: {data_root}")
    channels = {"3ch": 3, "5ch": 5, "6ch": 6}[args.mode]
    is_multichannel = channels > 3
    split_manifest = getattr(args, "split_manifest", None)

    if args.no_labels:
        splits = {"test": stems}
    elif getattr(args, "all_train", False):
        splits = {"train": stems}
    else:
        if split_manifest:
            splits = _load_split_manifest(split_manifest, stems)
            train, val = splits["train"], splits["val"]
        else:
            train, val = split_by_scene(stems, args.val_ratio, args.seed)
            splits = {"train": train, "val": val}
        if not train or not val:
            raise RuntimeError(f"划分失败: train={len(train)} val={len(val)}（数据太少？）")

    ext = ".tiff" if is_multichannel else ".png"
    repaired_labels = 0
    for split, group in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        if not args.no_labels:
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in group:
            vis_p, ir_p, dep_p = find_triplet(data_root, stem)
            vis = read_image(vis_p)
            ir = load_ir_u8(ir_p)
            dep, dep_valid = load_depth_u8_with_validity(dep_p)
            if vis.ndim != 3 or vis.shape[2] != 3 or vis.dtype != np.uint8:
                raise ValueError(f"{stem} 可见光必须为3通道 uint8，实际 {vis.shape}/{vis.dtype}")
            if not vis.shape[:2] == ir.shape[:2] == dep.shape[:2]:
                raise ValueError(f"{stem} 三模态尺寸不一致: {vis.shape} {ir.shape} {dep.shape}")
            if channels == 6:
                fused = build_6ch(vis, ir, dep, dep_valid)
            elif channels == 5:
                fused = build_5ch(vis, ir, dep)
            else:
                fused = build_3ch(vis, ir, dep)
            dst = out / "images" / split / f"{stem}{ext}"
            if is_multichannel:  # 多页 TIFF：CHW，每通道一页
                written = cv2.imwritemulti(str(dst), np.transpose(fused, (2, 0, 1)))
            else:
                written = cv2.imwrite(str(dst), fused)
            if not written:
                raise IOError(f"写入失败: {dst}")
            if not args.no_labels:
                repaired_labels += sanitize_label_file(
                    data_root / "labels" / f"{stem}.txt",
                    out / "labels" / split / f"{stem}.txt",
                )

    cfg = {"path": str(out.resolve()), "train": "images/train",
           "val": "images/val" if "val" in splits else None,
           "names": NAMES}
    if is_multichannel:
        cfg["channels"] = channels
    repeat_stats = None
    if not args.no_labels and getattr(args, "rare_repeat", False):
        repeat_stats = write_weighted_train_list(out, splits["train"], ext,
                                                  data_root / "labels")
        cfg["train"] = "train_weighted.txt"
    if args.no_labels:
        cfg["test"] = "images/test"
        cfg.pop("train"), cfg.pop("val")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    manifest = {
        "data_root": str(data_root.resolve()),
        "mode": args.mode,
        "network_channel_order": (
            ["R", "G", "B", "IR", "Depth", "DepthValid"] if channels == 6
            else ["R", "G", "B", "IR", "Depth"] if channels == 5
            else ["R", "IR", "Depth"]
        ),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "repaired_label_rows": repaired_labels,
        "source_split_manifest": str(Path(split_manifest).resolve()) if not args.no_labels and split_manifest else None,
        "rare_repeat": bool(not args.no_labels and getattr(args, "rare_repeat", False)),
        "repeat_stats": repeat_stats,
        "splits": splits,
    }
    (out / "split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[prepare] {len(stems)} 组 → {out} (mode={args.mode}, "
          f"splits={ {k: len(v) for k, v in splits.items()} }, "
          f"repaired_labels={repaired_labels})")
    return data_yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="./data", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["3ch", "5ch", "6ch"], default="3ch")
    ap.add_argument("--no-labels", action="store_true", help="测试集模式（无标注）")
    ap.add_argument("--all-train", action="store_true",
                    help="最终训练模式：全部有标注样本进入 train，data.yaml 的 val=null")
    ap.add_argument("--val-ratio", default=0.1, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--split-manifest", type=Path,
                    help="使用已有 JSON 划分清单；优先于 --val-ratio/--seed")
    ap.add_argument("--rare-repeat", action="store_true",
                    help="训练列表中 ball×2、tricycle×3；验证集不重复")
    ap.add_argument("--refresh-rare-list", action="store_true",
                    help="只对现有派生数据启用稀有类重复，不重建图像")
    ap.add_argument("--overwrite", action="store_true", help="清理该输出目录中的旧 images/labels 后重建")
    main(ap.parse_args())
