"""生成可复现的场景隔离、多标签分层 train/val 清单。"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from src.prepare_data import NAMES, label_classes, scene_key
except ModuleNotFoundError:  # 兼容 `python src/make_splits.py`
    from prepare_data import NAMES, label_classes, scene_key


def stratified_group_split(labels_dir: Path, val_ratio: float, seed: int) -> dict:
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob("*.txt"))
    if not files or not 0 < val_ratio < 1:
        raise ValueError("需要非空标签目录且 val_ratio 位于 (0,1)")

    groups: dict[str, list[str]] = defaultdict(list)
    sample_classes = {}
    for path in files:
        groups[scene_key(path.stem)].append(path.stem)
        sample_classes[path.stem] = label_classes(path)

    group_classes = {
        group: Counter(c for stem in stems for c in sample_classes[stem])
        for group, stems in groups.items()
    }
    totals = Counter(c for classes in sample_classes.values() for c in classes)
    target_images = round(len(files) * val_ratio)
    targets = {c: max(1.0, totals[c] * val_ratio) for c in NAMES}
    rng = random.Random(seed)
    candidates = list(groups)
    rng.shuffle(candidates)
    tie_break = {group: rng.random() for group in candidates}
    chosen, current, current_images = set(), Counter(), 0

    def objective(group: str) -> tuple[float, float, float, str]:
        next_images = current_images + len(groups[group])
        gain = sum(
            min(group_classes[group][c], max(0.0, targets[c] - current[c])) / targets[c]
            for c in NAMES
        ) / max(len(groups[group]), 1)
        overshoot = max(0, next_images - target_images)
        # min()：优先最大化各类相对缺口收益，再避免超过目标图数。
        return -gain, overshoot, tie_break[group], group

    while current_images < target_images and candidates:
        group = min(candidates, key=objective)
        candidates.remove(group)
        chosen.add(group)
        current.update(group_classes[group])
        current_images += len(groups[group])

    # 稀有类覆盖门禁：缺失类别加入最小的相关场景；保留至少一个相关场景在训练侧。
    for cls in NAMES:
        if current[cls] == 0:
            pool = [g for g in candidates if group_classes[g][cls]]
            if not pool:
                raise RuntimeError(f"类别 {cls} 无法进入验证集")
            group = min(pool, key=lambda g: (len(groups[g]), g))
            candidates.remove(group)
            chosen.add(group)
            current.update(group_classes[group])
            current_images += len(groups[group])
        all_groups = [g for g in groups if group_classes[g][cls]]
        if all(g in chosen for g in all_groups):
            removable = [g for g in all_groups if sum(group_classes[h][cls] for h in chosen if h != g) > 0]
            if not removable:
                raise RuntimeError(f"类别 {cls} 无法同时覆盖训练与验证")
            group = min(removable, key=lambda g: (len(groups[g]), g))
            chosen.remove(group)
            current.subtract(group_classes[group])
            current_images -= len(groups[group])

    val = sorted(stem for group in chosen for stem in groups[group])
    train = sorted(stem for group in groups if group not in chosen for stem in groups[group])
    train_groups = {scene_key(s) for s in train}
    val_groups = {scene_key(s) for s in val}
    train_classes = set().union(*(sample_classes[s] for s in train))
    val_classes = set().union(*(sample_classes[s] for s in val))
    assert not train_groups & val_groups
    assert train_classes == val_classes == set(NAMES)
    return {
        "kind": "stratified_group_90_10",
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {"all": len(files), "train": len(train), "val": len(val)},
        "class_image_counts": {
            "all": {str(c): totals[c] for c in NAMES},
            "train": {str(c): sum(c in sample_classes[s] for s in train) for c in NAMES},
            "val": {str(c): sum(c in sample_classes[s] for s in val) for c in NAMES},
        },
        "splits": {"train": train, "val": val},
    }


def main(args: argparse.Namespace) -> Path:
    manifest = stratified_group_split(args.labels_dir, args.val_ratio, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] seed={args.seed} counts={manifest['counts']} -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--val-ratio", default=0.1, type=float)
    ap.add_argument("--out", required=True, type=Path)
    main(ap.parse_args())
