"""提交前自检与打包：非法行清洗、每图≤100框、缺失TXT兜底、打zip。

规则依据任务第九节：非法类别/坐标/置信度 → 该预测无效不参与运算；
每图最大100框（超限按置信度截断）；每图必须有同名TXT（含空文件）。
"""
import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

NUM_CLASSES = 12
FIELDS = 6  # class cx cy w h conf
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def clean_line(parts: list) -> list | None:
    """清洗一行预测；不可挽救返回 None。"""
    if len(parts) != FIELDS:
        return None
    try:
        cls_value = float(parts[0])
        if not math.isfinite(cls_value) or not cls_value.is_integer():
            return None
        cls = int(cls_value)
        vals = [float(x) for x in parts[1:]]
    except (TypeError, ValueError, OverflowError):
        return None
    if not 0 <= cls < NUM_CLASSES:
        return None
    cx, cy, w, h, conf = vals
    # 竞赛要求非法类别、坐标或置信度的预测无效：严格丢弃，不能裁剪后伪装成合法框。
    if not all(math.isfinite(v) for v in vals):
        return None
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
            0.0 < w <= 1.0 and 0.0 < h <= 1.0 and 0.0 <= conf <= 1.0):
        return None
    return [cls, cx, cy, w, h, conf]


def validate_and_pack(pred_dir: Path, image_dir: Path, out_zip: Path,
                      max_det: int = 100, staging: Path | None = None,
                      strict: bool = False) -> dict:
    pred_dir, image_dir = Path(pred_dir), Path(image_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"预测目录不存在: {pred_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not 1 <= max_det <= 100:
        raise ValueError(f"竞赛规定每图最多100框，max_det 必须在[1,100]，实际 {max_det}")
    if strict:
        expected = [p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        actual = {p.stem for p in pred_dir.glob("*.txt")}
        if len(expected) != len(set(expected)) or set(expected) != actual:
            raise ValueError("Strict packaging: missing/extra/ambiguous TXT files")
        for txt in pred_dir.glob("*.txt"):
            lines = [line for line in txt.read_text().splitlines() if line.strip()]
            if len(lines) > max_det or any(clean_line(line.split()) is None for line in lines):
                raise ValueError(f"Strict packaging: invalid predictions in {txt}")
    staging = Path(staging) if staging else Path(out_zip).parent / "staging"
    if staging.exists():
        shutil.rmtree(staging)  # 清掉上次运行残留，防止混入正式提交包
    staging.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "kept": 0, "dropped": 0, "created_empty": 0,
             "extra_txt": 0, "truncated": 0, "at_max_det": 0}

    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"图像目录中没有支持的图片: {image_dir}")
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise RuntimeError("图像目录存在同名不同扩展名文件，无法唯一映射预测TXT")
    expected = set(stems)
    stats["extra_txt"] = len([p for p in pred_dir.glob("*.txt") if p.stem not in expected])

    for img in images:
        stats["images"] += 1
        stem = img.stem
        src = pred_dir / f"{stem}.txt"
        rows = []
        if src.exists():
            for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                cleaned = clean_line(parts)
                if cleaned is None:
                    if parts:  # 非空行才计数
                        stats["dropped"] += 1
                    continue
                rows.append(cleaned)
        else:
            stats["created_empty"] += 1
        rows.sort(key=lambda r: r[5], reverse=True)
        stats["truncated"] += max(0, len(rows) - max_det)
        rows = rows[:max_det]
        stats["at_max_det"] += int(len(rows) == max_det)
        stats["kept"] += len(rows)
        dst = staging / f"{stem}.txt"
        dst.write_text("".join(
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n"
            for c, cx, cy, w, h, conf in rows), encoding="utf-8")

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for txt in sorted(staging.glob("*.txt")):
            z.write(txt, txt.name)
    digest = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    stats["zip_bytes"] = out_zip.stat().st_size
    stats["zip_sha256"] = digest
    print(f"[check] images={stats['images']} kept={stats['kept']} "
          f"dropped={stats['dropped']} truncated={stats['truncated']} "
          f"created_empty={stats['created_empty']} extra_txt={stats['extra_txt']} "
          f"at_max_det={stats['at_max_det']} sha256={digest} → {out_zip}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path, help="测试图像目录（用于枚举全部图片名）")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-det", default=100, type=int)
    ap.add_argument("--audit-json", type=Path)
    ap.add_argument("--strict", action="store_true", help="拒绝缺失文件、非法行和超过100框")
    args = ap.parse_args()
    stats = validate_and_pack(args.pred_dir, args.image_dir, args.out, args.max_det, strict=args.strict)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
