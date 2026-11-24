"""独立校验一个提交 ZIP 是否符合评测集细则（不改动、只读取）。

规则依据（格式规范第九节 + 附件2）：
  · 每张测试图必须有同名 TXT（空文件合法）；
  · 每行 6 个字段：class_id cx cy w h conf，均为归一化值；
  · class_id ∈ [0, 11]；cx,cy,conf ∈ [0,1]；0 < w,h ≤ 1；
  · 非法类别/坐标/置信度的预测无效，不参与运算（不是裁剪，是丢弃）；
  · 每图最多 100 框。
"""
import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

NUM_CLASSES = 12
FIELDS = 6
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CL = ["person", "boat", "animal", "seat", "sign", "bicycle",
      "car", "ball", "light", "garbage", "uav", "tricycle"]
import re
STRICT_RE = re.compile(rb"^\d+ \d\.\d{6} \d\.\d{6} \d\.\d{6} \d\.\d{6} \d\.\d{6}$")


def clean_line(parts):
    """与 src/check_submit.py 完全一致的清洗逻辑。"""
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
    if not all(math.isfinite(v) for v in vals):
        return None
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
            0.0 < w <= 1.0 and 0.0 < h <= 1.0 and 0.0 <= conf <= 1.0):
        return None
    return [cls, cx, cy, w, h, conf]


def expected_stems(manifest):
    m = json.loads(Path(manifest).read_text())
    return {s["stem"] for s in m["samples"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--manifest", default="results-snapshot/phase2-test-manifest.json")
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()

    zp = Path(args.zip)
    expect = expected_stems(args.manifest)
    print(f"校验对象: {zp}")
    print(f"           {zp.stat().st_size} 字节")
    print(f"评测集图片（基准）: {len(expect)} 张")
    print()

    problems = []
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        dirs = [n for n in names if n.endswith("/")]
        non_txt = [n for n in names if not n.endswith("/") and
                   Path(n).suffix.lower() != ".txt"]
        nested = [n for n in names if "/" in n.strip("/")]
        files = [n for n in names if not n.endswith("/")]

        print("── 结构 ──")
        print(f"  条目总数        : {len(names)}")
        print(f"  目录条目        : {len(dirs)}")
        print(f"  非 TXT 文件     : {len(non_txt)}" + (f"  {non_txt[:5]}" if non_txt else "  ✓"))
        print(f"  含子目录路径    : {len(nested)}" + (f"  {nested[:5]}" if nested else "  ✓ 全部在根目录"))
        if non_txt:
            problems.append(f"存在非 TXT 文件 {len(non_txt)} 个")
        if nested:
            problems.append(f"存在嵌套路径条目 {len(nested)} 个")

        got = {Path(n).stem for n in files}
        missing, extra = expect - got, got - expect
        print(f"  TXT 文件数      : {len(files)}")
        print(f"  缺失（应有未有）: {len(missing)}" + (f"  {sorted(missing)[:5]}" if missing else "  ✓"))
        print(f"  多余（不应有）  : {len(extra)}" + (f"  {sorted(extra)[:5]}" if extra else "  ✓"))
        if missing:
            problems.append(f"缺失 {len(missing)} 个 TXT")
        if extra:
            problems.append(f"多余 {len(extra)} 个 TXT")

        print()
        print("── 逐行内容 ──")
        total_rows = bad_rows = 0
        empty_files = 0
        over_cap = []
        per_class = Counter()
        confs = []
        worst = []
        for n in sorted(files):
            raw = z.read(n).decode("utf-8", "replace")
            lines = [l for l in raw.splitlines() if l.strip()]
            total_rows += len(lines)
            if not lines:
                empty_files += 1
            if len(lines) > args.max_det:
                over_cap.append((n, len(lines)))
            nbad = 0
            for l in lines:
                r = clean_line(l.split())
                if r is None:
                    nbad += 1
                    if len(worst) < 5:
                        worst.append((n, l[:80]))
                else:
                    per_class[r[0]] += 1
                    confs.append(r[5])
            bad_rows += nbad

        print(f"  总行数          : {total_rows}")
        print(f"  非法行          : {bad_rows}" + ("  ✓" if bad_rows == 0 else "  ✗"))
        print(f"  空 TXT          : {empty_files} （细则允许）")
        print(f"  超过 {args.max_det} 框的图  : {len(over_cap)}" +
              (f"  {over_cap[:5]}" if over_cap else "  ✓"))
        if bad_rows:
            problems.append(f"{bad_rows} 行非法（示例 {worst}）")
        if over_cap:
            problems.append(f"{len(over_cap)} 张图超过每图 100 框上限")

        if confs:
            confs.sort()
            print(f"  置信度 min/中位/max: {confs[0]:.6f} / "
                  f"{confs[len(confs)//2]:.6f} / {confs[-1]:.6f}")

        print()
        print("── 编码与数字格式 ──")
        crlf = bom = nonascii = trailing = blank = badfmt = 0
        blob_files = [n for n in files]
        for n in blob_files:
            b = z.read(n)
            if b"\r" in b:
                crlf += 1
            if b.startswith(b"\xef\xbb\xbf"):
                bom += 1
            if any(c > 127 for c in b):
                nonascii += 1
            if b and not b.endswith(b"\n"):
                trailing += 1
            if b"\n\n" in b:
                blank += 1
            for line in b.split(b"\n"):
                if line.strip() and not STRICT_RE.match(line):
                    badfmt += 1
        for label, val, ok in (("含 \\r（Windows 换行）", crlf, crlf == 0),
                               ("UTF-8 BOM", bom, bom == 0),
                               ("非 ASCII 字节", nonascii, nonascii == 0),
                               ("末行缺换行", trailing, trailing == 0),
                               ("连续空行", blank, blank == 0),
                               ("不符合严格数字格式的行", badfmt, badfmt == 0)):
            print(f"  {label:<24}: {val}" + ("  ✓" if ok else "  ✗"))
        if crlf or bom or nonascii or trailing or blank or badfmt:
            problems.append(
                f"编码/格式异常: crlf={crlf} bom={bom} nonascii={nonascii} "
                f"trailing={trailing} blank={blank} badfmt={badfmt}")

        print()
        print("── 类别分布 ──")
        for i, c in enumerate(CL):
            print(f"  {i:>2} {c:<10} {per_class.get(i, 0):>7}")

    print()
    print("=" * 72)
    if problems:
        print("结论: ✗ 不符合细则，需要修正：")
        for p in problems:
            print("   -", p)
    else:
        print("结论: ✓ 完全符合提交格式规范，可以提交")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
