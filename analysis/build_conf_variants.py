"""从已存的原始预测批量生成不同工作点的提交包（纯 CPU，不需要 GPU）。

原理（见 scripts/apply_nms.py 文档）：output_rows 的语义是「先取全局 top-100 再按 conf 过滤」，
因此 top100 ∩ {conf >= t} 与「直接用阈值 t 推理」的集合完全相同，NMS 只删框不造框。
所以换 conf / NMS 只是纯后处理，可以从 conf=0.001 的原始预测直接重算。

用途：评测分 54.819，比基线掉 5.99 分，且评测集预测框数比基线多 33%（说明工作点在目标域上可能失配）。
先用同一模型扫工作点，把「模型问题」和「阈值问题」分开。
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_MANIFEST = "runs/formal/rf48c/phase2-test-manifest.json"
SUBMISSIONS = ROOT / "runs/formal/submissions"


def stems_of(manifest):
    from src.trimodal_data import load_manifest
    _, samples = load_manifest(ROOT / manifest, "test")
    return {r["stem"] for r in samples}


def make_zip(pred_dir, zip_path):
    files = sorted(Path(pred_dir).glob("*.txt"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert len(names) == len(files), "条目数不一致"
        assert all("/" not in n for n in names), "必须全部在压缩包根目录"
    data = zip_path.read_bytes()
    return {"zip": str(zip_path), "files": len(files), "zip_bytes": len(data),
            "zip_sha256": hashlib.sha256(data).hexdigest()}


def stats_of(pred_dir):
    rows, empty, at_cap = 0, 0, 0
    for f in Path(pred_dir).glob("*.txt"):
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        rows += len(lines)
        if not lines:
            empty += 1
        if len(lines) >= 100:
            at_cap += 1
    return {"boxes": rows, "empty": empty, "at_max_det": at_cap}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="conf=0.001 的原始预测目录")
    ap.add_argument("--prefix", required=True, help="输出包名前缀，如 ep67-i1280")
    ap.add_argument("--confs", default="0.45,0.50,0.55,0.60")
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    expect = stems_of(TEST_MANIFEST)
    src = ROOT / args.source
    got = {p.stem for p in src.glob("*.txt")}
    if got != expect:
        raise SystemExit(f"原始预测与评测集清单不匹配 missing={len(expect-got)} extra={len(got-expect)}")

    work = ROOT / "runs/formal/rf48c/conf-variants"
    work.mkdir(parents=True, exist_ok=True)
    table = []
    for conf in [float(c) for c in args.confs.split(",")]:
        tag = f"{args.prefix}-c{str(conf).replace('.', 'p')}-n{str(args.nms).replace('.', 'p')}"
        dest = work / tag
        if dest.exists():
            shutil.rmtree(dest)
        cmd = [sys.executable, "-u", "-m", "scripts.apply_nms",
               "--source", str(src), "--dest", str(dest),
               "--conf", str(conf), "--nms", str(args.nms)]
        print(f"[conf {conf}] -> {dest.name}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        st = stats_of(dest)
        info = make_zip(dest, SUBMISSIONS / f"phase2-{tag}.zip")
        info.update({"conf": conf, "nms": args.nms, **st})
        table.append(info)
        print(f"   框数 {st['boxes']:>6}  空图 {st['empty']:>3}  满100框 {st['at_max_det']:>3}  "
              f"zip {info['zip_bytes']} B  sha256 {info['zip_sha256'][:16]}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'conf':>6}{'NMS':>6}{'框数':>9}{'空图':>7}{'满框':>7}   sha256")
    for r in table:
        print(f"{r['conf']:>6}{r['nms']:>6}{r['boxes']:>9}{r['empty']:>7}{r['at_max_det']:>7}   "
              f"{r['zip_sha256'][:16]}")
    out = work / f"{args.prefix}-conf-sweep.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
