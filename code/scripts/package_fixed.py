"""固定配置打包：用于「无验证集」实验（无法做平台选参时）。

与 finalize_rf48b.py 的区别：
  · 不做验证集推理、不扫 conf×NMS（因为压根没有验证集）；
  · conf / NMS / imgsz 由命令行显式给定（沿用已验证的 0.35 / 0.6 / 训练尺寸）；
  · 可以指定任意检查点文件（例如 epoch-070.pt），而不是固定的 best.pt。

用法：
  python -m scripts.package_fixed \
      --run-dir runs/formal/rf48c/run-all2000 --checkpoint epoch-070.pt \
      --test-manifest runs/formal/rf48c/phase2-test-manifest.json \
      --image-dir "<DATA_ROOT>/visible" \
      --conf 0.35 --nms 0.6 --tag all2000-ep70 --zip-prefix phase2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="训练输出目录（含 identity.json）")
    ap.add_argument("--checkpoint", default="best.pt", help="检查点文件名，如 epoch-070.pt")
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--image-dir", required=True, help="测试图像目录（用于枚举图片名）")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=None, help="默认取运行身份里的训练尺寸")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zip-prefix", default="phase1")
    ap.add_argument("--zip-dir", default=None)
    ap.add_argument("--out-root", default=None, help="中间产物根目录，默认在训练目录下")
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    weights = run_dir / args.checkpoint
    assert weights.exists(), f"缺少检查点 {weights}"
    identity = json.loads((run_dir / "identity.json").read_text())
    size = args.imgsz or int(identity["config"]["imgsz"])
    out_root = Path(args.out_root) if args.out_root else run_dir / f"package-{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    raw = out_root / f"test-{size}-raw"
    run("src.predict_rfdetr", "--weights", weights, "--sample-manifest", args.test_manifest,
        "--out-dir", raw, "--side", "test", "--imgsz", size, "--conf", 0.001)

    merged = out_root / f"{args.tag}-predictions"
    run("scripts.apply_nms", "--source", raw, "--dest", merged,
        "--conf", args.conf, "--nms", args.nms)

    from src.check_submit import validate_and_pack
    from src.trimodal_data import sha256
    zip_dir = Path(args.zip_dir) if args.zip_dir else (ROOT / "runs/formal/submissions")
    zip_path = zip_dir / f"{args.zip_prefix}-{args.tag}-i{size}-c{str(args.conf).replace('.', 'p')}" \
                         f"-n{str(args.nms).replace('.', 'p')}.zip"
    stats = validate_and_pack(merged, Path(args.image_dir), zip_path,
                              staging=out_root / f"{args.tag}-staging", strict=True)
    assert stats["images"] == 1000, stats
    assert stats["created_empty"] == stats["dropped"] == stats["extra_txt"] == 0, stats

    report = {
        "zip": str(zip_path), "weights": str(weights), "weights_sha256": sha256(weights),
        "config": {"imgsz": size, "conf": args.conf, "nms_iou": args.nms, "max_det": 100, "tta": False},
        "stats": stats, "test_manifest": args.test_manifest,
        "note": "无验证集实验：未做平台选参，使用历史验证过的固定配置",
    }
    (out_root / "package-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("zip", "weights_sha256")}, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
