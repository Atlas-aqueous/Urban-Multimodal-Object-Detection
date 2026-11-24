"""RF48b 收尾流水线：平台选参 -> 测试集推理 -> 严格打包 -> 交付记录。

设计要点
--------
1. 分阶段且幂等：每阶段产物已存在即跳过，可反复运行、可从任意阶段续跑。
2. 选参用「平滑后的平台最优」而非单点 argmax：195 张验证集上直接取 117 个配置的
   最大值会过拟合噪声，故按 3x3 邻域均值选点。
3. 只读既有资产、只写本阶段自己的目录；不删除、不上传。
4. 交付记录同时给出双口径与官方标定结论（官方 ≈ 本地 Ultralytics - 0.2 ± 1.0）。
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/formal/rf48b"
RUN = BASE / "run1280"
FINAL = BASE / "final"
ZIP_DIR = ROOT / "runs/formal/submissions"
TRAIN_MANIFEST = "runs/formal/rf48/train-manifest.json"
TEST_MANIFEST = "runs/formal/rf48/test-manifest.json"
SPLIT = "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
CURRENT_BEST_LOCAL = 61.7232      # ep65 包本地分，官方 60.7360（2026-09-20 更新）
SUBMIT_GATE_LOCAL = 59.1          # 本地需 >= 此值才值得占用一次提交机会
BAND_CONF = (0.40, 0.45, 0.50, 0.55, 0.60)
BAND_NMS = (None, 0.5, 0.55, 0.6, 0.65)


def sh(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def current_weights_sha256():
    from src.trimodal_data import sha256
    return sha256(RUN / "best.pt")


def guard_cache(out, force):
    """预测缓存必须与当前 best.pt 一致，否则报错而非静默复用旧缓存的指标。"""
    if force and out.exists():
        shutil.rmtree(out)
        return
    identity = out / "identity.json"
    if identity.exists():
        cached = read_json(identity).get("weights_sha256")
        current = current_weights_sha256()
        if cached != current:
            raise RuntimeError(
                f"{out} 的缓存权重 {str(cached)[:12]}… 与当前 best.pt {current[:12]}… 不一致；"
                f"请加 --force（清理本流水线自己的派生缓存）或换 --run-name。")


def run_size():
    """推理尺寸必须与训练尺寸一致 —— 从运行身份里读，绝不硬编码。

    教训：本流水线最初为 1280 实验写成，把 1280 写死在推理与命名里；
    1536 实验复用时就发生了训练/推理分辨率不匹配，把 1536 模型低估了约 0.7 分。
    """
    identity = read_json(RUN / "identity.json")
    size = int(identity["config"]["imgsz"])
    return size


def stage_val(size, force=False):
    out = FINAL / f"val-{size}"
    if (out / "metrics.json").exists() and not force:
        guard_cache(out, force=False)
        print(f"[val] 已存在且权重一致，跳过 {out}")
        return out
    guard_cache(out, force)
    sh("src.predict_rfdetr", "--weights", RUN / "best.pt", "--sample-manifest", TRAIN_MANIFEST,
       "--out-dir", out, "--side", "val", "--imgsz", size, "--conf", 0.001)
    return out


def stage_sweep(val_dir, run_name, force=False):
    out = FINAL / "sweep"
    if (out / "sweep.json").exists() and not force:
        print(f"[sweep] 已存在，跳过 {out}")
        return out
    sh("scripts.sweep_rf48_postprocess", "--manifest", TRAIN_MANIFEST,
       "--dirs", f"{run_name}={val_dir.relative_to(ROOT)}", "--out", out, "--workers", 32)
    return out


def smooth_select(sweep_path, dir_name=None):
    """3x3 邻域平滑后的平台最优；同时返回原始 argmax 与平台统计。"""
    data = read_json(sweep_path / "sweep.json")
    rows = data["results"]
    if dir_name is not None:
        rows = [r for r in rows if r["dir"] == dir_name]
        assert rows, f"sweep 中没有目录 {dir_name}"
    confs = sorted({r["conf"] for r in rows})
    ious = sorted({r["nms_iou"] for r in rows}, key=lambda v: (v is not None, v if v is not None else -1))
    grid = {(r["conf"], r["nms_iou"]): r for r in rows}
    ci = {c: i for i, c in enumerate(confs)}
    ni = {n: i for i, n in enumerate(ious)}
    smoothed = []
    for (conf, nms), row in grid.items():
        i, j = ci[conf], ni[nms]
        neighbourhood = [grid[(confs[a], ious[b])].get("ultralytics_map")
                         for a in range(max(0, i - 1), min(len(confs), i + 2))
                         for b in range(max(0, j - 1), min(len(ious), j + 2))
                         if (confs[a], ious[b]) in grid]
        smoothed.append({**row, "neighbourhood_mean": statistics.mean(neighbourhood)})
    best_smooth = max(smoothed, key=lambda r: r["neighbourhood_mean"])
    best_raw = max(rows, key=lambda r: r["ultralytics_map"])
    band = [grid[(c, n)].get("ultralytics_map") for c in BAND_CONF for n in BAND_NMS
            if (c, n) in grid]
    return {"selected": {"conf": best_smooth["conf"], "nms_iou": best_smooth["nms_iou"],
                         "ultralytics_map": best_smooth["ultralytics_map"],
                         "rule_map": best_smooth["rule_map"],
                         "neighbourhood_mean": best_smooth["neighbourhood_mean"],
                         "ultralytics_per_class": best_smooth["ultralytics_per_class"],
                         "tiny_recall50": best_smooth["tiny_recall50"],
                         "at_max_det": best_smooth["at_max_det"]},
            "raw_argmax": {"conf": best_raw["conf"], "nms_iou": best_raw["nms_iou"],
                           "ultralytics_map": best_raw["ultralytics_map"]},
            "band": {"confs": list(BAND_CONF), "nms": list(BAND_NMS),
                     "mean": statistics.mean(band), "min": min(band), "max": max(band)}}


def stage_test(size, conf, manifest=TEST_MANIFEST, tag="test", force=False):
    out = FINAL / f"{tag}-{size}-raw"
    if (out / "identity.json").exists() and not force:
        guard_cache(out, force=False)
        print(f"[test] 已存在且权重一致，跳过 {out}")
        return out
    guard_cache(out, force)
    sh("src.predict_rfdetr", "--weights", RUN / "best.pt", "--sample-manifest", manifest,
       "--out-dir", out, "--side", "test", "--imgsz", size, "--conf", conf)
    return out


def stage_pack(test_dir, conf, nms, selection, tag, size, image_dir=None,
               zip_prefix="phase1", force=False):
    label = f"{tag}-i{size}-c{str(conf).replace('.', 'p')}-n{str(nms).replace('.', 'p')}"
    pred_dir = FINAL / f"{label}-predictions"
    zip_path = ZIP_DIR / f"{zip_prefix}-{label}.zip"
    if zip_path.exists() and not force:
        print(f"[pack] 已存在，跳过 {zip_path}")
        return zip_path, pred_dir
    suffix = "none" if nms is None else str(nms)
    sh("scripts.apply_nms", "--source", test_dir, "--dest", pred_dir,
       "--conf", conf, "--nms", suffix)
    from src.check_submit import validate_and_pack
    stats = validate_and_pack(pred_dir, image_dir or (ROOT / "formal_data/phase1_test/visible"), zip_path,
                              staging=FINAL / f"{label}-staging", strict=True)
    assert stats["images"] == 1000, stats
    assert stats["created_empty"] == stats["dropped"] == stats["extra_txt"] == 0, stats
    write_json(FINAL / f"{label}-package.json",
               {"zip": str(zip_path), "stats": stats, "selection": selection})
    return zip_path, pred_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run-name", default="run1280", help="BASE 下的训练目录名")
    ap.add_argument("--tag", default=None, help="提交包名标签，默认取 run-name")
    ap.add_argument("--stop-after", choices=["select", "test", "pack"], default="pack",
                    help="流水线提前停止点；dry-run 默认 select")
    ap.add_argument("--dry-run", action="store_true",
                    help="写入 dryrun-<run-name>/ 并在选参后停止，不产出提交包")
    ap.add_argument("--test-manifest", default=TEST_MANIFEST,
                    help="测试集清单；评测阶段请指向评测阶段 manifest")
    ap.add_argument("--test-tag", default="test",
                    help="测试集预测缓存目录标签；评测阶段用 phase2 以区别于基线阶段缓存")
    ap.add_argument("--zip-prefix", default="phase1",
                    help="提交包文件名前缀；评测阶段用 phase2")
    ap.add_argument("--image-dir", default=None,
                    help="用于枚举全部图片名的测试图像目录；评测阶段请指向评测阶段 visible/")
    ap.add_argument("--zip-dir", default=None,
                    help="提交包输出目录；默认 runs/formal/submissions，打包演练可指向临时目录")
    args = ap.parse_args()
    global RUN, FINAL, ZIP_DIR
    RUN = BASE / args.run_name
    FINAL = BASE / (f"dryrun-{args.run_name}" if args.dry_run else f"final-{args.run_name}")
    if args.zip_dir:
        ZIP_DIR = Path(args.zip_dir)
    if args.dry_run and args.stop_after == "pack":
        args.stop_after = "select"
    tag = args.tag or args.run_name
    assert (RUN / "best.pt").exists(), f"缺少权重 {RUN / 'best.pt'}"
    FINAL.mkdir(parents=True, exist_ok=True)

    size = run_size()
    print(f"[run] 推理尺寸取自运行身份：{size}", flush=True)
    val_dir = stage_val(size, args.force)
    sweep_dir = stage_sweep(val_dir, args.run_name, args.force)
    selection = smooth_select(sweep_dir, args.run_name)
    write_json(FINAL / "selection.json", selection)
    sel = selection["selected"]
    print(json.dumps(selection, ensure_ascii=False, indent=1)[:1200], flush=True)
    if args.stop_after == "select":
        print("[stop] 已在选参阶段停止（dry-run / --stop-after select）", flush=True)
        return

    test_dir = stage_test(size, 0.001, manifest=args.test_manifest,
                          tag=args.test_tag, force=args.force)
    if args.stop_after == "test":
        print("[stop] 已在测试集推理后停止", flush=True)
        return
    zip_path, pred_dir = stage_pack(test_dir, sel["conf"], sel["nms_iou"], selection, tag,
                                    size, image_dir=args.image_dir,
                                    zip_prefix=args.zip_prefix, force=args.force)

    package_json = FINAL / (f"{tag}-i{size}-c{str(sel['conf']).replace('.', 'p')}"
                            f"-n{str(sel['nms_iou']).replace('.', 'p')}-package.json")
    local = sel["ultralytics_map"] * 100
    delivery = {
        "candidate_zip": str(zip_path),
        "zip_sha256": read_json(package_json)["stats"]["zip_sha256"],
        "weights": str(RUN / "best.pt"),
        "run_name": args.run_name,
        "config": {"canvas": f"{size}x{size} square letterbox", "epochs": 100,
                   "conf": sel["conf"], "nms_iou": sel["nms_iou"], "max_det": 100, "tta": False},
        "local": {"ultralytics_map": sel["ultralytics_map"], "rule_map": sel["rule_map"],
                  "tiny_recall50": sel["tiny_recall50"],
                  "ultralytics_per_class": sel["ultralytics_per_class"]},
        "platform": selection["band"],
        "comparison": {"current_best_local": CURRENT_BEST_LOCAL,
                       "gain_over_current_best": local - CURRENT_BEST_LOCAL,
                       "current_official_best": 60.7360,
                       "submit_gate_local": SUBMIT_GATE_LOCAL,
                       "passes_submit_gate": local >= SUBMIT_GATE_LOCAL},
        "calibration": "官方 ≈ 本地 Ultralytics - 0.2 ± 1.0（E0/E9/RF48 三点）",
        "scope": "A 划分单验证集；官方成绩未知；不自动上传",
    }
    write_json(FINAL / "delivery.json", delivery)
    print("\n[delivery] " + json.dumps(
        {k: delivery[k] for k in ("zip_sha256", "local", "comparison")},
        ensure_ascii=False)[:900], flush=True)


if __name__ == "__main__":
    main()
