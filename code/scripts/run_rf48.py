"""Single-GPU durable queue: probe -> RGB10 -> MM60 -> validation -> strict ZIP.

Run from the project root in screen. Never uploads to the leaderboard.
"""
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.trimodal_data import write_json, sha256
from src.check_submit import validate_and_pack
from src.evaluate_txt import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/formal/rf48"
SPLIT = "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
TRAIN = "runs/formal/rf48/train-manifest.json"
TEST = "runs/formal/rf48/test-manifest.json"
E9 = "runs/formal/e9-yolov8l-p2-1280-aggressive-b4/weights/best.pt"
RESTORE = "runs/formal/submissions/phase1-e9-l-p2-i1280-c0p4-n0p4.zip"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(stage, module, *args, timeout=8*3600):
    deadline = datetime.datetime.fromisoformat("2026-09-18T02:59:47+00:00").timestamp()
    remaining = deadline - time.time()
    if stage in {"rgb-control", "mm-candidate", "e9-fallback"}:
        remaining -= 6 * 3600  # reserve validation, inference and packaging time
    if remaining <= 0:
        raise RuntimeError("48-hour execution budget exhausted; keep E9 restore package")
    timeout = min(timeout, remaining)
    write_json(OUT / "pipeline-status.json", {"stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "log": str(OUT / (stage + ".log"))})
    with (OUT / (stage + ".log")).open("a") as stream:
        stream.write(f"\nSTART {now()} {module} {args}\n")
        stream.flush()
        print("START", stage, now(), flush=True)
        subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    print("DONE", stage, now(), flush=True)


def screen_gate(candidate, baseline):
    total = candidate["rule_map"] - baseline["rule_map"]
    rare = sum(candidate["rule_per_class"][c] - baseline["rule_per_class"][c] for c in (7, 11))/2
    return {"passed": total >= .005 or (rare >= .03 and total >= -.002),
            "total_gain": total, "rare_mean_gain": rare}


def package(predictions, label, weights, metrics):
    zip_path = ROOT / "runs/formal/submissions" / f"phase1-{label}.zip"
    # Dedicated staging path, never shared with older submit workflows.
    stats = validate_and_pack(predictions, ROOT / "formal_data/phase1_test/visible", zip_path,
                              staging=OUT / (label + "-staging"), strict=True)
    assert stats["images"] == 1000 and stats["created_empty"] == stats["dropped"] == stats["extra_txt"] == 0
    baseline = json.loads((OUT / "baseline-summary.json").read_text())["e9-low"]
    result = {"candidate_zip": str(zip_path), "candidate_weights": str(weights),
              "weights_sha256": sha256(weights), "package": stats, "validation": metrics,
              "gate": screen_gate(metrics, baseline), "restore_zip": RESTORE,
              "restore_sha256": sha256(ROOT / RESTORE), "official_reference_user_reported": 55.5370,
              "scope": "A split only; hidden test score unknown; no automatic upload"}
    result["train_manifest_sha256"] = sha256(ROOT / TRAIN)
    result["test_manifest_sha256"] = sha256(ROOT / TEST)
    result["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p)
                               for p in sorted((OUT / "mm-candidate/source").glob("*.py"))}
    result["public_pretrained_sha256"] = sha256(ROOT / "weights/public/rf-detr-medium.pth")
    write_json(OUT / "delivery.json", result)
    (OUT / "DELIVERY.md").write_text(
        f"# RF48 候选交付\n\n候选包：`{zip_path}`\n\n"
        f"A 划分细则复刻分：{metrics['rule_map']*100:.4f}，旧工具算法 TXT 复算：{metrics['ultralytics_map']*100:.4f}。"
        f"本地晋级门槛：{'通过，可试榜' if result['gate']['passed'] else '未通过，仅探索候选'}。\n\n"
        f"官方成绩待测；不是三划分最终模型。E9 官方最优仍为用户反馈的 55.5370，恢复包：`{RESTORE}`。"
        "上传前保留一次恢复机会；不根据本地分数推算官方得分。\n")
    write_json(OUT / "pipeline-status.json", {"stage": "candidate_ready", "updated_utc": now(),
               "delivery": str(OUT / "delivery.json"), "screen_gate_passed": result["gate"]["passed"]})


def rfdetr_route():
    config = yaml.safe_load((ROOT / "configs/rf48-mm.yaml").read_text())
    selected = OUT / "selected-config.json"
    write_json(selected, config)
    if not (OUT / "probe.json").exists():
        for batch, size in ((2, 768), (1, 768), (1, 576)):
            config.update(batch=batch, accumulation=16//batch, imgsz=size)
            write_json(selected, config)
            try:
                command(f"probe-b{batch}-s{size}", "scripts.probe_rfdetr", "--config", selected,
                        "--out", OUT / "probe.json", timeout=2*3600)
                break
            except subprocess.CalledProcessError:
                log = (OUT / f"probe-b{batch}-s{size}.log").read_text()
                if "out of memory" not in log.lower():
                    raise
        else:
            raise RuntimeError("All memory probes failed")
    probe = json.loads((OUT / "probe.json").read_text())
    assert probe["passed"] and probe["steps"] >= 100
    config = probe["config"]
    write_json(selected, config)
    rgb = config | {"multimodal": False, "max_hours": 5}
    rgb_path = OUT / "rgb-config.json"
    write_json(rgb_path, rgb)
    for label, cfg, stop in (("rgb-control", rgb_path, 10), ("mm-candidate", selected, 60)):
        run_dir = OUT / label
        args = ["--config", cfg, "--out", run_dir, "--split-manifest", SPLIT, "--stop-after", stop]
        if (run_dir / "last.pt").exists():
            args += ["--resume", run_dir / "last.pt"]
        command(label, "src.train_rfdetr", *args, timeout=22*3600 if label == "mm-candidate" else 6*3600)
    best = OUT / "mm-candidate/best.pt"
    choices = []
    for size in (768, 960):
        directory = OUT / f"mm-val-{size}"
        command(f"val-{size}", "src.predict_rfdetr", "--weights", best,
                "--sample-manifest", TRAIN, "--out-dir", directory, "--side", "val", "--imgsz", size)
        metrics = json.loads((directory / "metrics.json").read_text())
        sensitive = evaluate(TRAIN, directory, threshold=.30)
        write_json(directory / "conf-030-diagnostic.json", sensitive)
        choices.append((size, metrics))
    top = max(r["rule_map"] for _, r in choices)
    size, metrics = min((s, r) for s, r in choices if r["rule_map"] >= top - .002)
    write_json(OUT / "selected-inference.json", {"imgsz": size, "conf": .001, "nms": None, "metrics": metrics})
    for modality in ("ir", "depth"):
        command(f"zero-{modality}", "src.predict_rfdetr", "--weights", best,
                "--sample-manifest", TRAIN, "--out-dir", OUT / f"mm-zero-{modality}",
                "--side", "val", "--imgsz", size, "--zero-modality", modality)
    predictions = OUT / "phase1-mm-predictions"
    command("phase1-inference", "src.predict_rfdetr", "--weights", best,
            "--sample-manifest", TEST, "--out-dir", predictions, "--imgsz", size)
    package(predictions, f"rf48-mm-i{size}-c0p001", best, metrics)


def fallback_route(reason):
    write_json(OUT / "rfdetr-failure.json", {"reason": str(reason), "utc": now()})
    run_dir = OUT / "e9-fallback"
    command("e9-fallback", "src.yolo_raw", "--out", run_dir, "--weights", E9,
            "--split-manifest", SPLIT, timeout=5*3600)
    best = run_dir / "best-rule.pt"
    validation = OUT / "fallback-val"
    command("fallback-val", "src.predict_yolo_raw", "--weights", best,
            "--manifest", TRAIN, "--out", validation, "--side", "val")
    metrics = json.loads((validation / "metrics.json").read_text())
    baseline = json.loads((OUT / "baseline-summary.json").read_text())["e9-low"]
    if not screen_gate(metrics, baseline)["passed"]:
        write_json(OUT / "pipeline-status.json", {"stage": "no_trusted_upgrade_keep_e9", "updated_utc": now(),
                   "restore_zip": RESTORE, "restore_sha256": sha256(ROOT / RESTORE),
                   "fallback_metrics": metrics, "official_score_user_reported": 55.5370})
        return
    predictions = OUT / "phase1-fallback-predictions"
    command("fallback-phase1", "src.predict_yolo_raw", "--weights", best,
            "--manifest", TEST, "--out", predictions, "--side", "test")
    package(predictions, "rf48-e9-fallback-i1280-c0p001", best, metrics)


def main():
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (OUT / "execution-start.json").exists():
        write_json(OUT / "execution-start.json", {"implementation_started_utc": "2026-09-16T02:59:47+00:00",
                   "queue_started_utc": now(), "budget_hours": 48, "deadline_utc": "2026-09-18T02:59:47+00:00"})
    assert json.loads((OUT / "input-parity.json").read_text())["passed"]
    assert json.loads((OUT / "prediction-parity.json").read_text())["passed"]
    for path, digest in json.loads((OUT / "protected-assets.json").read_text()).items():
        assert sha256(ROOT / path) == digest, path
    assert shutil.disk_usage(ROOT).free >= 8 * 2**30
    try:
        rfdetr_route()
    except (subprocess.SubprocessError, RuntimeError) as exc:
        try:
            fallback_route(exc)
        except Exception as fallback_exc:
            write_json(OUT / "pipeline-status.json", {"stage": "stopped_preserve_e9", "utc": now(),
                       "rfdetr_error": str(exc), "fallback_error": str(fallback_exc), "restore_zip": RESTORE})
            raise


if __name__ == "__main__":
    main()
