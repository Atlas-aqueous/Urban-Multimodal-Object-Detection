"""训练 YOLOv8x-P2 5ch，完成搜索后无条件产出官方校准 ZIP。"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "datasets/formal-train-active"
DATA = PROJECT / "configs/formal-train-active-unweighted.yaml"
PRETRAIN = PROJECT / "runs/formal/pretrain/yolov8x-p2-5ch.pt"
REFERENCE_ZIP = PROJECT / "runs/formal/submissions/phase1-e9-l-p2-i1280-c0p4-n0p4.zip"
REFERENCE_LOCAL = 0.5458204402220287
REFERENCE_OFFICIAL = 0.55537
STATE = PROJECT / "runs/formal/e12-continuation-state.json"
SEARCH_OUT = PROJECT / "runs/formal/e12-inference-grid/results.json"


def write_state(stage: str, **extra) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **extra,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(STATE)
    print(f"[e12] {stage}: {extra}", flush=True)


def run(command: list[str]) -> None:
    print("[e12] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def validate_inputs() -> None:
    generated = ACTIVE / "data.yaml"
    if not generated.is_file():
        raise RuntimeError("活动训练集缺少 data.yaml")
    cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    train = sum(1 for _ in (ACTIVE / "images/train").glob("*.tiff"))
    val = sum(1 for _ in (ACTIVE / "images/val").glob("*.tiff"))
    if cfg.get("channels") != 5 or train != 1805 or val != 195:
        raise RuntimeError(
            f"活动训练集不完整: channels={cfg.get('channels')} "
            f"train={train} val={val}"
        )
    unweighted = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if unweighted.get("train") != "images/train" or unweighted.get("channels") != 5:
        raise RuntimeError("E12 非重复采样数据配置错误")
    for path in (PRETRAIN, REFERENCE_ZIP):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"缺少 E12 必要文件: {path}")
    free = shutil.disk_usage(PROJECT).free
    if free < 1536 * 1024**2:
        raise RuntimeError(f"E12 启动前可用磁盘不足 1.5 GiB: {free / 1024**3:.2f}")


def train_command(batch: int, name: str) -> list[str]:
    return [
        sys.executable, "src/train.py", "--data", str(DATA),
        "--model", str(PRETRAIN), "--imgsz", "1280", "--epochs", "100",
        "--batch", str(batch), "--patience", "20", "--workers", "8", "--seed", "42",
        "--optimizer", "AdamW", "--lr0", "0.001", "--lrf", "0.01", "--cos-lr",
        "--weight-decay", "0.0005", "--warmup-epochs", "3",
        "--box", "7.5", "--cls", "0.5", "--dfl", "1.5",
        "--close-mosaic", "10", "--mosaic", "1.0", "--mixup", "0.0",
        "--no-multi-scale", "--device", "0", "--modality-dropout", "0",
        "--eval-imgsz", "1280", "--eval-conf", "0.005", "--eval-iou", "0.7",
        "--project", "runs/formal", "--name", name,
    ]


def train_e12() -> Path:
    name = "e12-yolov8x-p2-1280-aggressive-b4"
    run_dir = PROJECT / "runs/formal" / name
    record = run_dir / "experiment.json"
    if record.is_file():
        return record
    try:
        run(train_command(4, name))
    except subprocess.CalledProcessError as first_error:
        fallback_name = "e12-yolov8x-p2-1280-aggressive-b2"
        fallback_dir = PROJECT / "runs/formal" / fallback_name
        write_state(
            "batch4_failed_retrying_batch2", returncode=first_error.returncode,
            failed_run=str(run_dir), fallback_run=str(fallback_dir),
        )
        fallback_record = fallback_dir / "experiment.json"
        if not fallback_record.is_file():
            run(train_command(2, fallback_name))
        record = fallback_record
    if not record.is_file():
        raise RuntimeError("E12 训练结束但未生成 experiment.json")
    return record


def package_name(best: dict) -> str:
    conf = f"{float(best['conf']):g}".replace(".", "p")
    iou = f"{float(best['iou']):g}".replace(".", "p")
    return f"phase1-e12-x-p2-i{int(best['imgsz'])}-c{conf}-n{iou}.zip"


def main() -> None:
    validate_inputs()
    write_state(
        "training_e12", batch=4, oom_fallback_batch=2,
        reference_local=REFERENCE_LOCAL, reference_official=REFERENCE_OFFICIAL,
    )
    record = train_e12()
    experiment = json.loads(record.read_text(encoding="utf-8"))
    weights = Path(experiment["selected_weights"])
    if not weights.is_file():
        raise RuntimeError(f"E12 最优权重不存在: {weights}")

    write_state("searching_inference", weights=str(weights))
    run([
        sys.executable, "src/search_inference.py", "--weights", str(weights),
        "--data", str(DATA), "--imgsz", "960,1152,1280",
        "--conf", "0.25,0.30,0.35,0.40,0.45,0.50,0.55",
        "--iou", "0.35,0.40,0.45,0.50,0.55,0.60",
        "--device", "0", "--half", "--project", "runs/formal/e12-inference-grid",
        "--out", str(SEARCH_OUT),
    ])
    best = json.loads(SEARCH_OUT.read_text(encoding="utf-8"))["best"]
    gain = float(best["map50_95"]) - REFERENCE_LOCAL
    output = PROJECT / "runs/formal/submissions" / package_name(best)
    pred_dir = PROJECT / "runs/formal" / output.stem.replace("phase1-", "phase1-predictions-")
    write_state(
        "building_submission", best=best, gain_over_e9_local=gain,
        output=str(output), packaging_reason="E9 reversed local leaderboard ranking",
    )
    run([
        sys.executable, "src/build_optimized_submission.py", "--weights", str(weights),
        "--old-zip", str(REFERENCE_ZIP), "--out-zip", str(output),
        "--pred-dir", str(pred_dir), "--imgsz", str(best["imgsz"]),
        "--conf", str(best["conf"]), "--iou", str(best["iou"]),
    ])
    write_state(
        "complete", best=best, gain_over_e9_local=gain, submission=str(output),
        requires_official_calibration=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_state("failed", error=f"{type(error).__name__}: {error}")
        raise
