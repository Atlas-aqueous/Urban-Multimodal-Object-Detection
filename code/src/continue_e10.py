"""E9 训练/调优结束后，串行训练并评估 YOLOv8l-P3 5ch/960。"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "datasets/formal-train-active"
DATA = PROJECT / "configs/formal-train-active-unweighted.yaml"
PRETRAIN = PROJECT / "runs/formal/pretrain/yolov8l-5ch.pt"
UPSTREAM_STATE = PROJECT / "runs/formal/after-e9-state.json"
STATE = PROJECT / "runs/formal/e10-continuation-state.json"
SEARCH_OUT = PROJECT / "runs/formal/e10-inference-grid/results.json"
BASELINE_SCORE = 0.5528247602465499
BASELINE_ZIP = PROJECT / "runs/formal/submissions/phase1-e0-p3-960-c040-i055.zip"


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
    print(f"[e10] {stage}: {extra}", flush=True)


def pid_matches(pid: int, marker: str) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text().split()[2]
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except FileNotFoundError:
        return False
    if state == "Z":
        return False
    if marker not in command:
        raise RuntimeError(f"PID {pid} 已被复用: {command}")
    return True


def run(command: list[str]) -> None:
    print("[e10] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def validate_inputs() -> None:
    generated = ACTIVE / "data.yaml"
    if not generated.is_file():
        raise RuntimeError("活动训练集缺少 data.yaml")
    cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    tiffs = sum(1 for _ in (ACTIVE / "images").rglob("*.tiff"))
    train = sum(1 for _ in (ACTIVE / "images/train").glob("*.tiff"))
    val = sum(1 for _ in (ACTIVE / "images/val").glob("*.tiff"))
    if cfg.get("channels") != 5 or tiffs != 2000 or train != 1805 or val != 195:
        raise RuntimeError(
            f"活动训练集不完整: channels={cfg.get('channels')} "
            f"tiffs={tiffs} train={train} val={val}"
        )
    unweighted = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if unweighted.get("train") != "images/train" or unweighted.get("channels") != 5:
        raise RuntimeError("E10 非重复采样数据配置错误")
    for path in (PRETRAIN, BASELINE_ZIP):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"缺少必要文件: {path}")
    free = shutil.disk_usage(PROJECT).free
    if free < 1024**3:
        raise RuntimeError(f"可用磁盘不足 1 GiB: {free / 1024**3:.2f} GiB")


def train_command(batch: int, name: str, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable, "src/train.py", "--data", str(DATA),
        "--model", str(resume or PRETRAIN), "--imgsz", "960", "--epochs", "100",
        "--batch", str(batch), "--patience", "20", "--workers", "8", "--seed", "42",
        "--optimizer", "AdamW", "--lr0", "0.001", "--lrf", "0.01", "--cos-lr",
        "--weight-decay", "0.0005", "--warmup-epochs", "3",
        "--box", "7.5", "--cls", "0.5", "--dfl", "1.5",
        "--close-mosaic", "10", "--mosaic", "1.0", "--mixup", "0.0",
        "--no-multi-scale", "--device", "0", "--modality-dropout", "0",
        "--eval-imgsz", "960", "--eval-conf", "0.005", "--eval-iou", "0.7",
        "--project", "runs/formal", "--name", name,
    ]
    if resume is not None:
        command.append("--resume")
    return command


def train_e10() -> Path:
    name = "e10-yolov8l-p3-960-aggressive-b8"
    run_dir = PROJECT / "runs/formal" / name
    record = run_dir / "experiment.json"
    if record.is_file():
        write_state("e10_already_complete", experiment=str(record))
        return record
    last = run_dir / "weights/last.pt"
    if last.is_file():
        write_state("resuming_e10_batch8", checkpoint=str(last))
        run(train_command(8, name, last))
    else:
        try:
            run(train_command(8, name))
        except subprocess.CalledProcessError as first_error:
            fallback_name = "e10-yolov8l-p3-960-aggressive-b4"
            fallback_dir = PROJECT / "runs/formal" / fallback_name
            write_state(
                "e10_batch8_failed_retrying_batch4",
                returncode=first_error.returncode,
                failed_run=str(run_dir),
                fallback_run=str(fallback_dir),
            )
            fallback_record = fallback_dir / "experiment.json"
            fallback_last = fallback_dir / "weights/last.pt"
            if fallback_record.is_file():
                return fallback_record
            run(train_command(4, fallback_name, fallback_last if fallback_last.is_file() else None))
            record = fallback_record
    if not record.is_file():
        raise RuntimeError("E10 结束但未生成 experiment.json")
    return record


def retained_candidate() -> tuple[float, Path, dict]:
    score = BASELINE_SCORE
    archive = BASELINE_ZIP
    upstream = {}
    if UPSTREAM_STATE.is_file():
        upstream = json.loads(UPSTREAM_STATE.read_text(encoding="utf-8"))
        best = upstream.get("best")
        candidate = upstream.get("submission")
        if (
            isinstance(best, dict)
            and float(best.get("map50_95", -1)) > score
            and candidate
            and Path(candidate).is_file()
        ):
            score = float(best["map50_95"])
            archive = Path(candidate)
    return score, archive, upstream


def submission_name(best: dict) -> str:
    conf = f"{float(best['conf']):g}".replace(".", "p")
    iou = f"{float(best['iou']):g}".replace(".", "p")
    return f"phase1-e10-l-p3-i{int(best['imgsz'])}-c{conf}-n{iou}.zip"


def main(args: argparse.Namespace) -> None:
    while pid_matches(args.wait_pid, "continue_after_e9.py"):
        write_state("waiting_for_e9_evaluation", pid=args.wait_pid)
        time.sleep(args.poll_seconds)
    validate_inputs()
    reference_score, reference_zip, upstream = retained_candidate()
    write_state(
        "training_e10", batch=8, data=str(DATA),
        reference_score=reference_score, reference_submission=str(reference_zip),
        upstream_stage=upstream.get("stage"),
    )
    record = train_e10()
    experiment = json.loads(record.read_text(encoding="utf-8"))
    weights = Path(experiment["selected_weights"])
    if not weights.is_file():
        raise RuntimeError(f"E10 最优权重不存在: {weights}")

    write_state("searching_e10_inference", weights=str(weights))
    run([
        sys.executable, "src/search_inference.py", "--weights", str(weights),
        "--data", str(DATA), "--imgsz", "960,1152",
        "--conf", "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55",
        "--iou", "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        "--device", "0", "--half", "--project", "runs/formal/e10-inference-grid",
        "--out", str(SEARCH_OUT),
    ])
    best = json.loads(SEARCH_OUT.read_text(encoding="utf-8"))["best"]
    gain = float(best["map50_95"]) - reference_score
    write_state(
        "e10_inference_complete", best=best, gain_over_retained=gain,
        retained_submission=str(reference_zip),
    )
    run([sys.executable, "src/report_experiments.py",
         "--baseline", "runs/formal/inference-grid/selection.json"])
    if gain <= 0:
        write_state(
            "complete_no_submission", best=best, gain_over_retained=gain,
            retained_submission=str(reference_zip),
        )
        return

    output = PROJECT / "runs/formal/submissions" / submission_name(best)
    pred_dir = PROJECT / "runs/formal" / output.stem.replace("phase1-", "phase1-predictions-")
    write_state("building_e10_submission", best=best, output=str(output))
    run([
        sys.executable, "src/build_optimized_submission.py", "--weights", str(weights),
        "--old-zip", str(reference_zip), "--out-zip", str(output),
        "--pred-dir", str(pred_dir), "--imgsz", str(best["imgsz"]),
        "--conf", str(best["conf"]), "--iou", str(best["iou"]),
    ])
    write_state(
        "complete", best=best, gain_over_retained=gain, submission=str(output)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parsed = parser.parse_args()
    try:
        main(parsed)
    except Exception as error:
        write_state("failed", error=f"{type(error).__name__}: {error}")
        raise
