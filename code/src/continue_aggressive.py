"""等待活动训练集恢复，执行高分辨率 E0 搜索并强制运行 E9 大模型。"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "datasets/formal-train-active"
DATA = PROJECT / "configs/formal-train-active-unweighted.yaml"
E0_WEIGHTS = PROJECT / "runs/formal/yolov8m-5ch-revised-960/weights/best.pt"
E9_PRETRAIN = PROJECT / "runs/formal/pretrain/yolov8l-p2-5ch.pt"
HIGHRES_RESULT = PROJECT / "runs/formal/e0-highres-grid/results.json"
STATE = PROJECT / "runs/formal/aggressive-continuation-state.json"


def write_state(stage: str, **extra) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **extra,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(STATE)
    print(f"[aggressive] {stage}: {extra}", flush=True)


def pid_matches(pid: int, marker: str) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text().split()[2]
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    if state == "Z":
        return False
    if marker not in command:
        raise RuntimeError(f"PID {pid} 已被其他进程复用: {command}")
    return True


def run(command: list[str]) -> None:
    print("[aggressive] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def validate_active() -> None:
    generated = ACTIVE / "data.yaml"
    if not generated.is_file():
        raise RuntimeError("活动数据集缺少 data.yaml")
    cfg = yaml.safe_load(generated.read_text(encoding="utf-8"))
    tiffs = sum(1 for _ in (ACTIVE / "images").rglob("*.tiff"))
    train = sum(1 for _ in (ACTIVE / "images/train").glob("*.tiff"))
    val = sum(1 for _ in (ACTIVE / "images/val").glob("*.tiff"))
    if cfg.get("channels") != 5 or tiffs != 2000 or train != 1805 or val != 195:
        raise RuntimeError(
            f"活动数据集不完整: channels={cfg.get('channels')} "
            f"tiffs={tiffs} train={train} val={val}"
        )
    unweighted = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if unweighted.get("train") != "images/train" or unweighted.get("channels") != 5:
        raise RuntimeError("E9 未重复采样数据配置错误")


def run_highres() -> dict:
    run([
        sys.executable, "src/search_inference.py", "--weights", str(E0_WEIGHTS),
        "--data", str(DATA), "--imgsz", "960,1152,1280,1408,1536",
        "--conf", "0.35,0.40,0.45", "--iou", "0.50,0.55,0.60",
        "--device", "0", "--half", "--project", "runs/formal/e0-highres-grid",
        "--out", str(HIGHRES_RESULT),
    ])
    payload = json.loads(HIGHRES_RESULT.read_text(encoding="utf-8"))
    return payload["best"]


def e9_command(batch: int, name: str) -> list[str]:
    return [
        sys.executable, "src/train.py", "--data", str(DATA),
        "--model", str(E9_PRETRAIN), "--imgsz", "1280", "--epochs", "100",
        "--batch", str(batch), "--patience", "20", "--workers", "8", "--seed", "42",
        "--optimizer", "AdamW", "--lr0", "0.001", "--lrf", "0.01", "--cos-lr",
        "--weight-decay", "0.0005", "--warmup-epochs", "3",
        "--box", "7.5", "--cls", "0.5", "--dfl", "1.5",
        "--close-mosaic", "10", "--mosaic", "1.0", "--mixup", "0.0",
        "--no-multi-scale", "--device", "0", "--modality-dropout", "0",
        "--eval-imgsz", "1280", "--eval-conf", "0.005", "--eval-iou", "0.7",
        "--project", "runs/formal", "--name", name,
    ]


def run_e9() -> Path:
    name = "e9-yolov8l-p2-1280-aggressive-b4"
    run_dir = PROJECT / "runs/formal" / name
    if (run_dir / "experiment.json").is_file():
        write_state("e9_already_complete", experiment=str(run_dir / "experiment.json"))
        return run_dir / "experiment.json"
    try:
        run(e9_command(4, name))
    except subprocess.CalledProcessError as first_error:
        # batch 4 的现场保留在独立目录；batch 2 不会覆盖它。
        fallback_name = "e9-yolov8l-p2-1280-aggressive-b2"
        fallback_dir = PROJECT / "runs/formal" / fallback_name
        write_state("e9_batch4_failed_retrying_batch2", returncode=first_error.returncode,
                    failed_run=str(run_dir), fallback_run=str(fallback_dir))
        run(e9_command(2, fallback_name))
        run_dir = fallback_dir
    record = run_dir / "experiment.json"
    if not record.is_file():
        raise RuntimeError("E9 结束但未生成 experiment.json")
    return record


def main(args: argparse.Namespace) -> None:
    while pid_matches(args.wait_pid, "build_optimized_submission.py"):
        write_state("waiting_for_restore", pid=args.wait_pid)
        time.sleep(args.poll_seconds)
    validate_active()
    for path in (E0_WEIGHTS, E9_PRETRAIN):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"缺少必要权重: {path}")
    write_state("searching_e0_highres")
    best = run_highres()
    write_state("e0_highres_complete", best=best)
    write_state("training_e9", batch=4, data=str(DATA))
    e9_record = run_e9()
    write_state("complete", e0_highres_best=best,
                e9_experiment=str(e9_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write_state("failed", error=f"{type(error).__name__}: {error}")
        raise
