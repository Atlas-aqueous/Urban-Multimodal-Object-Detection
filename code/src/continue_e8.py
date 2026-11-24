"""E7 完成后受保护地执行 E8 六通道实验，并恢复五通道派生数据。

该控制器只删除并重建 ``datasets/formal-train-active`` 下的派生文件；官方
ZIP、解压后的原始模态、已有权重和提交包均只读。任一前置校验失败即停止。
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from src.experiment_gate import official_metrics, screen_gate
except ModuleNotFoundError:
    from experiment_gate import official_metrics, screen_gate


PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "datasets/formal-train-active"
RAW = PROJECT / "formal_data/train_revised_labels"
SPLIT = PROJECT / "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
BASELINE = PROJECT / "runs/formal/inference-grid/selection.json"
E7_RECORD = PROJECT / "runs/formal/e7-p2-1280-rare/experiment.json"
STATE = PROJECT / "runs/formal/e8-continuation-state.json"
GUARD = PROJECT / "runs/formal/artifact-guard-before-e8.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def update_state(stage: str, **extra) -> None:
    payload = {"updated_at": utc_now(), "stage": stage, **extra}
    write_json(STATE, payload)
    print(f"[continue-e8] {stage}: {extra}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip(path: Path, expected_txt: int | None = None) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = [name for name in archive.namelist() if not name.endswith("/")]
    if bad is not None:
        raise RuntimeError(f"ZIP CRC 校验失败: {path} -> {bad}")
    if expected_txt is not None:
        txt_count = sum(Path(name).suffix.lower() == ".txt" for name in names)
        if txt_count != expected_txt:
            raise RuntimeError(f"提交 ZIP TXT 数量错误: {txt_count} != {expected_txt}")
    return {"files": len(names), "zip_crc": "ok"}


def safeguard_artifacts() -> None:
    data_root = PROJECT.parents[1] / "基线阶段数据集-面向城市场景的多模态目标检测"
    required = {
        "official_train_zip": data_root / "train/train.zip",
        "official_test_zip": data_root / "test/test.zip",
        "baseline_weights": PROJECT / "runs/formal/yolov8m-5ch-revised-960/weights/best.pt",
        "e7_weights": PROJECT / "runs/formal/e7-p2-1280-rare/weights/best.pt",
        "e8_pretrain": PROJECT / "runs/formal/pretrain/yolov8m-p2-6ch.pt",
        "baseline_submission": PROJECT / "runs/formal/submissions/baseline-preserved-before-swap.zip",
    }
    records = {}
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"关键产物缺失: {name} -> {path}")
        record = {"path": str(path.resolve()), "size": path.stat().st_size,
                  "sha256": sha256(path)}
        if name.startswith("official_"):
            record.update(validate_zip(path))
        elif name == "baseline_submission":
            record.update(validate_zip(path, expected_txt=1000))
        records[name] = record
    write_json(GUARD, {"created_at": utc_now(), "artifacts": records})


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


def wait_for_e7(pid: int, poll_seconds: int) -> None:
    while pid_matches(pid, "e7-p2-1280-rare"):
        update_state("waiting_for_e7", pid=pid)
        time.sleep(poll_seconds)
    if not E7_RECORD.is_file():
        raise RuntimeError("E7 进程已结束但未生成 experiment.json，停止 E8")


def run(command: list[str]) -> None:
    print("[continue-e8] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def prepare(mode: str) -> None:
    run([
        sys.executable, "src/prepare_data.py", "--data-root", str(RAW),
        "--out", str(ACTIVE), "--mode", mode, "--split-manifest", str(SPLIT),
        "--rare-repeat", "--overwrite",
    ])
    cfg = yaml.safe_load((ACTIVE / "data.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ACTIVE / "split.json").read_text(encoding="utf-8"))
    expected = int(mode.removesuffix("ch"))
    if cfg.get("channels") != expected or cfg.get("train") != "train_weighted.txt":
        raise RuntimeError(f"重建后的数据配置不符合 {mode}: {cfg}")
    if len(manifest["splits"]["train"]) != 1805 or len(manifest["splits"]["val"]) != 195:
        raise RuntimeError("重建后的训练/验证数量改变")


def run_report() -> None:
    run([sys.executable, "src/report_experiments.py", "--baseline", str(BASELINE)])


def run_e8() -> Path:
    name = "e8-p2-1280-6ch-rare"
    run([
        sys.executable, "src/train.py", "--data", str(ACTIVE / "data.yaml"),
        "--model", "runs/formal/pretrain/yolov8m-p2-6ch.pt",
        "--imgsz", "1280", "--epochs", "100", "--batch", "8",
        "--patience", "20", "--workers", "8", "--seed", "42",
        "--optimizer", "AdamW", "--lr0", "0.001", "--lrf", "0.01",
        "--cos-lr", "--weight-decay", "0.0005", "--warmup-epochs", "3",
        "--box", "7.5", "--cls", "0.5", "--dfl", "1.5",
        "--close-mosaic", "10", "--mosaic", "1.0", "--mixup", "0.0",
        "--no-multi-scale", "--device", "0", "--modality-dropout", "0",
        "--eval-imgsz", "1280", "--eval-conf", "0.005", "--eval-iou", "0.7",
        "--project", "runs/formal", "--name", name,
    ])
    record = PROJECT / "runs/formal" / name / "experiment.json"
    if not record.is_file():
        raise RuntimeError("E8 训练结束但缺少 experiment.json")
    return record


def main(args: argparse.Namespace) -> None:
    lock_path = PROJECT / "runs/formal/e8-continuation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("已有 E8 续跑控制器正在运行") from exc

    wait_for_e7(args.wait_pid, args.poll_seconds)
    e7_gate = screen_gate(official_metrics(BASELINE), official_metrics(E7_RECORD))
    update_state("e7_complete", gate=e7_gate)
    run_report()
    safeguard_artifacts()
    update_state("artifacts_verified", guard=str(GUARD))

    active_bytes = sum(p.stat().st_size for p in (ACTIVE / "images").rglob("*") if p.is_file())
    free_bytes = shutil.disk_usage(ACTIVE).free
    # 六通道有效性掩码高度可压缩；实测约为五通道体积的 1.01 倍。仍保留 1 GiB 余量。
    if free_bytes + active_bytes < int(active_bytes * 1.05) + 1024 ** 3:
        raise RuntimeError("删除可重建五通道数据后仍不足以安全生成六通道数据")

    update_state("building_6ch", recoverable_5ch_bytes=active_bytes)
    e8_gate = None
    try:
        prepare("6ch")
        update_state("training_e8")
        e8_record = run_e8()
        e8_gate = screen_gate(official_metrics(BASELINE), official_metrics(e8_record))
        update_state("e8_complete", gate=e8_gate)
        run_report()
    finally:
        # 即使六通道构建或训练失败，也尽力恢复进入流程前的五通道派生集。
        update_state("restoring_5ch")
        prepare("5ch")
    update_state("complete", e7_gate=e7_gate, e8_gate=e8_gate,
                 report=str(PROJECT / "runs/formal/experiment-summary.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parsed = parser.parse_args()
    if not 5 <= parsed.poll_seconds <= 60:
        parser.error("--poll-seconds 必须在 5–60 秒")
    try:
        main(parsed)
    except Exception as error:
        update_state("failed", error=f"{type(error).__name__}: {error}")
        raise
