"""E9 结束后自动调优推理参数，仅在超过当前最优时生成新提交包。"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
AGGRESSIVE_STATE = PROJECT / "runs/formal/aggressive-continuation-state.json"
STATE = PROJECT / "runs/formal/after-e9-state.json"
DATA = PROJECT / "configs/formal-train-active-unweighted.yaml"
SEARCH_OUT = PROJECT / "runs/formal/e9-inference-grid/results.json"
CURRENT_BEST = 0.5528247602465499
CURRENT_ZIP = PROJECT / "runs/formal/submissions/phase1-e0-p3-960-c040-i055.zip"


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
    print(f"[after-e9] {stage}: {extra}", flush=True)


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
        raise RuntimeError(f"PID {pid} 已被复用: {command}")
    return True


def run(command: list[str]) -> None:
    print("[after-e9] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def submission_name(best: dict) -> str:
    conf = f"{float(best['conf']):g}".replace(".", "p")
    iou = f"{float(best['iou']):g}".replace(".", "p")
    return f"phase1-e9-l-p2-i{int(best['imgsz'])}-c{conf}-n{iou}.zip"


def main(args: argparse.Namespace) -> None:
    while pid_matches(args.wait_pid, "continue_aggressive.py"):
        write_state("waiting_for_e9_controller", pid=args.wait_pid)
        time.sleep(args.poll_seconds)
    if not AGGRESSIVE_STATE.is_file():
        raise RuntimeError("缺少激进流水线状态")
    upstream = json.loads(AGGRESSIVE_STATE.read_text(encoding="utf-8"))
    if upstream.get("stage") != "complete":
        raise RuntimeError(f"E9 上游未正常完成: {upstream}")
    record = Path(upstream["e9_experiment"])
    if not record.is_file():
        raise RuntimeError(f"E9 experiment.json 不存在: {record}")
    experiment = json.loads(record.read_text(encoding="utf-8"))
    weights = Path(experiment["selected_weights"])
    if not weights.is_file():
        raise RuntimeError(f"E9 最优权重不存在: {weights}")

    write_state("searching_e9_inference", weights=str(weights))
    run([
        sys.executable, "src/search_inference.py", "--weights", str(weights),
        "--data", str(DATA), "--imgsz", "960,1152,1280,1408",
        "--conf", "0.05,0.10,0.20,0.30,0.40,0.50",
        "--iou", "0.40,0.55,0.70", "--device", "0", "--half",
        "--project", "runs/formal/e9-inference-grid", "--out", str(SEARCH_OUT),
    ])
    best = json.loads(SEARCH_OUT.read_text(encoding="utf-8"))["best"]
    gain = float(best["map50_95"]) - CURRENT_BEST
    write_state("e9_inference_complete", best=best, gain_over_current=gain)
    run([sys.executable, "src/report_experiments.py",
         "--baseline", "runs/formal/inference-grid/selection.json"])
    if gain <= 0:
        write_state("complete_no_submission", best=best, gain_over_current=gain,
                    retained_submission=str(CURRENT_ZIP))
        return

    output = PROJECT / "runs/formal/submissions" / submission_name(best)
    pred_dir = PROJECT / "runs/formal" / output.stem.replace("phase1-", "phase1-predictions-")
    write_state("building_e9_submission", best=best, output=str(output))
    run([
        sys.executable, "src/build_optimized_submission.py", "--weights", str(weights),
        "--old-zip", str(CURRENT_ZIP), "--out-zip", str(output),
        "--pred-dir", str(pred_dir), "--imgsz", str(best["imgsz"]),
        "--conf", str(best["conf"]), "--iou", str(best["iou"]),
    ])
    write_state("complete", best=best, gain_over_current=gain,
                submission=str(output))


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
