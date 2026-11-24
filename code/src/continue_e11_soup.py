"""E10 结束后搜索 E0/E1 权重插值，仅超过当前最优时打包。"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "configs/formal-train-active-unweighted.yaml"
LEFT = PROJECT / "runs/formal/yolov8m-5ch-revised-960/weights/best.pt"
RIGHT = PROJECT / "runs/formal/e1-dropout-000-p3-960/weights/best.pt"
STATE = PROJECT / "runs/formal/e11-soup-state.json"
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
    print(f"[e11-soup] {stage}: {extra}", flush=True)


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
    print("[e11-soup] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def retained_candidate() -> tuple[float, Path, list[dict]]:
    score = BASELINE_SCORE
    archive = BASELINE_ZIP
    states = []
    for state_path in (
        PROJECT / "runs/formal/after-e9-state.json",
        PROJECT / "runs/formal/e10-continuation-state.json",
    ):
        if not state_path.is_file():
            continue
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        states.append(payload)
        best = payload.get("best")
        candidate = payload.get("submission")
        if not isinstance(best, dict) or not candidate:
            continue
        path = Path(candidate)
        candidate_score = float(best.get("map50_95", -1))
        if candidate_score > score and path.is_file():
            score, archive = candidate_score, path
    return score, archive, states


def soup_path(alpha: float) -> Path:
    code = f"{alpha:.2f}".replace(".", "")
    return PROJECT / f"runs/formal/weight-soups/e0-e1-a{code}.pt"


def make_soup(alpha: float) -> Path:
    out = soup_path(alpha)
    if not out.is_file():
        run([
            sys.executable, "src/average_checkpoints.py",
            "--left", str(LEFT), "--right", str(RIGHT),
            "--alpha", str(alpha), "--out", str(out),
        ])
    return out


def probe(alpha: float, weights: Path) -> dict:
    code = f"{alpha:.2f}".replace(".", "")
    result = PROJECT / f"runs/formal/e11-soup-probe/a{code}/results.json"
    run([
        sys.executable, "src/search_inference.py", "--weights", str(weights),
        "--data", str(DATA), "--imgsz", "960",
        "--conf", "0.25,0.35,0.40,0.45,0.50",
        "--iou", "0.40,0.50,0.55,0.60,0.70",
        "--device", "0", "--half",
        "--project", f"runs/formal/e11-soup-probe/a{code}",
        "--out", str(result),
    ])
    best = json.loads(result.read_text(encoding="utf-8"))["best"]
    return {"alpha_right": alpha, "weights": str(weights), **best}


def submission_name(best: dict) -> str:
    conf = f"{float(best['conf']):g}".replace(".", "p")
    iou = f"{float(best['iou']):g}".replace(".", "p")
    alpha = f"{float(best['alpha_right']):g}".replace(".", "p")
    return f"phase1-e11-soup-a{alpha}-i{int(best['imgsz'])}-c{conf}-n{iou}.zip"


def main(args: argparse.Namespace) -> None:
    while pid_matches(args.wait_pid, "continue_e10.py"):
        write_state("waiting_for_e10", pid=args.wait_pid)
        time.sleep(args.poll_seconds)
    for path in (LEFT, RIGHT, DATA, BASELINE_ZIP):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"缺少 E11 必要文件: {path}")

    reference_score, reference_zip, upstream = retained_candidate()
    write_state(
        "probing_weight_interpolation", reference_score=reference_score,
        reference_submission=str(reference_zip), upstream_stages=[x.get("stage") for x in upstream],
    )
    probes = [probe(alpha, make_soup(alpha)) for alpha in (0.25, 0.50, 0.75)]
    best_probe = max(probes, key=lambda row: float(row["map50_95"]))
    write_state("probe_complete", probes=probes, best_probe=best_probe)
    if float(best_probe["map50_95"]) <= BASELINE_SCORE:
        write_state(
            "complete_no_full_search", best=best_probe,
            gain_over_baseline=float(best_probe["map50_95"]) - BASELINE_SCORE,
            retained_submission=str(reference_zip),
        )
        return

    full_result = PROJECT / "runs/formal/e11-soup-grid/results.json"
    write_state("searching_best_soup", best_probe=best_probe)
    run([
        sys.executable, "src/search_inference.py", "--weights", best_probe["weights"],
        "--data", str(DATA), "--imgsz", "960,1152",
        "--conf", "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55",
        "--iou", "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        "--device", "0", "--half", "--project", "runs/formal/e11-soup-grid",
        "--out", str(full_result),
    ])
    best = json.loads(full_result.read_text(encoding="utf-8"))["best"]
    best["alpha_right"] = best_probe["alpha_right"]
    best["weights"] = best_probe["weights"]
    gain = float(best["map50_95"]) - reference_score
    write_state("full_search_complete", best=best, gain_over_retained=gain)
    if gain <= 0:
        write_state(
            "complete_no_submission", best=best, gain_over_retained=gain,
            retained_submission=str(reference_zip),
        )
        return

    output = PROJECT / "runs/formal/submissions" / submission_name(best)
    pred_dir = PROJECT / "runs/formal" / output.stem.replace("phase1-", "phase1-predictions-")
    write_state("building_soup_submission", best=best, output=str(output))
    run([
        sys.executable, "src/build_optimized_submission.py",
        "--weights", best["weights"], "--old-zip", str(reference_zip),
        "--out-zip", str(output), "--pred-dir", str(pred_dir),
        "--imgsz", str(best["imgsz"]), "--conf", str(best["conf"]),
        "--iou", str(best["iou"]),
    ])
    write_state("complete", best=best, gain_over_retained=gain, submission=str(output))


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
