"""通用实验队列：跑满 config 里指定的轮数，崩溃后自动从 last.pt 续跑。

    python -m scripts.run_experiment --config configs/rf48c-early1280.yaml \
        --out runs/formal/rf48b/run-early1280

约束：不删除任何既有资产、不自动上传、一次只占一张卡。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"


def epochs_done(out_dir):
    history = Path(out_dir) / "history.jsonl"
    if not history.exists():
        return 0
    return sum(1 for line in history.read_text().splitlines() if line.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-manifest", default=DEFAULT_SPLIT)
    ap.add_argument("--tries", type=int, default=12)
    args = ap.parse_args()

    from src.trimodal_data import load_config
    config = load_config(args.config)
    target = int(config["epochs"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.tries + 1):
        done = epochs_done(out_dir)
        if done >= target:
            print(f"[queue] 已完成 {done}/{target} 轮，结束", flush=True)
            break
        command = [sys.executable, "-u", "-m", "src.train_rfdetr",
                   "--config", args.config, "--out", str(out_dir),
                   "--split-manifest", args.split_manifest]
        if (out_dir / "last.pt").exists():
            command += ["--resume", str(out_dir / "last.pt")]
        print(f"[queue] attempt {attempt} 已有 {done}/{target} 轮", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        print(f"[queue] attempt {attempt} 退出码 {result.returncode}，现有 {epochs_done(out_dir)} 轮",
              flush=True)
        if result.returncode == 0 and epochs_done(out_dir) >= target:
            break
        time.sleep(30)

    summary = {"config": args.config, "out": str(out_dir), "target_epochs": target,
               "epochs_done": epochs_done(out_dir)}
    for name in ("status.json", "best-metrics.json"):
        path = out_dir / name
        if path.exists():
            summary[name] = json.loads(path.read_text())
    result_path = out_dir / "queue-result.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("[queue] DONE " + json.dumps(summary, ensure_ascii=False)[:600], flush=True)


if __name__ == "__main__":
    main()
