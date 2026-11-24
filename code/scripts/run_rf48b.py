"""RF48b 队列：1280 画布、100 轮、按已标定口径选模；崩溃自动从 last.pt 续跑。

约束：不删除任何既有资产、不自动上传、不启动第二条 GPU 作业。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/formal/rf48b/run1280"
SPLIT = "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
TARGET_EPOCHS = 100


def epochs_done():
    history = OUT / "history.jsonl"
    if not history.exists():
        return 0
    return sum(1 for line in history.read_text().splitlines() if line.strip())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 13):
        done = epochs_done()
        if done >= TARGET_EPOCHS:
            print(f"[rf48b] 已完成 {done} 轮，队列结束", flush=True)
            break
        args = [sys.executable, "-u", "-m", "src.train_rfdetr",
                "--config", "configs/rf48b-1280.yaml", "--out", str(OUT),
                "--split-manifest", SPLIT]
        if (OUT / "last.pt").exists():
            args += ["--resume", str(OUT / "last.pt")]
        print(f"[rf48b] attempt {attempt} 已有 {done} 轮: {' '.join(args)}", flush=True)
        result = subprocess.run(args, cwd=ROOT)
        print(f"[rf48b] attempt {attempt} 退出码 {result.returncode}，"
              f"现有 {epochs_done()} 轮", flush=True)
        if result.returncode == 0 and epochs_done() >= TARGET_EPOCHS:
            break
        time.sleep(30)
    status = {}
    if (OUT / "status.json").exists():
        status = json.loads((OUT / "status.json").read_text())
    summary = {"epochs_done": epochs_done(), "status": status,
               "best_metrics": json.loads((OUT / "best-metrics.json").read_text())
               if (OUT / "best-metrics.json").exists() else None}
    print("[rf48b] DONE " + json.dumps(summary, ensure_ascii=False), flush=True)
    (ROOT / "runs/formal/rf48b/queue-result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
