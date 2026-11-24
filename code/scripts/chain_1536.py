"""接力链：等 1280 训练与收尾全部结束后，自动启动 1536 画布实验。

顺序（单卡，不能并发）：
  1. 等 `run1280` 训练结束（跑满 100 轮，或进程停止且轮数 5 分钟不再增长）；
  2. 等收尾作业完成 —— 以 `FINISHED.md` 出现为信号（它需要 GPU 做验证集与测试集推理）；
  3. 启动 `scripts.run_experiment --config configs/rf48d-1536.yaml`。

边界：不删除任何东西、不上传、不改动既有产物。

    screen -L -Logfile runs/formal/rf48b/chain.log -dmS aic-chain \
        .venv-rfdetr/bin/python -u -m scripts.chain_1536
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/formal/rf48b"
RUN = BASE / "run1280"
FINISHED = BASE / "FINISHED.md"
CONFIG = "configs/rf48d-1536.yaml"
OUT = BASE / "run1536"
TARGET_EPOCHS = 100
FINISH_WAIT_SECONDS = 40 * 60


def log(message):
    print(f"[chain {time.strftime('%H:%M:%S', time.gmtime())}Z] {message}", flush=True)


def epochs_done():
    history = RUN / "history.jsonl"
    if not history.exists():
        return 0
    return sum(1 for line in history.read_text().splitlines() if line.strip())


def training_alive():
    result = subprocess.run(["bash", "-lc",
                             "ps -eo args | grep -E 'train_rfde[t]r|run_rf4[8]b' | grep -v grep"],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def wait_for_training():
    last_lines, stable_since = -1, None
    while True:
        lines, alive = epochs_done(), training_alive()
        if lines >= TARGET_EPOCHS:
            log(f"1280 训练完成 {lines}/{TARGET_EPOCHS} 轮")
            return lines
        if lines != last_lines:
            log(f"1280 进度 {lines}/{TARGET_EPOCHS} 轮")
            last_lines, stable_since = lines, time.time()
        elif not alive and stable_since and time.time() - stable_since > 300:
            log(f"1280 训练进程已停止且 {lines} 轮未增长，按结束处理")
            return lines
        time.sleep(60)


def main():
    log("接力链启动：等待 1280 训练结束")
    lines = wait_for_training()

    deadline = time.time() + FINISH_WAIT_SECONDS
    while time.time() < deadline and not FINISHED.exists():
        time.sleep(30)
    if FINISHED.exists():
        log(f"收尾作业已完成（FINISHED.md 已生成），1280 共 {lines} 轮")
    else:
        log(f"等待收尾超时（{FINISH_WAIT_SECONDS // 60} 分钟），仍继续启动 1536")

    if OUT.exists() and (OUT / "history.jsonl").exists():
        done = sum(1 for line in (OUT / "history.jsonl").read_text().splitlines() if line.strip())
        log(f"注意：{OUT} 已存在且已完成 {done} 轮，run_experiment 会自动续跑")

    log(f"启动 1536 画布实验：{CONFIG} -> {OUT}")
    result = subprocess.run([sys.executable, "-u", "-m", "scripts.run_experiment",
                             "--config", CONFIG, "--out", str(OUT)], cwd=ROOT)
    log(f"1536 实验结束，退出码 {result.returncode}")


if __name__ == "__main__":
    main()
