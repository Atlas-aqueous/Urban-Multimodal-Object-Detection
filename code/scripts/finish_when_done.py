"""守候 A 训练结束，自动执行收尾流水线与配对判定，并写出结论摘要。

边界（刻意保守）：
- **不训练**、**不上传**、不删除任何东西；
- 只调用已验证过的 scripts/finalize_rf48b.py 与 scripts/paired_bootstrap.py；
- 第二个实验**不自动启动**（选择依赖对 A 结果的判断，必须留人在环）。

    screen -L -Logfile runs/formal/rf48b/finish.log -dmS aic-finish \
        .venv-rfdetr/bin/python -u -m scripts.finish_when_done
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/formal/rf48b"
RUN = BASE / "run1280"
CURRENT_BEST = ROOT / "runs/formal/rf48/mm-val-768"
CURRENT_BEST_CONF, CURRENT_BEST_NMS = 0.5, 0.6
CURRENT_BEST_LOCAL = 61.7232   # ep65 包本地分（官方 60.7360）
CURRENT_OFFICIAL = 60.7360   # ep65 包官方实测（2026-09-19 更新）


def log(message):
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[finish {stamp}Z] {message}", flush=True)


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


def wait_for_training(target_epochs, max_hours):
    deadline = time.time() + max_hours * 3600
    last_lines, stable_since = -1, None
    while time.time() < deadline:
        lines = epochs_done()
        alive = training_alive()
        if lines >= target_epochs:
            log(f"训练已完成 {lines}/{target_epochs} 轮")
            return "completed", lines
        if lines != last_lines:
            log(f"进度 {lines}/{target_epochs} 轮，训练进程{'在' if alive else '不在'}")
            last_lines, stable_since = lines, time.time()
        elif lines > 0 and not alive and stable_since and time.time() - stable_since > 300:
            log(f"训练进程已停止且 {lines} 轮未再增长，按提前结束处理")
            return "stopped", lines
        elif lines == 0 and not alive:
            # 实验尚未启动（例如由接力链排队中）：不判定为停止，继续等。
            stable_since = None
        # 睡眠不得超过剩余预算，否则小 max-hours 的检查会先睡满再超时。
        time.sleep(max(1.0, min(60.0, deadline - time.time())))
    log("等待超时")
    return "timeout", epochs_done()


def run(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    log("执行 " + " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-epochs", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--tag", default="rf48b1280")
    ap.add_argument("--run-name", default="run1280", help="BASE 下的训练目录名")
    ap.add_argument("--baseline-dir", default=str(CURRENT_BEST),
                    help="配对比较的基线预测目录")
    ap.add_argument("--baseline-conf", type=float, default=CURRENT_BEST_CONF)
    ap.add_argument("--baseline-nms", type=float, default=CURRENT_BEST_NMS)
    ap.add_argument("--baseline-label", default="当前最优")
    ap.add_argument("--check-only", action="store_true",
                    help="只验证守候逻辑，不执行收尾、不写结论")
    args = ap.parse_args()

    global RUN
    RUN = BASE / args.run_name
    summary_path = BASE / ("FINISHED.md" if args.run_name == "run1280"
                           else f"FINISHED-{args.run_name}.md")

    lock = BASE / f"finish-{args.run_name.replace('/', '_')}.lock"
    if lock.exists():
        log(f"发现 {lock}，疑似已有守候作业在运行，退出")
        return
    lock.write_text(str(os.getpid()))

    log(f"开始守候 {args.run_name}：目标 {args.target_epochs} 轮，最长 {args.max_hours} 小时")
    outcome, lines = wait_for_training(args.target_epochs, args.max_hours)
    log(f"守候结束：outcome={outcome} epochs={lines}")
    if args.check_only:
        log("--check-only：到此为止")
        lock.unlink(missing_ok=True)
        return
    if lines == 0 or not (RUN / "best.pt").exists():
        log("没有可用权重，不执行收尾")
        (summary_path).write_text(
            f"# 守候结果（{args.run_name}）\n\n训练未产生可用权重"
            f"（outcome={outcome}, epochs={lines}）。\n")
        return

    code = run("scripts.finalize_rf48b", "--run-name", args.run_name, "--tag", args.tag)
    log(f"收尾流水线退出码 {code}")
    final_dir = BASE / f"final-{args.run_name}"
    selection_path = final_dir / "selection.json"
    bootstrap = None
    if code == 0 and selection_path.exists():
        selection = json.loads(selection_path.read_text())
        conf = selection["selected"]["conf"]
        nms = selection["selected"]["nms_iou"]
        size = int(json.loads((RUN / "identity.json").read_text())["config"]["imgsz"])
        bootstrap_args = ["--a", str(final_dir / f"val-{size}"), "--a-conf", conf,
                          "--b", args.baseline_dir, "--b-conf", args.baseline_conf,
                          "--b-nms", args.baseline_nms, "--iterations", 400,
                          "--out", str(BASE / "paired"
                                       / f"{args.run_name.replace('/', '_')}-vs-baseline.json")]
        if nms is not None:
            bootstrap_args += ["--a-nms", nms]
        run("scripts.paired_bootstrap", *bootstrap_args)
        path = BASE / "paired" / f"{args.run_name.replace('/', '_')}-vs-baseline.json"
        if path.exists():
            bootstrap = json.loads(path.read_text())

    delivery = json.loads((final_dir / "delivery.json").read_text()) \
        if (final_dir / "delivery.json").exists() else None
    write_summary(outcome, lines, delivery, bootstrap, finalize_code=code,
                  run_name=args.run_name, tag=args.tag,
                  baseline_label=args.baseline_label,
                  baseline_dir=args.baseline_dir, baseline_conf=args.baseline_conf,
                  baseline_nms=args.baseline_nms, summary_path=summary_path)


def write_summary(outcome, lines, delivery, bootstrap, finalize_code=None,
                  run_name="run1280", tag="rf48b1280", baseline_label="当前最优",
                  baseline_dir=None, baseline_conf=None, baseline_nms=None,
                  summary_path=None):
    lines_out = ["# 训练收尾结论", "",
                 f"- 实验：`{run_name}`（tag `{tag}`）",
                 f"- 守候结果：`{outcome}`，完成 {lines} 轮",
                 f"- 收尾流水线退出码：`{finalize_code}`",
                 f"- 对比基线（{baseline_label}）：`{baseline_dir}` "
                 f"@ conf {baseline_conf} / NMS {baseline_nms}", ""]
    if finalize_code not in (0, None):
        lines_out += [
            "## ⚠ 收尾流水线失败", "",
            f"退出码 {finalize_code}。请查看 `runs/formal/rf48b/finish.log` 定位，",
            "或按 `runs/formal/rf48b/RUNBOOK.md` 手动执行：", "",
            "```bash",
            f".venv-rfdetr/bin/python -u -m scripts.finalize_rf48b "
            f"--run-name {run_name} --tag {tag}",
            "```", "",
            "**未产出新提交包，当前官方最优不受影响。**", ""]
    if delivery:
        local = delivery["local"]["ultralytics_map"] * 100
        cmp_ = delivery["comparison"]
        lines_out += [
            "## 新候选", "",
            f"- ZIP：`{delivery['candidate_zip']}`",
            f"- SHA-256：`{delivery['zip_sha256']}`",
            f"- 配置：画布 {delivery['config']['canvas']}，conf {delivery['config']['conf']}，"
            f"NMS {delivery['config']['nms_iou']}",
            f"- 本地 Ultralytics 口径：**{local:.4f}**（现最优 {CURRENT_BEST_LOCAL}，"
            f"增量 {cmp_['gain_over_current_best']:+.4f}）",
            f"- 本地 rule 口径：{delivery['local']['rule_map']*100:.4f}",
            f"- 平台均值 {delivery['platform']['mean']*100:.3f}，"
            f"最低 {delivery['platform']['min']*100:.3f}",
            f"- 是否过本地门槛（≥{cmp_['submit_gate_local']}）："
            f"{'是' if cmp_['passes_submit_gate'] else '否'}", ""]
    if bootstrap:
        d = bootstrap["paired_difference_ultralytics"]
        lines_out += [
            f"## 配对自助法（对{baseline_label}，同批 195 张图）", "",
            f"- 差值：**{d['mean']:+.4f} ± {d['std']:.4f}** 分",
            f"- 95% 区间：{d['p2.5']:+.4f} ~ {d['p97.5']:+.4f}",
            f"- **P(新候选更优) = {d['probability_a_better']*100:.1f}%**",
            f"- 提交判据为 ≥90%："
            f"{'**达到**' if d['probability_a_better'] >= 0.9 else '**未达到**'}", ""]
    lines_out += [
        "## 下一步", "",
        "1. 若 P ≥ 90%：把 ZIP 交给用户**手工上传**（本流程永不自动上传）。",
        "2. 若未达到：不出包，按 `docs/reports/2026-09-19-STATE.md` 第四节选择第二实验。",
        "3. 任何情况下保留恢复包 `phase1-rf48-mm-i768-c0p5-n0p6.zip` 与 `phase1-e9-*.zip`。",
        f"4. 详细状态与文档索引见 `docs/reports/2026-09-19-STATE.md`。", ""]
    path = summary_path or (BASE / "FINISHED.md")
    path.write_text("\n".join(lines_out) + "\n")
    log(f"结论已写入 {path}")
    print("\n".join(lines_out), flush=True)


if __name__ == "__main__":
    main()
