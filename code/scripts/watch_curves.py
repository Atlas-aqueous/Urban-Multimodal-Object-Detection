"""无人值守地盯着运行目录，把新出现的检查点自动评分并画曲线。

用途：实验 A（全量 2000 训练，其中含验证集 195 张）没有干净验证集，
其验证分**含记忆化虚高**。本脚本持续采集「虚高曲线」，
并与干净曲线对比，判断虚高是否已经饱和 —— 只有饱和之后，
虚高曲线的峰值位置才等于真实峰值位置。

    python -u -m scripts.watch_curves --interval 240 --max-hours 16
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_checkpoints import score_at  # noqa: E402

CLEAN_REF = "runs/formal/rf48b/run-early1280/history.jsonl"
STATE = ROOT / "runs/formal/rf48c/curve-watch.json"


def load_ref():
    """干净参考：早融合 1280 运行，从未见过这 195 张。

    history 里的 selection_score = max(ul@默认, ul@conf0.50)，
    与我们打包用的 conf0.35/nms0.6 略有差别，故只当作「形状参考」。
    """
    p = ROOT / CLEAN_REF
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("selection_score")
        if s is not None:
            out[r["epoch"]] = s
    return out


def discover(run_dir, min_age=90.0):
    now = time.time()
    found = {}
    for ck in sorted(run_dir.glob("epoch-*.pt")):
        try:
            if now - ck.stat().st_mtime < min_age:
                continue  # 可能正在写
        except FileNotFoundError:
            continue
        found[int(ck.stem.split("-")[1])] = ck
    return found


def evaluate(run_dir, ckpt, manifest, conf, nms, size, side="val"):
    pred_dir = run_dir / f"val-{ckpt.stem}"
    if not (pred_dir / "identity.json").exists():
        log(f"  推理 {ckpt.name} ...")
        t0 = time.time()
        subprocess.run([sys.executable, "-u", "-m", "src.predict_rfdetr",
                        "--weights", str(ckpt), "--sample-manifest", manifest,
                        "--out-dir", str(pred_dir), "--side", side,
                        "--imgsz", str(size), "--conf", "0.001"],
                       cwd=ROOT, check=True)
        log(f"  推理完成 {time.time()-t0:.0f}s")
    rep = score_at(pred_dir, manifest, conf, nms)
    return rep["ultralytics_map"], rep["rule_map"], rep["ultralytics_per_class"]


LOG_PATH = ROOT / "runs/formal/rf48c/curve-watch.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


def diagnose(a_rows, b_rows, ref, p1=0.50, p2=0.35, p3=0.30):
    """打印虚高曲线诊断。"""
    line = "=" * 74
    log(line)
    if a_rows:
        log("实验 A（全量 2000，验证集在训练集内 → 分数虚高）")
        log(f"  {'轮次':>5} {'虚高分':>9} {'干净参考':>10} {'虚高Δ':>8} {'Δ增量':>8}")
        prev_d = None
        for r in a_rows:
            e = r["epoch"]
            c = ref.get(e)
            dl = None if c is None else r["ultralytics_map"] - c
            inc = "" if (dl is None or prev_d is None) else f"{(dl - prev_d)*100:+.3f}"
            if dl is not None:
                prev_d = dl
            log(f"  {e:>5} {r['ultralytics_map']*100:>9.4f} "
                f"{(c*100 if c else float('nan')):>10.4f} "
                f"{(dl*100 if dl is not None else float('nan')):>8.3f} {inc:>8}")
        best = max(a_rows, key=lambda r: r["ultralytics_map"])
        log(f"  虚高曲线当前峰值：第 {best['epoch']} 轮 {best['ultralytics_map']*100:.4f}")
        if len(a_rows) >= 3:
            d1 = a_rows[-1]["ultralytics_map"] - a_rows[-2]["ultralytics_map"]
            d2 = a_rows[-2]["ultralytics_map"] - a_rows[-3]["ultralytics_map"]
            trend = "仍在上升" if d1 > p2 / 100 else ("走平/见顶" if abs(d1) < p2 / 100 else "回落")
            log(f"  最近两档增量 {d2*100:+.3f} → {d1*100:+.3f}  →  {trend}")
    if b_rows:
        log("实验 B（1805 训练 / 195 留出 → 干净曲线）")
        best = max(b_rows, key=lambda r: r["ultralytics_map"])
        log(f"  峰值：第 {best['epoch']} 轮 {best['ultralytics_map']*100:.4f}")
        for r in b_rows:
            log(f"    ep{r['epoch']:>3} {r['ultralytics_map']*100:.4f}")
    log(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=240)
    ap.add_argument("--max-hours", type=float, default=16)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.6)
    ap.add_argument("--manifest", default="runs/formal/rf48/train-manifest.json")
    ap.add_argument("--run-a", default="runs/formal/rf48c/run-all2000")
    ap.add_argument("--run-b", default="runs/formal/rf48c/run-aug")
    args = ap.parse_args()

    state = {"conf": args.conf, "nms": args.nms, "runs": {}}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text())
            state.setdefault("runs", {})
        except Exception:  # noqa: BLE001
            pass

    ref = load_ref()
    deadline = time.time() + args.max_hours * 3600
    known = {"a": {}, "b": {}}
    for key, rel in (("a", args.run_a), ("b", args.run_b)):
        for r in state["runs"].get(key, {}).get("rows", []):
            known[key][r["epoch"]] = r

    log(f"启动监控 interval={args.interval}s max={args.max_hours}h "
        f"conf={args.conf} nms={args.nms}")

    idle = 0
    while time.time() < deadline:
        for key, rel in (("a", args.run_a), ("b", args.run_b)):
            run_dir = ROOT / rel
            if not run_dir.exists():
                continue
            idf = run_dir / "identity.json"
            if not idf.exists():
                continue
            size = int(json.loads(idf.read_text())["config"]["imgsz"])
            for epoch, ck in discover(run_dir).items():
                if epoch in known[key]:
                    continue
                try:
                    ul, rule, per = evaluate(run_dir, ck, args.manifest,
                                             args.conf, args.nms, size)
                except Exception as exc:  # noqa: BLE001
                    log(f"  !! {key} ep{epoch} 评估失败：{exc!r}")
                    continue
                known[key][epoch] = {"epoch": epoch, "ultralytics_map": ul,
                                     "rule_map": rule,
                                     "ultralytics_per_class": per}
                rows = [known[key][e] for e in sorted(known[key])]
                state["runs"][key] = {"run_dir": rel, "imgsz": size, "rows": rows}
                STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
                log(f"  {key} ep{epoch} @imgsz{size} → ul {ul*100:.4f} (rule {rule*100:.4f})")

        a_rows = [known["a"][e] for e in sorted(known["a"])]
        b_rows = [known["b"][e] for e in sorted(known["b"])]
        diagnose(a_rows, b_rows, ref)

        # 两个运行都到 100 轮且无新点 → 收工
        done = all((ROOT / rel / "epoch-100.pt").exists()
                   for rel in (args.run_a, args.run_b))
        if done and len(a_rows) >= 10 and len(b_rows) >= 10:
            log("两个运行的 100 轮检查点均已评估，收工")
            break
        idle = idle + 1 if not a_rows and not b_rows else 0
        time.sleep(args.interval)

    log("监控退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
