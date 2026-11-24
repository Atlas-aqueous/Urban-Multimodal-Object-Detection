"""可恢复地切换活动 TIFF 数据集，生成并校验 E6 优化提交包。"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ACTIVE = PROJECT / "datasets/formal-train-active"
TRAIN_RAW = PROJECT / "formal_data/train_revised_labels"
TEST_RAW = PROJECT / "formal_data/phase1_test"
SPLIT = PROJECT / "runs/formal/yolov8m-5ch-revised-960/reproducibility/split.json"
STATE = PROJECT / "runs/formal/optimized-submission-state.json"


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
    print(f"[optimized-submit] {stage}: {extra}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("[optimized-submit] run:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def prepare_test() -> Path:
    run([
        sys.executable, "src/prepare_data.py", "--data-root", str(TEST_RAW),
        "--out", str(ACTIVE), "--mode", "5ch", "--no-labels", "--overwrite",
    ])
    cfg = yaml.safe_load((ACTIVE / "data.yaml").read_text(encoding="utf-8"))
    image_dir = ACTIVE / "images/test"
    count = sum(1 for _ in image_dir.glob("*.tiff"))
    if cfg.get("channels") != 5 or count != 1000:
        raise RuntimeError(f"测试派生集校验失败: channels={cfg.get('channels')}, count={count}")
    return image_dir


def restore_train() -> None:
    run([
        sys.executable, "src/prepare_data.py", "--data-root", str(TRAIN_RAW),
        "--out", str(ACTIVE), "--mode", "5ch", "--split-manifest", str(SPLIT),
        "--rare-repeat", "--overwrite",
    ])
    cfg = yaml.safe_load((ACTIVE / "data.yaml").read_text(encoding="utf-8"))
    count = sum(1 for _ in (ACTIVE / "images").rglob("*.tiff"))
    if cfg.get("channels") != 5 or cfg.get("train") != "train_weighted.txt" or count != 2000:
        raise RuntimeError(f"训练派生集恢复失败: channels={cfg.get('channels')}, "
                           f"train={cfg.get('train')}, count={count}")


def main(args: argparse.Namespace) -> None:
    weights = args.weights.resolve()
    old_zip = args.old_zip.resolve()
    out_zip = args.out_zip.resolve()
    pred_dir = args.pred_dir.resolve()
    audit = out_zip.with_suffix(".audit.json")
    if not weights.is_file() or not old_zip.is_file():
        raise FileNotFoundError("缺少 E6 权重或旧提交包")
    if out_zip == old_zip:
        raise ValueError("新提交包不得覆盖旧提交包")
    guard = {
        "weights": {"path": str(weights), "sha256": sha256(weights)},
        "old_submission": {"path": str(old_zip), "sha256": sha256(old_zip)},
    }
    write_state("artifacts_verified", guard=guard)
    completed = False
    try:
        write_state("building_test_5ch", guard=guard)
        image_dir = prepare_test()
        write_state("predicting", image_dir=str(image_dir), conf=args.conf, iou=args.iou)
        run([
            sys.executable, "src/predict.py", "--weights", str(weights),
            "--image-dir", str(image_dir), "--out-dir", str(pred_dir),
            "--imgsz", str(args.imgsz), "--conf", str(args.conf),
            "--iou", str(args.iou), "--max-det", "100", "--device", "0", "--half",
        ])
        write_state("packing", prediction_dir=str(pred_dir))
        run([
            sys.executable, "src/check_submit.py", "--pred-dir", str(pred_dir),
            "--image-dir", str(image_dir), "--out", str(out_zip),
            "--max-det", "100", "--audit-json", str(audit),
        ])
        completed = True
    finally:
        write_state("restoring_train_5ch", submission_created=completed)
        restore_train()
    if not completed or not out_zip.is_file():
        raise RuntimeError("优化提交包未生成")
    write_state("complete", submission=str(out_zip), audit=str(audit),
                sha256=sha256(out_zip), guard=guard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=PROJECT / "runs/formal/e6-p2-1280/weights/best.pt")
    parser.add_argument("--old-zip", type=Path,
                        default=PROJECT / "runs/formal/submissions/phase1-e6-yolov8m-p2-5ch-1280.zip")
    parser.add_argument("--out-zip", type=Path,
                        default=PROJECT / "runs/formal/submissions/phase1-e6-p2-1280-c013-i040.zip")
    parser.add_argument("--pred-dir", type=Path,
                        default=PROJECT / "runs/formal/phase1-predictions-e6-p2-1280-c013-i040")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.13)
    parser.add_argument("--iou", type=float, default=0.40)
    main(parser.parse_args())
