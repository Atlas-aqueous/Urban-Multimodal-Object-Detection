"""为旧实验记录回填后续新增的数据划分和环境字段。"""
import argparse
import json
from pathlib import Path

import torch

try:
    from src.train import data_provenance, load_yaml
except ModuleNotFoundError:
    from train import data_provenance, load_yaml


def backfill(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("command_args", {})
    if args.get("data") and "data_provenance" not in payload:
        payload["data_provenance"] = data_provenance(Path(args["data"]))
    payload.setdefault("ultralytics_train_args", load_yaml(path.parent / "args.yaml"))
    environment = payload.setdefault("environment", {})
    environment.setdefault("cuda", torch.version.cuda)
    environment.setdefault("cudnn", torch.backends.cudnn.version())
    # 旧进程结束后无法还原 reserved 峰值，明确记 null 而不伪造。
    payload.setdefault("peak_gpu_reserved_gib", None)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[backfill] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", nargs="+", type=Path)
    args = ap.parse_args()
    for record in args.records:
        backfill(record)
