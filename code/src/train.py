"""训练入口（方案A/B通用）。HSV增强关闭（IR/Depth通道不适用色彩抖动）。

用法:
  方案B: python src/train.py --data datasets/train-b/data.yaml --model yolov8m.pt --name exp-b
  方案A: python src/train.py --data datasets/train-a/data.yaml --model runs/pretrain/yolov8m-5ch.pt --name exp-a
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO

try:
    from src.diagnostics import size_recall
except ModuleNotFoundError:
    from diagnostics import size_recall


def apply_modality_dropout(image: torch.Tensor, probability: float) -> torch.Tensor:
    """原位执行模态 dropout；6ch 的 Depth 与 DepthValid 必须同步清零。"""
    if image.ndim != 4 or image.shape[1] not in (5, 6):
        raise RuntimeError("modality_dropout 仅适用于5/6通道多模态输入")
    batch_size = image.shape[0]
    ir_mask = torch.rand(batch_size, device=image.device) < probability
    depth_mask = torch.rand(batch_size, device=image.device) < probability
    image[ir_mask, 3] = 0
    image[depth_mask, 4] = 0
    if image.shape[1] == 6:
        image[depth_mask, 5] = 0
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def data_provenance(data_yaml: Path, split_manifest: Path | None = None) -> dict:
    """固化数据配置和完整划分清单，避免后续重建活动数据集后丢失实验身份。"""
    data_yaml = Path(data_yaml).resolve()
    split_path = Path(split_manifest).resolve() if split_manifest else data_yaml.parent / "split.json"
    if split_manifest and not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not split_path.exists():
        cfg = load_yaml(data_yaml) or {}
        root = Path(cfg.get("path") or data_yaml.parent)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        split_path = root / "split.json"
    payload = {
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
        "data_config": load_yaml(data_yaml),
        "split_manifest": None,
    }
    if split_path.exists():
        payload["split_manifest"] = {
            "path": str(split_path),
            "sha256": sha256(split_path),
            "content": json.loads(split_path.read_text(encoding="utf-8")),
        }
    return payload


def metric_payload(result) -> dict:
    box = getattr(result, "box", result)
    names = getattr(result, "names", {})
    maps = list(map(float, getattr(box, "maps", [])))
    speed = getattr(result, "speed", {}) or {}
    return {
        "map50_95": float(getattr(box, "map", float("nan"))),
        "map50": float(getattr(box, "map50", float("nan"))),
        "map75": float(getattr(box, "map75", float("nan"))),
        "speed_ms_per_image": {str(key): float(value) for key, value in speed.items()},
        "per_class_ap50_95": {
            str(names.get(i, i)): value for i, value in enumerate(maps)
        },
    }


def validation_label_dir(data_yaml: Path) -> Path | None:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    if not cfg.get("val"):
        return None
    root = Path(cfg.get("path") or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    val = Path(cfg["val"])
    parts = list(val.parts)
    if "images" not in parts:
        return None
    parts[parts.index("images")] = "labels"
    return root.joinpath(*parts)


def main(args: argparse.Namespace):
    model = YOLO(args.model)
    if not 0.0 <= args.modality_dropout < 1.0:
        raise ValueError("modality_dropout 必须在 [0,1) 内")

    if args.modality_dropout:
        def install_modality_dropout(trainer):
            """在设备传输/归一化后包装 batch 预处理，避免依赖未公开的当前 batch 属性。"""
            original = trainer.preprocess_batch

            def preprocess_with_dropout(batch):
                batch = original(batch)
                image = batch["img"]
                apply_modality_dropout(image, args.modality_dropout)
                return batch

            trainer.preprocess_batch = preprocess_with_dropout

        model.add_callback("on_train_start", install_modality_dropout)

    # project 必须转绝对路径：ultralytics 对相对 project 的解析在中文目录下
    # 会退回到其 settings 的 runs_dir（本机实测丢掉中文段，产物散落到桌面）
    project = Path(args.project).resolve()
    start_record = project / args.name / "start-provenance.json"
    start_record.parent.mkdir(parents=True, exist_ok=True)
    start_record.write_text(json.dumps({
        "data_provenance": data_provenance(Path(args.data), getattr(args, "split_manifest", None)),
        "initial_weights_sha256": sha256(Path(args.model)),
        "command_args": vars(args),
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = model.train(
        data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(project), name=args.name, exist_ok=True,
        resume=args.resume,
        patience=args.patience, seed=args.seed, deterministic=True, val=not args.no_val,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, cos_lr=args.cos_lr,
        weight_decay=args.weight_decay, device=args.device, cache=args.cache, amp=not args.no_amp,
        warmup_epochs=args.warmup_epochs, box=args.box, cls=args.cls, dfl=args.dfl,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,   # 多通道/含IR-D通道，禁用色彩抖动
        fliplr=0.5, mosaic=args.mosaic, mixup=args.mixup,
        close_mosaic=args.close_mosaic, multi_scale=args.multi_scale,
        workers=args.workers, plots=True, verbose=True,
    )
    if results is not None:
        # ultralytics 8.3.253: train() 返回 DetMetrics，.map/.map50 在其 .box(Metric) 上
        m = getattr(results, "box", results)
        if hasattr(m, "map"):
            print(f"[train] mAP50-95 = {m.map:.4f} | mAP50 = {m.map50:.4f}")
    run_dir = project / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    selected = best if best.exists() else last
    official_metrics = None
    if not args.no_official_eval and not args.no_val:
        evaluated = YOLO(selected).val(
            data=str(Path(args.data).resolve()), imgsz=args.eval_imgsz or args.imgsz,
            conf=args.eval_conf, iou=args.eval_iou, max_det=100,
            device=args.device, half=not args.no_amp, augment=False,
            agnostic_nms=False, plots=False, verbose=True, save_txt=True, save_conf=True,
            project=str(run_dir), name="official-val", exist_ok=True,
        )
        official_metrics = metric_payload(evaluated)
        label_dir = validation_label_dir(Path(args.data))
        prediction_dir = Path(evaluated.save_dir) / "labels"
        if label_dir and prediction_dir.exists():
            official_metrics["size_diagnostics"] = size_recall(label_dir, prediction_dir)
    payload = {
        "elapsed_seconds": time.time() - started,
        "command_args": vars(args) | {"data": str(args.data), "model": str(args.model),
                                      "project": str(args.project)},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        },
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        ),
        "peak_gpu_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        ),
        "data_provenance": data_provenance(Path(args.data), getattr(args, "split_manifest", None)),
        "ultralytics_train_args": load_yaml(run_dir / "args.yaml"),
        "selected_weights": str(selected.resolve()),
        "weights_sha256": sha256(selected),
        "official_validation": official_metrics,
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"[train] selected weights → {selected}")
    print(f"[train] experiment record → {run_dir / 'experiment.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="data.yaml 路径")
    ap.add_argument("--split-manifest", type=Path, help="显式绑定不可变场景划分清单")
    ap.add_argument("--model", required=True, type=Path, help=".pt 权重（手术后的5ch或官方3ch）")
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch", default=16, type=int)
    ap.add_argument("--patience", default=20, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--optimizer", default="AdamW", type=str,
                    help="显式固定优化器，避免 optimizer=auto 随版本改变")
    ap.add_argument("--lr0", default=1e-3, type=float)
    ap.add_argument("--lrf", default=1e-2, type=float)
    ap.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--weight-decay", default=5e-4, type=float)
    ap.add_argument("--warmup-epochs", default=3.0, type=float)
    ap.add_argument("--box", default=7.5, type=float)
    ap.add_argument("--cls", default=0.5, type=float)
    ap.add_argument("--dfl", default=1.5, type=float)
    ap.add_argument("--close-mosaic", default=10, type=int)
    ap.add_argument("--mosaic", default=1.0, type=float)
    ap.add_argument("--mixup", default=0.0, type=float)
    ap.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--device", default=None, help="如 0、cpu；默认自动选择")
    ap.add_argument("--cache", action="store_true", help="将图像缓存到内存（全量数据需足够RAM）")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--modality-dropout", default=0.0, type=float,
                    help="5/6ch训练时 IR/Depth 整模态置零概率")
    ap.add_argument("--no-val", action="store_true", help="全量最终训练时关闭验证与早停")
    ap.add_argument("--no-official-eval", action="store_true", help="训练后不做 max_det=100 复验")
    ap.add_argument("--resume", action="store_true",
                    help="从 --model 指定的 last.pt 恢复优化器、epoch 和随机状态")
    ap.add_argument("--eval-imgsz", type=int, help="训练后复验分辨率；默认等于训练分辨率")
    ap.add_argument("--eval-conf", default=0.001, type=float)
    ap.add_argument("--eval-iou", default=0.7, type=float)
    ap.add_argument("--project", default="runs", type=Path)
    ap.add_argument("--name", default="exp", type=str)
    main(ap.parse_args())
