"""Offline, provenance-bound RF-DETR trainer and exported-TXT validation."""
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.trimodal_data import TrimodalDataset, collate, sha256, write_json, PREPROCESS_VERSION, load_config
from src.inference_checkpoint import load_legacy_checkpoint
from src.trimodal_rfdetr import build_detector, NestedTensor, output_rows
from src.evaluate_txt import evaluate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(config, side, size=None):
    dataset = TrimodalDataset(config["manifest"], side, size or config["imgsz"], side == "train",
                              aug=config.get("augmentation"))
    return DataLoader(dataset, batch_size=config["batch"] if side == "train" else 2,
                      shuffle=side == "train", num_workers=config["workers"],
                      pin_memory=True, collate_fn=collate, worker_init_fn=seed_worker)


@torch.no_grad()
def predict(model, data, out_dir, size, threshold=.001, zero=None):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {r["stem"] for r in data.dataset.samples}
    if {p.stem for p in out_dir.glob("*.txt")} - expected:
        raise ValueError("Extra predictions in output directory")
    started = time.monotonic()
    for images, masks, _, metadata in data:
        images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        if zero == "ir":
            images[:, 3:4] = 0
        elif zero == "depth":
            images[:, 4:6] = 0
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(NestedTensor(images, masks))
        if not torch.isfinite(out["pred_logits"]).all() or not torch.isfinite(out["pred_boxes"]).all():
            raise FloatingPointError("Non-finite predictions")
        for logit, box, meta in zip(out["pred_logits"], out["pred_boxes"], metadata):
            lines = output_rows(logit, box, meta["geometry"], size, threshold)
            (out_dir / (meta["stem"] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    torch.cuda.synchronize()
    return time.monotonic() - started


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(config, out_dir, split_manifest, stop_after=None, resume=None):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(config["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config["manifest"]).read_text())
    # 清单未记录划分来源时（例如"全部数据训练"的清单）跳过一致性校验；
    # 有来源时才要求与 --split-manifest 完全一致。
    if manifest.get("split_sha256") and manifest["split_sha256"] != sha256(split_manifest):
        raise ValueError("Explicit split and sample manifest disagree")
    identity = {"config": config, "manifest_sha256": sha256(config["manifest"]),
                "split_sha256": sha256(split_manifest), "preprocess": PREPROCESS_VERSION,
                "classes": manifest["classes"], "pretrained_sha256": sha256(config["pretrained"])}
    identity_path = out_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Refusing cross-identity resume/output overwrite")
    if (out_dir / "last.pt").exists() and not resume:
        raise ValueError("Existing run requires explicit --resume")
    write_json(identity_path, identity)
    write_json(out_dir / "split.json", manifest["split_manifest"])
    source_dir = out_dir / "source"
    source_dir.mkdir(exist_ok=True)
    import shutil
    for name in ("trimodal_data.py", "trimodal_rfdetr.py", "train_rfdetr.py", "evaluate_txt.py"):
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    environment = subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)
    (out_dir / "environment.txt").write_text(environment)
    model, criterion, transfer = build_detector(config)
    write_json(out_dir / "weight-transfer.json", transfer)
    model, criterion = model.cuda(), criterion.cuda()
    encoder, other = [], []
    for name, param in model.named_parameters():
        (encoder if name.startswith("backbone.0.encoder.") else other).append(param)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": config["lr_encoder"]},
                                   {"params": other, "lr": config["lr"]}], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=512.)
    has_val = any(row.get("split") == "val" for row in manifest["samples"])
    train_data = loader(config, "train")
    val_data = loader(config, "val") if has_val else None
    print(f"[train] has_val={has_val} train={len(train_data.dataset)}"
          f" val={len(val_data.dataset) if val_data else 0}", flush=True)
    steps_per_epoch = math.ceil(len(train_data) / config["accumulation"])
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = steps_per_epoch
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        span = total_steps - warmup_steps
        if span <= 0:
            # 退化情形（总步数不超过 warmup，例如极短探针）：保持恒等，避免除零。
            return 1.0
        return .5 * (1 + math.cos(math.pi * (step - warmup_steps) / span))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start_epoch, best, best_epoch, elapsed_prior = 0, -1., -1, 0.
    if resume:
        checkpoint = load_legacy_checkpoint(resume)
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["best_epoch"]
        elapsed_prior = checkpoint["elapsed_seconds"]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    max_epoch = min(config["epochs"], stop_after or config["epochs"])
    for epoch in range(start_epoch, max_epoch):
        if elapsed_prior + time.monotonic() - started >= config["max_hours"] * 3600:
            break
        # Epoch-specific seeds make clean epoch-boundary resumes reproducible.
        seed_all(config["seed"] + epoch)
        for param in encoder:
            param.requires_grad_(epoch >= config["freeze_epochs"])
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        consecutive_overflows = 0
        epoch_started = time.monotonic()
        for step, (images, masks, targets, _) in enumerate(train_data):
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            group_start = step // config["accumulation"] * config["accumulation"]
            divisor = min(config["accumulation"], len(train_data) - group_start)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(NestedTensor(images, masks))
                loss_dict = criterion(output, targets)
                loss = sum(v * criterion.weight_dict[k] for k, v in loss_dict.items() if k in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step}")
            scaler.scale(loss / divisor).backward()
            if (step + 1) % config["accumulation"] == 0 or step + 1 == len(train_data):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(norm):
                    scheduler.step()
                    consecutive_overflows = 0
                else:
                    consecutive_overflows += 1
                    print("AMP_overflow_step_skipped", epoch, step, scaler.get_scale(), flush=True)
                    if consecutive_overflows >= 5:
                        raise FloatingPointError("Five consecutive nonfinite gradient steps")
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                status = {"stage": "training", "epoch": epoch + 1, "step": step, "steps": len(train_data),
                          "loss": float(loss.detach()), "best_score": best,
                          "elapsed_seconds": elapsed_prior + time.monotonic() - started,
                          "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30}
                write_json(out_dir / "status.json", status)
                print(json.dumps(status), flush=True)
        if val_data is not None:
            seconds = predict(model, val_data, out_dir / "val-latest", config["imgsz"])
            metrics = evaluate(config["manifest"], out_dir / "val-latest", coco=epoch == 0)
            # 2026-09-19 口径定案：官方 = Ultralytics 一族（三点偏差 -0.82/+1.21/-0.16）。
            # 选模改用该口径，并在与最终打包一致的高置信度平台上取较优值。
            high = evaluate(config["manifest"], out_dir / "val-latest",
                            threshold=config.get("select_conf", .50))
            score = max(metrics["ultralytics_map"], high["ultralytics_map"])
            metrics["selection_score"] = score
            metrics["select_conf"] = config.get("select_conf", .50)
            metrics["ultralytics_map_hi"] = high["ultralytics_map"]
            metrics["rule_map_hi"] = high["rule_map"]
            improved = score > best
            if improved:
                best, best_epoch = score, epoch
        else:
            # 无验证集模式（用全部数据训练）：不做评估，改为按 checkpoint_every 定期存点，
            # 事后按已知的峰值位置（本配方为 65~70 轮）挑选交付检查点。
            seconds, metrics, improved = 0.0, {"selection_score": None, "no_val": True}, False
        elapsed = elapsed_prior + time.monotonic() - started
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics,
                  "epoch_seconds": time.monotonic() - epoch_started, "inference_seconds": seconds,
                  "elapsed_seconds": elapsed, "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
                  "best_epoch": best_epoch + 1}
        with (out_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {"model": model.state_dict(), "identity": identity, "epoch": epoch,
                   "best": best, "best_epoch": best_epoch, "elapsed_seconds": elapsed,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
        save_checkpoint(out_dir / "last.pt", payload)
        weights_only = {k: v for k, v in payload.items() if k not in {"optimizer", "scheduler", "scaler"}}
        if improved:
            save_checkpoint(out_dir / "best.pt", weights_only)
            write_json(out_dir / "best-metrics.json", record)
        every = config.get("checkpoint_every") or 0
        if every and (epoch + 1) % every == 0:
            save_checkpoint(out_dir / f"epoch-{epoch + 1:03d}.pt", weights_only)
        print("epoch_complete", json.dumps(record), flush=True)
        # patience=0 / 缺省表示关闭早停：现模型正是被 195 张验证集的噪声在第 40 轮掐掉的。
        patience = config.get("patience") or 0
        if patience and epoch + 1 >= 20 and epoch - best_epoch >= patience:
            break
    if val_data is None and not (out_dir / "best.pt").exists() and (out_dir / "last.pt").exists():
        # 无验证集时给下游流水线一个可用的 best.pt（取最后一轮）；正式交付建议改用
        # epoch-0XX.pt 中位于已知峰值（65~70 轮）的那一份。
        final = load_legacy_checkpoint(out_dir / "last.pt")
        save_checkpoint(out_dir / "best.pt",
                        {k: v for k, v in final.items() if k not in {"optimizer", "scheduler", "scaler"}})
    write_json(out_dir / "status.json", {"stage": "finished_or_budget_stop", "best_score": best,
               "best_epoch": best_epoch + 1, "no_val": val_data is None,
               "elapsed_seconds": elapsed_prior + time.monotonic() - started})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--resume")
    args = p.parse_args()
    train(load_config(args.config), args.out, args.split_manifest, args.stop_after, args.resume)
