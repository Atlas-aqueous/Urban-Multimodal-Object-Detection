"""轻量显存/吞吐探针：在指定画布上实跑训练步，检查损失有限性与峰值显存。

与 scripts/probe_rfdetr.py 的区别：不做"零残差 == RGB 对照"的等价性检查
（该性质已在 768 画布验证），只测真实数据下的训练步成本，避免同时驻留两个模型。
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.trimodal_data import load_config, write_json
from src.trimodal_rfdetr import NestedTensor, build_detector
from src.train_rfdetr import loader, seed_all


def main(config, steps, output):
    torch.set_num_threads(4)
    seed_all(config["seed"])
    data = loader(config, "train")
    model, criterion, transfer = build_detector(config)
    model, criterion = model.cuda(), criterion.cuda()
    model.train()
    criterion.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=512.)
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(data)
    losses, seen, started = [], 0, time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        images, masks, targets, _ = next(iterator)
        images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(NestedTensor(images, masks))
            parts = criterion(out, targets)
            loss = sum(v * criterion.weight_dict[k] for k, v in parts.items()
                       if k in criterion.weight_dict)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        scaler.scale(loss / config["accumulation"]).backward()
        if (step + 1) % config["accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        seen += int(images.shape[0])
        print(f"step {step} loss {losses[-1]:.4f} peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
              flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    report = {"config": config, "steps": steps, "images": seen, "seconds": seconds,
              "images_per_second": seen / seconds,
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "loss_first": losses[0], "loss_last": losses[-1], "finite": True,
              "weight_transfer": {k: v for k, v in transfer.items() if k != "missing_keys"}}
    write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--imgsz", type=int, help="覆盖画布尺寸（小尺寸快速验证训练路径）")
    p.add_argument("--fusion", choices=["late", "early"], help="覆盖融合方式")
    a = p.parse_args()
    config = load_config(a.config)
    if a.imgsz:
        config["imgsz"] = a.imgsz
    if a.fusion:
        config["fusion"] = a.fusion
    main(config, a.steps, a.out)
