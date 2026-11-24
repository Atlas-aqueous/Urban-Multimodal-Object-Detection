"""Real-data probe: zero fusion, encoder gradients, AMP and batch memory."""
import argparse
import gc
import time
from pathlib import Path

import torch
import yaml

from src.trimodal_data import write_json, sha256, load_config
from src.trimodal_rfdetr import build_detector, NestedTensor
from src.train_rfdetr import loader, seed_all


def main(config, steps, output):
    torch.set_num_threads(4)
    seed_all(config["seed"])
    data = loader(config, "train")
    images, masks, targets, _ = next(iter(data))
    images, masks = images.cuda(), masks.cuda()
    model, criterion, transfer = build_detector(config)
    model.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        fused = model(NestedTensor(images, masks))
    control, _, _ = build_detector(config | {"multimodal": False})
    control.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        rgb = control(NestedTensor(images, masks))
    difference = max(float((fused[k] - rgb[k]).abs().max()) for k in ("pred_logits", "pred_boxes"))
    assert difference == 0, difference
    del control, fused, rgb
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    criterion.cuda().train()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=512.)
    started, losses = time.monotonic(), []
    iterator = iter(data)
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        images, masks, targets, _ = next(iterator)
        images, masks = images.cuda(), masks.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16):
            output_dict = model(NestedTensor(images, masks))
            parts = criterion(output_dict, targets)
            loss = sum(v * criterion.weight_dict[k] for k, v in parts.items() if k in criterion.weight_dict)
        assert torch.isfinite(loss), f"Nonfinite loss at {step}"
        scaler.scale(loss / config["accumulation"]).backward()
        if (step + 1) % config["accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if step % 20 == 0:
            print("probe", step, losses[-1], torch.cuda.max_memory_allocated()/2**30, flush=True)
    # For steps divisible by accumulation, do a fresh gradient-only check.
    optimizer.zero_grad(set_to_none=True)
    output_dict = model(NestedTensor(images, masks))
    parts = criterion(output_dict, targets)
    sum(v * criterion.weight_dict[k] for k, v in parts.items() if k in criterion.weight_dict).backward()
    gradients = {}
    for group in ("encoder", "ir_branch.body", "depth_branch.body"):
        values = [p.grad for n, p in model.named_parameters() if f"backbone.0.{group}" in n and p.grad is not None]
        gradients[group] = sum(float(v.abs().sum()) for v in values)
        assert values and all(torch.isfinite(v).all() for v in values) and gradients[group] > 0, group
    seconds = time.monotonic() - started
    report = {"passed": True, "steps": steps, "config": config, "zero_fusion_max_difference": difference,
              "gradients": gradients, "seconds": seconds, "images_per_second": steps * config["batch"] / seconds,
              "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
              "first_loss": losses[0], "last_loss": losses[-1], "transfer": transfer,
              "pretrained_sha256": sha256(config["pretrained"]),
              "pretrained_url": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth"}
    assert report["peak_gpu_gib"] <= 28
    write_json(output, report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rf48-mm.yaml")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="runs/formal/rf48/probe.json")
    args = p.parse_args()
    main(load_config(args.config), args.steps, args.out)
