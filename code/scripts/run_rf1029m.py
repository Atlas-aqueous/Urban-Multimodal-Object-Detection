"""Train RF1029-M or export predictions with the semifinal postprocessing."""
import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.apply_nms import apply
from src.inference_checkpoint import load_inference_checkpoint
from src.rf1029_data import RF1029Dataset
from src.rf1029_model import build_detector
from src.train_rfdetr import save_checkpoint, seed_all, seed_worker
from src.trimodal_data import collate, load_config, load_manifest, write_json
from src.trimodal_rfdetr import NestedTensor, output_rows

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/rf1029m-1728x960.yaml"


def loader(dataset, config, seed, shuffle=False):
    return DataLoader(dataset, batch_size=config["batch"], num_workers=config["workers"],
                      shuffle=shuffle, pin_memory=True, collate_fn=collate,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def device_batch(batch):
    images, masks, targets, metadata = batch
    samples = NestedTensor(images.cuda(non_blocking=True), masks.cuda(non_blocking=True))
    samples.content_valid = torch.stack([row["content_valid"] for row in metadata]).cuda(non_blocking=True)
    targets = [{key: value.cuda(non_blocking=True) for key, value in row.items()} for row in targets]
    return samples, targets, metadata


def fp32(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: fp32(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(fp32(item) for item in value)
    return value


def train(config, args):
    manifest, _ = load_manifest(args.manifest)
    checkpoint, _ = load_inference_checkpoint(args.init_weights, args.init_weights_sha256)
    model, criterion = build_detector(config, checkpoint["model"], initialize=True)
    del checkpoint
    model = model.cuda()
    criterion = criterion.cuda()
    groups = {"encoder": [], "existing": [], "validity": []}
    for name, parameter in model.named_parameters():
        if name.endswith(".validity_gamma"):
            groups["validity"].append(parameter)
        elif name.startswith("backbone.0.encoder."):
            groups["encoder"].append(parameter)
        else:
            groups["existing"].append(parameter)
    rates = {"encoder": config["lr_encoder"], "existing": config["lr"], "validity": config["lr_new"]}
    optimizer = torch.optim.AdamW(
        [{"name": name, "params": params, "lr": rates[name]} for name, params in groups.items()],
        weight_decay=config["weight_decay"])

    dataset = RF1029Dataset(config, args.manifest, "train")
    accumulation = config["accumulation"]
    batches = math.ceil(len(dataset) / config["batch"])
    total = math.ceil(batches / accumulation) * config["epochs"]
    warmup = config["warmup_updates"]

    def schedule_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 1.0
        return .5 * (1 + math.cos(math.pi * (min(step, total) - warmup) / (total - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_factor)
    updates = 0
    model.train()
    criterion.train()
    for epoch in range(1, config["epochs"] + 1):
        epoch_seed = config["seed"] + epoch - 1
        seed_all(epoch_seed)
        dataset.epoch = epoch - 1
        data = loader(dataset, config, epoch_seed, shuffle=True)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(data):
            samples, targets, _ = device_batch(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(samples)
            values = criterion(fp32(output), targets)
            loss = sum(value * criterion.weight_dict[key] for key, value in values.items()
                       if key in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
            group_size = min(accumulation, batches - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
                updates += 1
            losses.append(float(loss.detach()))

        payload = {"model": model.state_dict(), "epoch": epoch, "optimizer_updates": updates,
                   "identity": {"config": config, "classes": manifest["classes"],
                                "preprocess": manifest["preprocess"]}}
        if epoch in (4, config["epochs"]):
            save_checkpoint(args.out / f"ep{epoch:03d}.pt", payload)
        save_checkpoint(args.out / "last.pt", {**payload, "optimizer": optimizer.state_dict(),
                                               "scheduler": schedule.state_dict()})
        status = {"epoch": epoch, "updates": updates, "loss": float(np.mean(losses))}
        write_json(args.out / "status.json", status)
        print(status, flush=True)


@torch.inference_mode()
def infer(config, args):
    manifest, _ = load_manifest(args.manifest, args.side)
    checkpoint, weights_sha256 = load_inference_checkpoint(args.weights, args.weights_sha256)
    identity = checkpoint["identity"]
    if any(identity[key] != manifest[key] for key in ("classes", "preprocess")):
        raise ValueError("Checkpoint and manifest disagree on classes or preprocessing")
    model, _ = build_detector(config, checkpoint["model"])
    del checkpoint
    model = model.requires_grad_(False).eval().cuda()

    dataset = RF1029Dataset(config, args.manifest, args.side)
    data = loader(dataset, config, config["seed"])
    raw = args.out / "raw"
    raw.mkdir()
    for index, batch in enumerate(data, start=1):
        samples, _, metadata = device_batch(batch)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(samples)
        if not all(torch.isfinite(result[key]).all() for key in ("pred_logits", "pred_boxes")):
            raise FloatingPointError(f"Non-finite model output for {metadata[0]['stem']}")
        for logits, boxes, row in zip(result["pred_logits"], result["pred_boxes"], metadata):
            lines = output_rows(logits, boxes, row["geometry"], dataset.size, config["raw_conf"])
            (raw / (row["stem"] + ".txt")).write_text("".join(line + "\n" for line in lines))
        if index % 50 == 0 or index == len(data):
            print(f"Predicted {min(index * config['batch'], len(dataset))}/{len(dataset)} images", flush=True)

    predictions = args.out / "predictions"
    stats = apply(raw, predictions, config["nms_input_conf"], config["nms"])
    files = sorted(predictions.glob("*.txt"))
    if len(files) != len(dataset) or stats["dropped_rows"]:
        raise ValueError("Incomplete or invalid prediction output")
    boxes = empty = 0
    with zipfile.ZipFile(args.out / "result.zip", "w", zipfile.ZIP_DEFLATED) as package:
        for path in files:
            # Apply the final threshold to the six-decimal TXT exported by NMS.
            lines = [line for line in path.read_bytes().splitlines(keepends=True)
                     if float(line.split()[5]) >= config["conf"]]
            path.write_bytes(b"".join(lines))
            package.write(path, path.name)
            boxes += len(lines)
            empty += not lines
    write_json(args.out / "summary.json", {"images": len(files), "boxes": boxes,
               "empty_txt": empty, "weights_sha256": weights_sha256, "config": config})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fine-tune from the v3 checkpoint")
    training.add_argument("--init-weights", required=True, type=Path)
    training.add_argument("--init-weights-sha256", help="Expected full SHA-256 of the initial checkpoint")
    prediction = commands.add_parser("infer", help="Predict on the test or validation split")
    prediction.add_argument("--weights", required=True, type=Path)
    prediction.add_argument("--weights-sha256", help="Expected full SHA-256 of the inference checkpoint")
    prediction.add_argument("--side", choices=("test", "val"), default="test")
    for command in (training, prediction):
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed_all(config["seed"])
    if args.command == "train":
        train(config, args)
    else:
        infer(config, args)


if __name__ == "__main__":
    main()
