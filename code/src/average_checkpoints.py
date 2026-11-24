"""对完全同构的 Ultralytics 权重做线性插值，输出仍是单一模型。"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from src.inference_checkpoint import load_legacy_checkpoint


def checkpoint_model(checkpoint: dict):
    model = checkpoint.get("ema")
    if model is None:
        model = checkpoint.get("model")
    if model is None:
        raise ValueError("权重中无 ema/model")
    return model.float()


def main(args: argparse.Namespace) -> Path:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha 必须在 [0, 1] 内")
    left_ckpt = load_legacy_checkpoint(args.left)
    right_ckpt = load_legacy_checkpoint(args.right)
    left = checkpoint_model(left_ckpt)
    right = checkpoint_model(right_ckpt)
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise ValueError("两份权重的 state_dict 键不一致")

    merged = deepcopy(left)
    merged_state = merged.state_dict()
    with torch.no_grad():
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if left_value.shape != right_value.shape:
                raise ValueError(
                    f"形状不一致: {key}: {left_value.shape} != {right_value.shape}"
                )
            if left_value.is_floating_point():
                value = left_value.mul(1.0 - args.alpha).add(
                    right_value, alpha=args.alpha
                )
            else:
                value = left_value
            merged_state[key].copy_(value)
    merged.load_state_dict(merged_state, strict=True)
    merged.args = getattr(left, "args", {})

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": merged.half(),
            "train_args": getattr(left, "args", {}),
            "soup": {
                "left": str(args.left.resolve()),
                "right": str(args.right.resolve()),
                "alpha_right": args.alpha,
            },
        },
        out,
    )
    print(
        f"[soup] left={args.left} right={args.right} "
        f"alpha_right={args.alpha:g} -> {out}"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args())
