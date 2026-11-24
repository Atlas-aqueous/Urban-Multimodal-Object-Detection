"""Load tensor-only inference checkpoints and verify their optional SHA-256."""
import hashlib

from packaging.version import Version
import torch


def load_inference_checkpoint(path, expected_sha256=None):
    if Version(torch.__version__) < Version("2.6.0"):
        raise RuntimeError("Inference checkpoint loading requires PyTorch 2.6 or newer.")

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {path}")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=True), actual_sha256
