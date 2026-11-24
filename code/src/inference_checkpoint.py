"""Load inference artifacts and project training checkpoints."""
import hashlib
import json
from pathlib import Path

from packaging.version import Version
from safetensors.torch import load as load_safetensors
import torch


def load_legacy_checkpoint(source):
    version = Version(torch.__version__)
    if version.is_prerelease or version < Version("2.10") or Version("2.14") <= version < Version("2.15"):
        raise RuntimeError("Legacy checkpoints require PyTorch 2.10–2.13 or 2.15+.")
    return torch.load(source, map_location="cpu", weights_only=True)


def load_inference_checkpoint(path, expected_sha256=None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {path}")
        stream.seek(0)
        if path.suffix.lower() == ".safetensors":
            metadata_bytes = path.with_suffix(".json").read_bytes()
            metadata = json.loads(metadata_bytes)
            checkpoint = {**metadata, "model": load_safetensors(stream.read()),
                          "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest()}
        else:
            checkpoint = load_legacy_checkpoint(stream)
    return checkpoint, actual_sha256
