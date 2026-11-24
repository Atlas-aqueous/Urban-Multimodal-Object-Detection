"""Build a local sample manifest from aligned RGB, infrared and depth folders."""
import argparse
import json
from pathlib import Path

from src.prepare_data import sanitize_label_file
from src.trimodal_data import make_manifest, write_json

REFERENCE = Path(__file__).resolve().parents[2] / "manifests/train-manifest.json"


def prepare(data_root, out, labels_root=None, reference=REFERENCE):
    indexed = make_manifest(data_root)
    if labels_root is None:
        out.mkdir(parents=True, exist_ok=False)
        path = out / "test-manifest.json"
        write_json(path, indexed)
        return path

    recorded = json.loads(reference.read_text(encoding="utf-8"))
    expected = {row["stem"]: row for row in recorded["samples"]}
    if {row["stem"] for row in indexed["samples"]} != expected.keys():
        raise ValueError("The image stems must match the reference training manifest")
    if {path.stem for path in labels_root.glob("*.txt")} != expected.keys():
        raise ValueError("The revised label stems must match the reference training manifest")

    out.mkdir(parents=True, exist_ok=False)
    split = out / "split.json"
    write_json(split, recorded["split_manifest"])
    labels = out / "labels"
    for side in ("train", "val"):
        (labels / side).mkdir(parents=True)
    for stem, row in expected.items():
        filename = stem + ".txt"
        sanitize_label_file(labels_root / filename, labels / row["split"] / filename)

    manifest = make_manifest(data_root, split, labels)
    for row in manifest["samples"]:
        if row["label_sha256"] != expected[row["stem"]]["label_sha256"]:
            raise ValueError(f"Revised label checksum differs: {row['stem']}")
    path = out / "train-manifest.json"
    write_json(path, manifest)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels-root", type=Path, help="Official revised training labels")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    print(prepare(args.data_root, args.out, args.labels_root, args.reference))
