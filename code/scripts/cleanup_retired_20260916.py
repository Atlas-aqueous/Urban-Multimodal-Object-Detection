"""Allowlisted cleanup authorized on 2026-09-16; defaults to dry-run.

Only failed E11 interpolations, completed/stripped last.pt checkpoints, and old
18-image sample derivatives are removed. Best checkpoints and all records stay.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/formal"
MANIFEST = ROOT / "docs/reports/2026-09-16-cleanup.json"
FINISHED = ["e9-yolov8l-p2-1280-aggressive-b4",
            "e10-yolov8l-p3-960-aggressive-b8",
            "e12-yolov8x-p2-1280-aggressive-b4"]
SAMPLES = ["repro-3ch", "repro-5ch", "improved-3ch", "improved-5ch"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def record(path, reason):
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise ValueError(f"Not a regular explicit local file: {path}")
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest(path), "reason": reason}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()).get("status") == "complete":
        print("Cleanup already completed; manifest retained.")
        return
    e11 = json.loads((RUNS / "e11-soup-state.json").read_text())
    assert e11["stage"] == "complete_no_full_search" and e11["gain_over_baseline"] < 0
    targets = []
    for suffix in ["025", "050", "075"]:
        targets.append(record(RUNS / f"weight-soups/e0-e1-a{suffix}.pt",
                              "E11 retired after negative validation probe; source weights preserved"))
    import torch
    for name in FINISHED:
        run = RUNS / name
        assert (run / "experiment.json").is_file() and (run / "weights/best.pt").is_file()
        last = run / "weights/last.pt"
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        assert checkpoint.get("epoch") == -1 and checkpoint.get("optimizer") is None
        del checkpoint
        targets.append(record(last, "Training complete, inferior final epoch, optimizer already stripped; best.pt retained"))
    for name in SAMPLES:
        image_root = ROOT / "datasets" / name / "images"
        files = sorted(p for p in image_root.rglob("*") if p.is_file())
        assert len(files) == 18
        for path in files:
            assert path.suffix in {".tiff", ".png"}
            for modality in ["visible", "infrared", "depth"]:
                assert any((ROOT / "data" / modality).glob(path.stem + ".*"))
            targets.append(record(path, "Obsolete sample preprocessing; original 18 triplets, labels/configs and reports retained"))
    protected = set(RUNS.glob("*/weights/best.pt")) | set((RUNS / "submissions").glob("*"))
    protected |= {p for p in RUNS.rglob("*") if p.is_file() and p.suffix in {".json", ".yaml", ".csv", ".md"}}
    guards = {str(p.relative_to(ROOT)): digest(p) for p in sorted(protected) if p.is_file()}
    assert not set(guards) & {r["path"] for r in targets}
    payload = {"status": "planned", "at_utc": datetime.now(timezone.utc).isoformat(),
               "files": targets, "bytes_selected": sum(r["bytes"] for r in targets),
               "protected_sha256": guards, "free_bytes_before": shutil.disk_usage(ROOT).free,
               "recovery": "No backup of removed files. Sample derivatives can be regenerated; exact last-epoch weights need retraining. E11 can be regenerated from retained E0/E1."}
    print("Selected", len(targets), "files,", round(payload["bytes_selected"] / 1024**3, 4), "GiB")
    for row in targets[:6]:
        print(row["path"], row["reason"])
    if not args.apply:
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    # Delete individually: never traverse a source-data symlink or remove a directory tree.
    for row in targets:
        path = ROOT / row["path"]
        assert not path.is_symlink() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
        path.unlink()
    for rel, expected in guards.items():
        assert digest(ROOT / rel) == expected, f"Protected artifact changed: {rel}"
    payload.update(status="complete", free_bytes_after=shutil.disk_usage(ROOT).free,
                   protected_artifacts_verified=len(guards))
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("Removed allowlisted files;", len(guards), "protected artifacts unchanged.")


if __name__ == "__main__":
    main()
