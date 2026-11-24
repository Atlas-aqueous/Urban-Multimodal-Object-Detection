import json
from pathlib import Path

import torch

from src.make_5ch_weights import surgery_conv1
from src.make_splits import stratified_group_split
from src.search_inference import select_best
from src.train import apply_modality_dropout, data_provenance
from src.experiment_gate import cross_split_gate, screen_gate
from src.diagnostics import size_recall
from src.report_experiments import best_epoch, markdown


def test_surgery_supports_six_channels():
    source = torch.arange(8 * 3 * 3 * 3, dtype=torch.float32).reshape(8, 3, 3, 3)
    target = surgery_conv1(source, 6)
    assert target.shape == (8, 6, 3, 3)
    assert torch.equal(target[:, :3], source)
    assert torch.allclose(target[:, 3], source.mean(1))


def test_six_channel_dropout_groups_depth_and_mask(monkeypatch):
    image = torch.ones(2, 6, 2, 2)
    draws = iter([torch.tensor([0.8, 0.2]), torch.tensor([0.2, 0.8])])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(draws))
    apply_modality_dropout(image, 0.5)
    assert torch.all(image[1, 3] == 0) and torch.all(image[0, 3] == 1)
    assert torch.all(image[0, 4:6] == 0) and torch.all(image[1, 4:6] == 1)


def test_stratified_split_has_all_classes_and_no_group_leak(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # 每类放在两个不同场景，保证两侧都可覆盖。
    for cls in range(12):
        for copy in range(2):
            stem = f"{cls:06d}_{copy:03d}_00000001"
            (labels / f"{stem}.txt").write_text(
                f"{cls} 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
    result = stratified_group_split(labels, val_ratio=0.5, seed=3407)
    assert result["counts"]["all"] == 24
    assert all(result["class_image_counts"][side][str(c)] > 0
               for side in ("train", "val") for c in range(12))


def test_inference_tie_prefers_lower_resolution():
    rows = [
        {"imgsz": 1280, "elapsed_seconds": 1.0, "map50_95": 0.5000},
        {"imgsz": 960, "elapsed_seconds": 2.0, "map50_95": 0.4985},
    ]
    assert select_best(rows)["imgsz"] == 960


def test_inference_same_resolution_prefers_accuracy():
    rows = [
        {"imgsz": 960, "elapsed_seconds": 1.0, "map50_95": 0.4985},
        {"imgsz": 960, "elapsed_seconds": 2.0, "map50_95": 0.5000},
    ]
    assert select_best(rows)["map50_95"] == 0.5000


def test_experiment_gates():
    baseline = {"map50_95": 0.50,
                "per_class_ap50_95": {"ball": 0.20, "tricycle": 0.20}}
    total_winner = {"map50_95": 0.506,
                    "per_class_ap50_95": {"ball": 0.20, "tricycle": 0.20}}
    rare_winner = {"map50_95": 0.499,
                   "per_class_ap50_95": {"ball": 0.24, "tricycle": 0.24}}
    assert screen_gate(baseline, total_winner)["passed"]
    assert screen_gate(baseline, rare_winner)["passed"]
    assert cross_split_gate([0.02, 0.016, 0.01])["passed"]
    assert not cross_split_gate([0.03, 0.02, -0.01])["passed"]


def test_size_recall_and_prediction_cap(tmp_path):
    labels, predictions = tmp_path / "labels", tmp_path / "predictions"
    labels.mkdir(); predictions.mkdir()
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.02 0.02\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    rows = ["0 0.5 0.5 0.02 0.02 0.9", "1 0.1 0.1 0.1 0.1 0.8"]
    rows.extend(["2 0.5 0.5 0.01 0.01 0.1"] * 98)
    (predictions / "a.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = size_recall(labels, predictions)
    assert result["recall"]["tiny"] == 1.0
    assert result["recall"]["large"] == 0.0
    assert result["at_100_predictions"] == 1


def test_data_provenance_embeds_split_manifest(tmp_path):
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    split = {"splits": {"train": ["a"], "val": ["b"]}}
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    result = data_provenance(tmp_path / "data.yaml")
    assert result["data_yaml_sha256"]
    assert result["split_manifest"]["sha256"]
    assert result["split_manifest"]["content"] == split


def test_experiment_report_helpers(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,metrics/mAP50-95(B)\n1,0.2\n2,0.3\n", encoding="utf-8"
    )
    assert best_epoch(csv_path) == 2
    text = markdown([{
        "id": "E1", "map50_95": 0.5, "map50": 0.7, "map75": 0.5,
        "per_class_ap50_95": {"ball": 0.2, "tricycle": 0.3},
        "tiny_recall_iou50": 0.6, "at_100_predictions": 1,
        "elapsed_seconds": 3600, "best_epoch": 2,
        "peak_gpu_memory_gib": 12.0,
    }])
    assert "E1" in text and "0.5000" in text
