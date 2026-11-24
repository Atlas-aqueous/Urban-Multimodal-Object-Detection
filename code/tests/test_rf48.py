import numpy as np
import pytest
import torch
import cv2

pytest.importorskip("pycocotools", reason="Run RF48 tests in .venv-rfdetr")
pytest.importorskip("rfdetr", reason="Run RF48 tests in .venv-rfdetr")

from src.trimodal_data import decode, transform
from src.evaluate_txt import integrate, score_records, coco_map, match


def test_coco_matches_envelope_and_global_cap():
    records = []
    for c in range(12):
        gt = np.array([[c, .2, .3, .1, .1], [c, .7, .6, .05, .05]], float)
        pred = np.array([[c, .2, .3, .1, .1, .9], [c, .2, .3, .1, .1, .8],
                         [c, .7, .6, .05, .05, .7], [c, .5, .5, .1, .1, .2]], float)
        records.append((str(c), gt, pred, 480, 640))
    assert abs(score_records(records)["rule_map"] - coco_map(records)) < 1e-12
    capped = np.repeat(records[0][2][:1], 101, axis=0)
    assert score_records([("cap", records[0][1], capped, 480, 640)], False)["at_max_det"] == 1
    records[0] = ("empty", records[0][1], np.empty((0, 6)), 480, 640)
    assert abs(score_records(records)["rule_map"] - coco_map(records)) < 1e-12


def test_fixed_prefix_rule_ap_does_not_reward_deleting_tail():
    assert integrate(np.array([1]), 2) == pytest.approx(51/101)
    assert integrate(np.r_[1, np.zeros(20)], 2) == pytest.approx(51/101)
    assert integrate(np.array([]), 1) == 0
    gt = np.array([[11, .5, .5, .2, .2], [11, .5, .5, .2, .2]])
    pred = np.array([[11, .5, .5, .2, .2, .5], [0, .5, .5, .2, .2, .5]])
    indices, _ = match(gt, pred)
    assert indices[0, 0] == 1 and indices[1, 0] == -1


def test_json_config_keeps_scientific_numbers(tmp_path):
    from src.trimodal_data import load_config, write_json
    path = tmp_path / "config.json"
    write_json(path, {"lr": 1e-5})
    assert isinstance(load_config(path)["lr"], float)


@pytest.mark.parametrize("extension,dtype", [(".png", np.uint16), (".jpg", np.uint8)])
def test_encoding_and_mask(tmp_path, extension, dtype):
    rgb = np.full((12, 20, 3), [30, 60, 90], np.uint8)
    ir = np.full((12, 20), 100, np.uint8)
    depth = np.full((12, 20), 1000 if dtype == np.uint16 else 100, dtype)
    depth[:, :8] = 0
    paths = {}
    for name, arr, ext in [("visible", rgb, ".png"), ("infrared", ir, ".png"), ("depth", depth, extension)]:
        path = tmp_path / (name + ext)
        assert cv2.imwrite(str(path), arr)
        paths[name] = str(path)
    paths["stem"] = "example"
    native, legacy = decode(paths), decode(paths, "legacy5")
    assert native.shape == (12, 20, 6) and legacy.shape == (12, 20, 5)
    assert np.allclose(native[0, -1, :3], [90/255, 60/255, 30/255])
    assert native[0, -1, 4] == pytest.approx(.05 if dtype == np.uint16 else 100/255, abs=.01)
    assert native[0, 0, 5] == 0 and native[0, -1, 5] == 1
    tensor, mask, labels, geometry = transform(native, 32, np.array([[11, .5, .5, .2, .2]], np.float32))
    assert set(tensor[5].unique().tolist()) <= {0., 1.}
    assert torch.all(tensor[4][tensor[5] == 0] == 0)
    assert tensor.shape == (6, 32, 32) and mask.shape == (32, 32)
    assert not mask.any()  # Fixed canvas coordinates; no centered NestedTensor padding.
    assert geometry[:2] == (12, 20) and labels[0, 0] == 11


def test_fusion_zero_and_gradient():
    from src.trimodal_rfdetr import SensorBranch
    torch.set_num_threads(2)
    branch = SensorBranch(2)
    rgb = torch.randn(2, 256, 2, 2)
    sensor = torch.randn(2, 2, 32, 32)
    assert torch.count_nonzero(branch(sensor, rgb)) == 0
    optimizer = torch.optim.SGD(branch.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss = ((rgb + branch(sensor, rgb))**2).mean()
        loss.backward()
        optimizer.step()
    assert branch.body[0].weight.grad.abs().sum() > 0


def test_box_inverse_and_class_11():
    from src.trimodal_rfdetr import output_rows
    logits = torch.full((2, 12), -30.)
    logits[0, 11] = 10
    boxes = torch.tensor([[.5, .5, .2, .12], [.5, .5, 0., 0.]])
    lines = output_rows(logits, boxes, (12, 20, 19, 32, 6, 0), 32)
    assert len(lines) == 1 and lines[0].startswith("11 ")
    assert float(lines[0].split()[3]) == pytest.approx(.2)


def test_strict_pack_never_creates_missing_files(tmp_path):
    from src.check_submit import validate_and_pack
    pred, images = tmp_path / "pred", tmp_path / "images"
    pred.mkdir()
    images.mkdir()
    (images / "a.png").touch()
    destination = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="missing"):
        validate_and_pack(pred, images, destination, strict=True)
    assert not destination.exists()
    (pred / "a.txt").write_text("11 .5 .5 .1 .1 .9\n")
    result = validate_and_pack(pred, images, destination, strict=True)
    assert result["images"] == 1 and result["created_empty"] == 0
    (pred / "extra.txt").touch()
    with pytest.raises(ValueError, match="extra"):
        validate_and_pack(pred, images, destination, strict=True)


@pytest.mark.parametrize("classes", [[], [0, 11]])
def test_criterion_empty_targets_and_class_endpoints(classes):
    from src.trimodal_rfdetr import build_detector
    config = dict(seed=42, imgsz=128, pretrained="unused-local-placeholder", multimodal=True)
    model, criterion, _ = build_detector(config, initialize=False)
    assert model.class_embed.out_features == criterion.num_classes == 12
    logits = torch.randn(1, 26, 12, requires_grad=True)
    boxes = torch.rand(1, 26, 4, requires_grad=True)
    target = {"labels": torch.tensor(classes, dtype=torch.long),
              "boxes": torch.tensor([[.5, .5, .2, .2]] * len(classes), dtype=torch.float32).reshape(-1, 4)}
    values = criterion({"pred_logits": logits, "pred_boxes": boxes}, [target])
    loss = sum(v * criterion.weight_dict[k] for k, v in values.items() if k in criterion.weight_dict)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
