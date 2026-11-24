"""合成微型数据集测试预处理。只依赖 numpy/opencv/pyyaml，无需 torch。"""
import cv2
import numpy as np
import pytest
import yaml
from pathlib import Path

from src.prepare_data import (
    build_3ch, build_5ch, build_6ch, load_depth_u8, load_depth_u8_with_validity,
    load_ir_u8, refresh_rare_train_list, sanitize_label_file, scene_key,
    split_by_scene, main,
)


def _make_tiny_dataset(root: Path):
    """2个场景×2帧。depth用uint16单通道，ir为3通道同值+偏移，vis为渐变BGR。"""
    for scene, frames in [("000001", ["000001_001_00000001", "000001_002_00000002"]),
                          ("000002", ["000002_001_00000003", "000002_002_00000004"])]:
        for i, stem in enumerate(frames):
            (root / "visible").mkdir(parents=True, exist_ok=True)
            (root / "infrared").mkdir(parents=True, exist_ok=True)
            (root / "depth").mkdir(parents=True, exist_ok=True)
            (root / "labels").mkdir(parents=True, exist_ok=True)
            h, w = 32, 48
            vis = np.zeros((h, w, 3), np.uint8)
            vis[:, :, 0] = np.linspace(10, 200, w, dtype=np.uint8)[None, :]  # B
            vis[:, :, 1] = 60
            vis[:, :, 2] = np.linspace(50, 250, w, dtype=np.uint8)[None, :]  # R
            cv2.imwrite(str(root / "visible" / f"{stem}.png"), vis)
            ir = np.zeros((h, w, 3), np.uint8)
            ir[:, :, 0] = 100
            ir[:, :, 1] = 100 + i
            ir[:, :, 2] = 100 + 2 * i
            cv2.imwrite(str(root / "infrared" / f"{stem}.png"), ir)
            dep = np.zeros((h, w), np.uint16)
            dep[:, :] = 10000 + i * 500  # 10m 附近
            cv2.imwrite(str(root / "depth" / f"{stem}.png"), dep)
            (root / "labels" / f"{stem}.txt").write_text(
                "6 0.5 0.5 0.2 0.3\n0 0.2 0.2 0.1 0.1\n")


class TestSceneSplit:
    def test_scene_key(self):
        assert scene_key("000005_011_00000001") == "000005_011"
        assert scene_key("000016") == "000016"
        assert scene_key("000014_002_suppl_00000167") == "000014_002"
        assert scene_key("shuming_343_00000288") == "shuming_343"
        assert scene_key("hehe_268_000002_058_00000051") == "hehe_000002_058"

    def test_split_no_scene_leak(self):
        stems = [f"{s:06d}_{i:03d}_000000{i:02d}" for s in range(1, 21) for i in range(3)]
        train, val = split_by_scene(stems, val_ratio=0.2, seed=42)
        tg = {scene_key(s) for s in train}
        vg = {scene_key(s) for s in val}
        assert tg and vg and not (tg & vg)          # 场景不重叠
        assert set(train) | set(val) == set(stems)  # 全覆盖
        assert len(val) / len(stems) == pytest.approx(0.2, abs=0.17)


class TestChannelOps:
    def test_sanitize_label_file_clips_box_geometry(self, tmp_path):
        src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
        src.write_text("0 1.00339 0.244907 0.0203125 0.0583333\n", encoding="utf-8")
        assert sanitize_label_file(src, dst) == 1
        cls, cx, cy, width, height = map(float, dst.read_text().split())
        assert cls == 0
        assert 0 <= cx <= 1 and 0 <= cy <= 1
        assert 0 < width <= 1 and 0 < height <= 1
        assert cx + width / 2 <= 1

    def test_sanitize_label_file_preserves_valid_edge_box(self, tmp_path):
        src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
        src.write_text("0 0.02 0.5 0.1 0.2\n", encoding="utf-8")
        assert sanitize_label_file(src, dst) == 0
        assert list(map(float, dst.read_text().split())) == [0, 0.02, 0.5, 0.1, 0.2]

    def test_ir_mean_of_three_channels(self, tmp_path):
        f = tmp_path / "ir.png"
        ir = np.zeros((4, 4, 3), np.uint8)
        ir[..., 0], ir[..., 1], ir[..., 2] = 90, 100, 111
        cv2.imwrite(str(f), ir)
        out = load_ir_u8(f)
        assert out.shape == (4, 4)
        assert out[0, 0] in (100, 101)  # round((90+100+111)/3)=100.33→100

    def test_depth_uint16_normalized(self, tmp_path):
        f = tmp_path / "d.png"
        dep = np.zeros((4, 4), np.uint16)
        dep[:, :] = 10000                 # 10m → 10000/20000*255=127.5→128
        dep[0, 0] = 30000                 # 超量程 → clip 20000 → 255
        cv2.imwrite(str(f), dep)
        out = load_depth_u8(f)
        assert out.dtype == np.uint8
        assert out[1, 1] == 128
        assert out[0, 0] == 255

    def test_depth_uint8_passthrough_and_3ch(self, tmp_path):
        f = tmp_path / "d.jpg"
        dep = np.zeros((4, 4, 3), np.uint8)
        dep[:, :] = 77
        cv2.imwrite(str(f), dep)
        out = load_depth_u8(f)
        assert out.shape == (4, 4) and out[0, 0] == 77

    def test_build_3ch_and_5ch_layout(self):
        vis = np.zeros((4, 4, 3), np.uint8)
        vis[:, 0, 0], vis[:, 0, 1], vis[:, 0, 2] = 11, 22, 33   # B G R
        ir = np.full((4, 4), 44, np.uint8)
        dep = np.full((4, 4), 55, np.uint8)
        three = build_3ch(vis, ir, dep)
        five = build_5ch(vis, ir, dep)
        assert three.shape == (4, 4, 3)
        assert list(three[0, 0]) == [55, 44, 33]                # 磁盘 D,IR,R；加载器翻为 R,IR,D
        assert five.shape == (4, 4, 5)
        assert list(five[0, 0]) == [33, 22, 11, 44, 55]         # R G B IR D

    def test_depth_validity_and_6ch_layout(self, tmp_path):
        path = tmp_path / "depth.png"
        raw = np.array([[0, 10000], [20000, 30000]], dtype=np.uint16)
        cv2.imwrite(str(path), raw)
        dep, valid = load_depth_u8_with_validity(path)
        assert dep.tolist() == [[0, 128], [255, 255]]
        assert valid.tolist() == [[0, 255], [255, 255]]
        vis = np.zeros((2, 2, 3), np.uint8)
        six = build_6ch(vis, np.ones((2, 2), np.uint8), dep, valid)
        assert six.shape == (2, 2, 6)
        assert list(six[0, 0, 4:]) == [0, 0]


class TestPipeline:
    @pytest.fixture()
    def tiny(self, tmp_path):
        _make_tiny_dataset(tmp_path / "raw")
        return tmp_path / "raw"

    def test_mode_3ch_outputs(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "ds3"
        args = argparse.Namespace(data_root=tiny, out=out, mode="3ch",
                                  no_labels=False, val_ratio=0.5, seed=42)
        yaml_path = main(args)
        cfg = yaml.safe_load(yaml_path.read_text())
        assert "channels" not in cfg
        assert cfg["names"][11] == "tricycle" and len(cfg["names"]) == 12
        imgs = list((out / "images" / "train").glob("*.png")) + \
               list((out / "images" / "val").glob("*.png"))
        assert len(imgs) == 4
        for p in imgs:
            assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[2] == 3
        # 每张图都有同名标签副本
        for p in imgs:
            lbl = out / "labels" / p.parent.name / f"{p.stem}.txt"
            assert lbl.exists() and lbl.read_text().startswith("6")

    def test_mode_5ch_tiff_roundtrip(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "ds5"
        args = argparse.Namespace(data_root=tiny, out=out, mode="5ch",
                                  no_labels=False, val_ratio=0.5, seed=42)
        main(args)
        tif = next((out / "images" / "train").glob("*.tiff"))
        ok, frames = cv2.imreadmulti(str(tif))
        assert ok and len(frames) == 5            # 多页TIFF：每通道一页
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        assert cfg["channels"] == 5

    def test_mode_6ch_and_rare_repeat(self, tiny, tmp_path):
        import argparse
        # 一个 ball 样本、一个 tricycle 样本；同图规则由最大倍数决定。
        label_files = sorted((tiny / "labels").glob("*.txt"))
        label_files[0].write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        label_files[1].write_text("11 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        out = tmp_path / "ds6"
        args = argparse.Namespace(data_root=tiny, out=out, mode="6ch",
                                  no_labels=False, val_ratio=0.5, seed=42,
                                  rare_repeat=True)
        main(args)
        tif = next(out.glob("images/*/*.tiff"))
        ok, frames = cv2.imreadmulti(str(tif))
        assert ok and len(frames) == 6
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        assert cfg["channels"] == 6 and cfg["train"] == "train_weighted.txt"
        split = __import__("json").loads((out / "split.json").read_text())
        assert split["repeat_stats"]["weighted_samples"] >= len(split["splits"]["train"])

    def test_refresh_rare_list_does_not_rebuild_or_repeat_val(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "ds5-refresh"
        args = argparse.Namespace(data_root=tiny, out=out, mode="5ch",
                                  no_labels=False, val_ratio=0.5, seed=42)
        main(args)
        image_mtimes = {p: p.stat().st_mtime_ns for p in out.glob("images/*/*.tiff")}
        split_before = __import__("json").loads((out / "split.json").read_text())
        train_stem = split_before["splits"]["train"][0]
        (tiny / "labels" / f"{train_stem}.txt").write_text(
            "11 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
        refresh_rare_train_list(out, tiny)
        rows = (out / "train_weighted.txt").read_text().splitlines()
        assert rows.count(f"./images/train/{train_stem}.tiff") == 3
        assert not any("/val/" in row for row in rows)
        assert image_mtimes == {p: p.stat().st_mtime_ns for p in out.glob("images/*/*.tiff")}
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        split_after = __import__("json").loads((out / "split.json").read_text())
        assert cfg["train"] == "train_weighted.txt"
        assert split_after["splits"]["val"] == split_before["splits"]["val"]

    def test_scene_split_applied(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "ds3b"
        args = argparse.Namespace(data_root=tiny, out=out, mode="3ch",
                                  no_labels=False, val_ratio=0.5, seed=42)
        main(args)
        tr = {scene_key(p.stem) for p in (out / "images" / "train").glob("*")}
        va = {scene_key(p.stem) for p in (out / "images" / "val").glob("*")}
        assert not (tr & va)                       # 同一序列不会跨两侧

    def test_no_labels_mode(self, tiny, tmp_path):
        import argparse
        # 删除labels后走 --no-labels（模拟测试集）
        import shutil
        shutil.rmtree(tiny / "labels")
        out = tmp_path / "dst"
        (tiny / "visible" / ".DS_Store").write_bytes(b"metadata")
        args = argparse.Namespace(data_root=tiny, out=out, mode="5ch",
                                  no_labels=True, val_ratio=0.5, seed=42)
        main(args)
        assert (out / "images" / "test").exists()
        assert len(list((out / "images" / "test").glob("*.tiff"))) == 4

    def test_all_train_mode_has_null_val(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "all"
        args = argparse.Namespace(data_root=tiny, out=out, mode="5ch",
                                  no_labels=False, all_train=True, val_ratio=0.1,
                                  seed=42)
        main(args)
        cfg = yaml.safe_load((out / "data.yaml").read_text())
        assert cfg["val"] is None
        assert len(list((out / "images" / "train").glob("*.tiff"))) == 4

    def test_nonempty_output_requires_overwrite(self, tiny, tmp_path):
        import argparse
        out = tmp_path / "existing"
        out.mkdir()
        (out / "unrelated.txt").write_text("keep")
        args = argparse.Namespace(data_root=tiny, out=out, mode="3ch",
                                  no_labels=False, val_ratio=0.5, seed=42, overwrite=False)
        with pytest.raises(FileExistsError):
            main(args)
