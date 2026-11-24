import zipfile
from pathlib import Path

import pytest

from src.check_submit import clean_line, validate_and_pack


class TestCleanLine:
    def test_valid_line_passthrough(self):
        assert clean_line([6, 0.5, 0.5, 0.2, 0.3, 0.9]) == [6, 0.5, 0.5, 0.2, 0.3, 0.9]

    def test_illegal_coords_dropped(self):
        assert clean_line([6, 1.2, -0.1, 0.5, 0.5, 0.9]) is None

    def test_illegal_conf_dropped(self):
        assert clean_line([6, 0.5, 0.5, 0.2, 0.3, 1.5]) is None
        assert clean_line([6, 0.5, 0.5, 0.2, 0.3, -0.2]) is None

    def test_nonfinite_and_zero_size_dropped(self):
        assert clean_line([6, "nan", 0.5, 0.2, 0.3, 0.9]) is None
        assert clean_line([6, 0.5, 0.5, 0.0, 0.3, 0.9]) is None

    def test_illegal_class_dropped(self):
        assert clean_line([12, 0.5, 0.5, 0.2, 0.3, 0.9]) is None
        assert clean_line([-1, 0.5, 0.5, 0.2, 0.3, 0.9]) is None

    def test_wrong_field_count_dropped(self):
        assert clean_line([6, 0.5, 0.5, 0.2, 0.9]) is None


class TestValidateAndPack:
    def _setup(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.png").write_bytes(b"x")
        (img_dir / "b.png").write_bytes(b"x")
        pred = tmp_path / "pred"
        pred.mkdir()
        return img_dir, pred

    def test_missing_txt_autocreated_and_packed(self, tmp_path):
        img_dir, pred = self._setup(tmp_path)
        (pred / "a.txt").write_text("6 0.5 0.5 0.2 0.3 0.9\n")  # b 缺失
        stats = validate_and_pack(pred, img_dir, tmp_path / "submit.zip")
        assert stats["created_empty"] == 1
        with zipfile.ZipFile(tmp_path / "submit.zip") as z:
            names = z.namelist()
            assert "a.txt" in names and "b.txt" in names
            assert z.read("b.txt") == b""

    def test_over_100_rows_truncated_by_conf(self, tmp_path):
        img_dir, pred = self._setup(tmp_path)
        lines = [f"6 0.5 0.5 0.1 0.1 {c / 300:.4f}" for c in range(300)]
        (pred / "a.txt").write_text("\n".join(lines) + "\n")
        (pred / "b.txt").write_text("")
        validate_and_pack(pred, img_dir, tmp_path / "s.zip")
        kept = (tmp_path / "staging" / "a.txt").read_text().strip().splitlines()
        assert len(kept) == 100
        confs = [float(l.split()[5]) for l in kept]
        assert confs == sorted(confs, reverse=True)   # 保留的是 conf 最高的100行

    def test_stale_staging_not_packed(self, tmp_path):
        img_dir, pred = self._setup(tmp_path)
        (pred / "a.txt").write_text("6 0.5 0.5 0.2 0.3 0.9\n")
        (pred / "b.txt").write_text("")
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "stale.txt").write_text("6 0.5 0.5 0.2 0.3 0.9\n")  # 上次运行残留
        (staging / "zzz.txt").write_text("6 0.5 0.5 0.2 0.3 0.9\n")     # 无对应图片的残留
        validate_and_pack(pred, img_dir, tmp_path / "s.zip", staging=staging)
        with zipfile.ZipFile(tmp_path / "s.zip") as z:
            names = z.namelist()
        assert "zzz.txt" not in names and "stale.txt" not in names
        assert names == ["a.txt", "b.txt"]  # zip 只含本次图片对应的 TXT

    def test_illegal_rows_dropped_valid_kept(self, tmp_path):
        img_dir, pred = self._setup(tmp_path)
        (pred / "a.txt").write_text(
            "6 0.5 0.5 0.2 0.3 0.9\n"      # 合法
            "12 0.5 0.5 0.2 0.3 0.9\n"     # 非法类别
            "garbage line\n"                # 解析失败
        )
        (pred / "b.txt").write_text("")
        stats = validate_and_pack(pred, img_dir, tmp_path / "s.zip",
                                  staging=tmp_path / "staging")
        assert stats["dropped"] == 2 and stats["kept"] == 1
        parts = (tmp_path / "staging" / "a.txt").read_text().split()
        assert list(map(float, parts)) == [6, 0.5, 0.5, 0.2, 0.3, 0.9]
