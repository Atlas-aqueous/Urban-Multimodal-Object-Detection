"""快速领域体检：只取少量样本，先看深度值域与模态配准这两个最要命的指标。"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = "data"
P1 = f"{ROOT}/数据集A-面向城市场景的多模态目标检测"
P2 = f"{ROOT}/评测集 B-面向城市场景的多模态目标检测"
SETS = [("训练集", f"{P1}/训练集"), ("评测集 A", f"{P1}/测试集"), ("评测集 B", P2)]
N = 24


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def find(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = Path(d) / f"{stem}{ext}"
        if p.exists():
            return p
    h = list(Path(d).glob(f"{stem}.*"))
    return h[0] if h else None


def gray(a):
    a = a.astype(np.float32)
    return a[:, :, :3].mean(2) if a.ndim == 3 else a


for name, root in SETS:
    vis_dir = Path(root) / "visible"
    stems = sorted(p.stem for p in vis_dir.iterdir()
                   if not p.name.startswith("."))[:N]
    print(f"\n===== {name}（抽样 {len(stems)}）=====", flush=True)
    dm, dmx, dzero, dtypes = [], [], [], set()
    c_ir, c_dp, vmean, vend = [], [], [], []
    for s in stems:
        fv, fi, fd = (find(Path(root) / "visible", s),
                      find(Path(root) / "infrared", s),
                      find(Path(root) / "depth", s))
        if not (fv and fi and fd):
            continue
        v, i, d = rd(fv), rd(fi), rd(fd)
        dtypes.add(f"{fd.suffix}{d.dtype}")
        gv, gi = gray(v), gray(i)
        dm.append(float(d.mean()))
        dmx.append(float(d.max()))
        dzero.append(float((d == 0).mean()))
        vmean.append(float(gv.mean()))
        vend.append(float(np.percentile(gv, 98)))
        if gv.shape == gi.shape and gv.std() > 1 and gi.std() > 1:
            c_ir.append(float(np.corrcoef(gv.ravel(), gi.ravel())[0, 1]))
        gd = d.astype(np.float32)
        if gv.shape == gd.shape and gd.std() > 1:
            c_dp.append(float(np.corrcoef(gv.ravel(), gd.ravel())[0, 1]))
    print(f"  深度格式/dtype : {sorted(dtypes)}")
    print(f"  深度 均值={np.median(dm):9.1f}  最大值={np.median(dmx):9.1f}  "
          f"零值占比={np.median(dzero):.4f}")
    print(f"  可见光 亮度均值={np.median(vmean):7.2f}  p98={np.median(vend):7.2f}")
    print(f"  可见光↔红外 相关 = {np.median(c_ir):.4f}  (n={len(c_ir)})")
    print(f"  可见光↔深度 相关 = {np.median(c_dp):.4f}  (n={len(c_dp)})")
