"""在已知真值的 A 验证集上校准「密度匹配」规则，再据此推断目标域的合适阈值。

逻辑：AP 是对整条 PR 曲线积分，阈值本质上是在截断曲线。合适的工作点应当让
「预测密度」与「真实目标密度」相当。A 验证集上我们知道 GT，可以验证这条规则
是否与实测最优阈值一致；一致则可外推到没有真值的评测集。
"""
import json
from pathlib import Path

import numpy as np

CL = ["person", "boat", "animal", "seat", "sign", "bicycle",
      "car", "ball", "light", "garbage", "uav", "tricycle"]
THS = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

import sys as _s
VAL_RAW = Path(_s.argv[1] if len(_s.argv) > 1 else "runs/formal/rf48b/final-run-early1280/val-1280")
MANIFEST = "runs/formal/rf48/train-manifest.json"

m = json.load(open(MANIFEST))
samples = [s for s in m["samples"] if s.get("split") == "val"]
gt = np.zeros(12)
for s in samples:
    for line in Path(s["label"]).read_text().splitlines():
        if line.strip():
            gt[int(float(line.split()[0]))] += 1
n_val = len(samples)

rows = []
for f in VAL_RAW.glob("*.txt"):
    for line in f.read_text().splitlines():
        if line.strip():
            rows.append([float(x) for x in line.split()])
a = np.array(rows)
print(f"A 验证集 {n_val} 图，GT {int(gt.sum())} 框（{gt.sum()/n_val:.3f} 框/图），"
      f"原始候选 {len(a)} 框")
print()
hdr = (f"{'阈值':>6}{'总密度':>9}{'总GT密度':>10}{'比值':>8}"
       f"{'person密度':>12}{'personGT':>10}")
print(hdr)
print("-" * len(hdr))
for t in [0.001] + THS:
    sub = a[a[:, 5] >= t]
    print(f"{t:>6}{len(sub)/n_val:>9.3f}{gt.sum()/n_val:>10.3f}"
          f"{(len(sub)/n_val)/(gt.sum()/n_val):>8.2f}"
          f"{(sub[:,0]==0).sum()/n_val:>12.3f}{gt[0]/n_val:>10.3f}")

print()
print("逐类：predicted 密度首次跌破 GT 密度的阈值区间")
print(f"{'类别':<9}{'GT密度':>9}  " + "".join(f"{t:>8}" for t in THS))
for i, c in enumerate(CL):
    cells = "".join(f"{(a[(a[:,0]==i)&(a[:,5]>=t)]).shape[0]/n_val:>8.3f}" for t in THS)
    print(f"{c:<9}{gt[i]/n_val:>9.3f}  {cells}")
