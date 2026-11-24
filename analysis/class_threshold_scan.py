"""逐类扫描置信度阈值：区分「根本没检出」与「检出了但置信度被压低」。

若某类在目标域上置信度整体被压低（域偏移的典型表现），把该类的阈值单独调低
就能把已检出但被阈值砍掉的框救回来 —— 这是零成本的后处理手段。

数据来源：conf=0.001 的原始预测（含全部候选框），无需 GPU。
"""
from pathlib import Path

import numpy as np

CL = ["person", "boat", "animal", "seat", "sign", "bicycle",
      "car", "ball", "light", "garbage", "uav", "tricycle"]
THS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
# 训练集里每张图的该类目标密度（用于判断是否漏检）
TRAIN_PER_IMG = np.array([2.775, 0.068, 1.510, 0.321, 0.394, 0.333,
                          0.975, 0.048, 0.917, 0.141, 0.104, 0.013])

D = Path("runs/formal/rf48b/final-run-early1280")
RAW = {"评测集 A": D / "test-1280-raw", "评测集 B": D / "phase2-1280-raw"}


def load(d):
    rows = []
    for f in d.glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append([float(x) for x in line.split()])
    return np.array(rows) if rows else np.zeros((0, 6))


for name, d in RAW.items():
    if not d.exists():
        print(f"[跳过] {d} 不存在")
        continue
    a = load(d)
    n_img = len(list(d.glob("*.txt")))
    print(f"\n===== {name}（{n_img} 图，原始候选 {len(a)} 框）=====")
    hdr = f"{'类别':<9}" + "".join(f"{('>=' + str(t)):>9}" for t in THS) + f"{'训练密度':>10}"
    print(hdr)
    for i, c in enumerate(CL):
        sub = a[a[:, 0] == i]
        cells = "".join(f"{int((sub[:,5] >= t).sum())/n_img:>9.3f}" for t in THS)
        print(f"{c:<9}{cells}{TRAIN_PER_IMG[i]:>10.3f}")

# 对比两套数据的置信度分布（同一模型）
print("\n===== 置信度分布对比（每类的中位置信度）=====")
a1, a2 = load(RAW["评测集 A"]), load(RAW["评测集 B"])
hdr = f"{'类别':<9}{'基线中位conf':>13}{'目标中位conf':>13}{'变化':>10}"
print(hdr)
for i, c in enumerate(CL):
    s1, s2 = a1[a1[:, 0] == i, 5], a2[a2[:, 0] == i, 5]
    m1 = np.median(s1) if len(s1) else np.nan
    m2 = np.median(s2) if len(s2) else np.nan
    print(f"{c:<9}{m1:>13.4f}{m2:>13.4f}{m2-m1:>+10.4f}")
