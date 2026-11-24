"""比较两个提交包的预测重合度：判断换一个包提交是否可能改变分数。

若两份预测高度重合（同类别 IoU>=0.5 匹配率高、置信度接近），
则它们的官方分几乎必然相同，用它占一个提交名额几乎没有信息量。
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np

CL = ["person", "boat", "animal", "seat", "sign", "bicycle",
      "car", "ball", "light", "garbage", "uav", "tricycle"]


def load(zp):
    out = {}
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if not n.endswith(".txt"):
                continue
            rows = []
            for line in z.read(n).decode("utf-8", "replace").splitlines():
                if line.strip():
                    rows.append([float(x) for x in line.split()])
            out[Path(n).stem] = np.array(rows) if rows else np.zeros((0, 6))
    return out


def xyxy(a):
    cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def iou_matrix(A, B):
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    a, b = xyxy(A), xyxy(B)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="基准包（我们已提交的）")
    ap.add_argument("--b", required=True, help="待比较包")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    stems = sorted(set(A) | set(B))
    n_a = sum(len(v) for v in A.values())
    n_b = sum(len(v) for v in B.values())
    print(f"A(基准) {len(A)} 图 / {n_a} 框")
    print(f"B(对照) {len(B)} 图 / {n_b} 框")
    print()

    matched_a = matched_b = 0
    per_class_a = np.zeros(12, int)
    per_class_ma = np.zeros(12, int)
    per_class_b = np.zeros(12, int)
    per_class_mb = np.zeros(12, int)
    conf_pairs = []
    exact_same = 0
    for s in stems:
        a, b = A.get(s, np.zeros((0, 6))), B.get(s, np.zeros((0, 6)))
        if len(a) and len(b) and np.array_equal(a, b):
            exact_same += 1
        M = iou_matrix(a, b)
        used_b = set()
        for i in range(len(a)):
            ci = int(a[i, 0])
            per_class_a[ci] += 1
            best, bj = 0.0, -1
            for j in range(len(b)):
                if j in used_b or int(b[j, 0]) != ci:
                    continue
                if M[i, j] > best:
                    best, bj = M[i, j], j
            if best >= args.iou:
                used_b.add(bj)
                matched_a += 1
                per_class_ma[ci] += 1
                conf_pairs.append((a[i, 5], b[bj, 5]))
        for j in range(len(b)):
            cj = int(b[j, 0])
            per_class_b[cj] += 1
            if j in used_b:
                per_class_mb[cj] += 1
                matched_b += 1

    print("── 整体重合度 ──")
    print(f"  A 的框在 B 中被匹配上 (IoU>={args.iou}, 同类): {matched_a}/{n_a} = {matched_a/n_a*100:.2f}%")
    print(f"  B 的框在 A 中被匹配上 (IoU>={args.iou}, 同类): {matched_b}/{n_b} = {matched_b/n_b*100:.2f}%")
    print(f"  逐图逐字节完全相同的图: {exact_same}/{len(stems)}")
    print(f"  只在一方出现的框数差异 : {abs(n_a-n_b)}")
    print()
    print("── 逐类匹配率（A→B）──")
    print(f"{'类别':<10}{'A框数':>8}{'B框数':>8}{'A被匹配':>9}{'匹配率':>9}")
    for i, c in enumerate(CL):
        r = per_class_ma[i] / per_class_a[i] * 100 if per_class_a[i] else float("nan")
        print(f"{c:<10}{per_class_a[i]:>8}{per_class_b[i]:>8}{per_class_ma[i]:>9}{r:>8.1f}%")
    if conf_pairs:
        ca = np.array([p[0] for p in conf_pairs])
        cb = np.array([p[1] for p in conf_pairs])
        print()
        print("── 匹配框的置信度一致性 ──")
        print(f"  相关系数 r = {np.corrcoef(ca, cb)[0,1]:.4f}")
        print(f"  平均绝对差 = {np.abs(ca-cb).mean():.4f}   中位绝对差 = {np.median(np.abs(ca-cb)):.4f}")


if __name__ == "__main__":
    main()
