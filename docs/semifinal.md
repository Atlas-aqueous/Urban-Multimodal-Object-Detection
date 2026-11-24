# 半决赛方案 · RF1029-M

RF1029-M 是第八届 AIC 算法大赛赛题 2 的半决赛单模型。第 8 轮权重在 1000 张测试图上提交后获得 **53.643**。

## 模型

- 采用 RF-DETR Medium 和 DINOv2 骨干，输入为 `[R, G, B, IR, Depth, DepthValid]`，画布为 1728×960。IR、深度和深度有效位各自标准化。
- 红外和深度分别在 P4 做交叉注意力。对齐到 P4 后，以图像内容比例和有效深度比例对传感器特征池化加权，再按有效比例调整注意力分数。
- 每路以零初始化的标量混合原注意力和有效性修正。模型从复赛单模型 v3 的权重开始微调，新增两个可学习参数。

## 模型权重

权重来自半决赛作品的模型目录。

| 文件 | 用途 | SHA256 |
|---|---|---|
| `RF1029M-ep008.pt` | 半决赛推理 | `12871f74f9dd7300164500e00277e0d4188000d138f706b8c90a5497c43a1d67` |
| `r6-mmv3-cleansplit-best.pt` | 微调初始化 | `4f1db6bc8aa6cbb725b4b0e79c71b2842283aa034137ed3d21ed2482bae6090a` |

推理和训练分别用 `--weights-sha256` 和 `--init-weights-sha256` 接收表中的完整摘要。

## 复现

按 [安装说明](../README.md#installation) 配置环境。RF-DETR 1.3.0 的窗口重排需要按矩形尺寸分别使用高、宽方向的 patch 数，安装脚本会应用 [对应补丁](../environment/rfdetr-1.3.0-rectangular.patch)。以下命令均从 `code/` 执行。

### 正式推理

```bash
python -m scripts.prepare_manifest --data-root /path/to/phase3 --out ../runs/phase3-input
python -m scripts.run_rf1029m infer \
    --manifest ../runs/phase3-input/test-manifest.json \
    --weights /path/to/RF1029M-ep008.pt --out ../runs/rf1029m-phase3
```

三模态图像放在 `visible/`、`infrared/` 和 `depth/` 下，按文件名的 stem 对齐。`raw/` 保存 FP16 推理输出，每图最多 100 框。后处理先用 conf 0.35 和类别独立 NMS 0.55 生成六位小数的 TXT，再按 TXT 第六列保留 conf ≥ 0.40 的整行。最终预测保存在 `predictions/` 和 `result.zip`，无检测图像保留空 TXT。

### 微调与开发集评估

```bash
python -m scripts.prepare_manifest --data-root /path/to/train \
    --labels-root /path/to/revised-labels --out ../runs/rf1029m-train-input
python -m scripts.run_rf1029m train \
    --manifest ../runs/rf1029m-train-input/train-manifest.json \
    --init-weights /path/to/r6-mmv3-cleansplit-best.pt --out ../runs/rf1029m-train
python -m scripts.run_rf1029m infer --side val \
    --manifest ../runs/rf1029m-train-input/train-manifest.json \
    --weights /path/to/RF1029M-ep008.pt --out ../runs/rf1029m-val
python -m src.evaluate_txt --manifest ../runs/rf1029m-train-input/train-manifest.json \
    --predictions ../runs/rf1029m-val/predictions --out ../runs/rf1029m-val/metrics.json --coco
```

[配置](../code/configs/rf1029m-1728x960.yaml) 采用固定 1805/195 划分，训练 8 轮共 904 次更新。batch 2，梯度累积 8，BF16 前向、FP32 匹配与损失。AdamW 对骨干、原模型其余参数和两个新系数分别使用 2e-6、1e-5 和 1e-3，warmup 113 次更新后余弦衰减。训练输出第 4 轮和第 8 轮权重，可直接传给推理入口。

## 结果

| 评测 | 指标 | 结果 |
|---|---|---:|
| 半决赛测试集（1000 张） | 官方成绩 | **53.643** |
| 本地开发集（195 张） | mAP@50:95 / AP50 / AP75，101 点插值 | 54.06 / 79.47 / 57.29 |

半决赛作品在 RTX 4090 上回放了全部 1000 张图。与正式提交相比，按类别匹配到 11,237 个框，匹配框的中位 IoU 为 0.9990，7 个 TXT 字节一致。比赛成绩对应原始提交 ZIP。

项目入口与原交付入口在同一台 RTX 4080 SUPER 上使用原实验 PyTorch 2.5.1 环境分别运行，1000 份原始预测 TXT 和 1000 份后处理 TXT 均逐字节一致。
