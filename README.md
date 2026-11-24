# Urban Multimodal Object Detection

面向城市场景的三模态（可见光 / 红外 / 深度）目标检测实现，基于 **RF-DETR** 与 DINOv2 骨干。项目包含第八届 AIC 算法大赛赛题 2 的 RF48 基线，以及半决赛取得 **53.643** 的 RF1029-M。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
<img alt="torch" src="https://img.shields.io/badge/PyTorch-2.x-orange">
<img alt="classes" src="https://img.shields.io/badge/classes-12-green">
<img alt="license" src="https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey">
</p>

项目提供六通道数据编码、融合模型、训练与推理入口，以及评测、后处理和提交格式校验工具。

---

## Highlights

- **六通道早融合** —— `[R, G, B, IR, Depth, DepthValid]` 直接送入补丁嵌入层，
  新增通道零初始化；骨干自第一层即可利用传感器信息。
- **零初始化双路融合** —— 早融合 + P4 门控残差晚融合两条路径，初始化时刻与纯 RGB 模型
  逐元素等价（实测差 `0.000e+00`）。
- **深度质量加权下采样** —— 避免无效零值污染插值，掩码走最近邻、深度走质量加权。
- **有效区域感知注意力** —— RF1029-M 在 P4 聚合传感器特征时，使用真实图像区域和深度有效性对注意力加权。
- **评测口径标定** —— 发现并量化了两种 AP 实现之间 10~13 分的系统性偏差，
  使选型决策从「凭感觉」变为「可标定」。
- **纯后处理的工作点搜索** —— 阈值与 NMS 调整无需重新推理，零 GPU 成本。
- **完整工程配套** —— 自动化测试、数据/标签一致性校验、提交格式校验器。

---

## Architecture

### RF48 基线

```
                    ┌─────────────────────────────────────────┐
   visible ─┐       │  patch_embed  (3ch → 6ch, zero-init)    │
   infrared ─┼─ 6ch ─┤            ↓                            │
   depth ────┤       │  DINOv2 windowed backbone (patch16)     │
   valid ────┘       │            ↓                            │
                     │  projector → P4 features                │
                     │        ↓              ┌──────────────┐  │
                     │  ┌──────────┐   IR ──►│ gated branch │  │
                     │  │ P4 feat  │◄──(+)──┤ (zero-init)  │  │
                     │  └────┬─────┘   Depth►└──────────────┘  │
                     │       ↓                                 │
                     │  DETR decoder (4 layers, 300 queries)   │
                     └───────┬─────────────────────────────────┘
                             ↓
                 boxes + class scores → conf filter → NMS → TXT
```

RF48 的六通道扩展和门控残差在初始化时使模型与 RGB 对照的输出一致。

### RF1029-M 半决赛模型

沿用六通道早融合和共享检测解码器，画布改为 1728×960。IR 和深度分支分别用图像内容掩码和深度有效掩码修正 P4 交叉注意力，每路各学习一个残差系数。训练和推理细节见 [半决赛方案](docs/semifinal.md)。

---

## Results

| 半决赛模型 | 半决赛官方成绩 | 195 张开发集 mAP@50:95（101 点 AP） |
|---|---:|---:|
| RF1029-M，第 8 轮，conf 0.40 / NMS 0.55 | **53.643** | 54.06 |

RF48 在 195 张留出验证集上的结果采用 [RF48 方法](docs/method.md) 第 2.2 节中的 Ultralytics AP 口径。

| 模型 | 画布 | 验证集 mAP |
|---|---|---|
| 晚融合（门控残差） | 1280 | 61.72 |
| **早融合（六通道）** | **1280** | **62.62** |
| 晚融合 | 1536 | 60.59 |
| 早融合 + 全量数据训练 | 1280 | 无留出验证集，选点方式见 `docs/notes/` |

分尺寸 AP 揭示了本任务的核心难点：

| 目标短边 | 目标数 | 占比 | AP |
|---|---|---|---|
| 16–32 px | 196 | 11.6% | **24.8** |
| 32–64 px | 553 | 32.7% | 45.7 |
| 64–128 px | 499 | 29.5% | 51.1 |
| ≥128 px | 430 | 25.4% | **73.5** |

小目标的 AP 只有大目标的三分之一，而短边 <64px 的目标占总数 **45%**。

---

## Repository Layout

```
code/
  src/            数据管线、融合模型、训练、推理、评测口径
  scripts/        实验编排、后处理、结果打包、诊断
  configs/        全部实验配置（YAML）
  tests/          自动化测试
docs/
  method.md       RF48 架构、训练与消融
  semifinal.md    RF1029-M 方法与复现命令
  method.pdf      方法 PDF
  notes/          工程笔记（口径标定、验证噪声、实现记录）
analysis/         离线分析工具（提交校验、阈值分析、口径校准）
environment/      依赖清单与环境脚本
manifests/        数据划分与样本清单
results/          训练曲线与模型身份记录
```

---

## Installation

```bash
git clone https://github.com/Atlas-aqueous/Urban-Multimodal-Object-Detection.git
cd Urban-Multimodal-Object-Detection
bash environment/setup_env.sh
source .venv/bin/activate
```

- Linux、Python 3.12、PyTorch 2.13 / CUDA 12.6
- 半决赛复现环境为 RTX 4090 24 GB。安装脚本同时应用 RF-DETR 1.3.0 的矩形画布修复。
- 历史训练环境见 [环境快照](environment/environment.txt)。

---

## Data Format

本项目不附带数据。数据需按如下约定组织：

```
<data_root>/
  visible/    infrared/    depth/     # 同一 stem 三模态一一对应
  labels/                            # 每行：class_id cx cy w h（归一化）
```

六通道编码 `native6-linear20m-v1`：

| 通道 | 内容 | 归一化 |
|---|---|---|
| 0–2 | R, G, B | ImageNet mean/std |
| 3 | Infrared | 灰度 / 255 |
| 4 | Depth | PNG: `clip(d, 0, 20000)/20000`；JPG: `/255` |
| 5 | DepthValid | 原始深度 `> 0` |

训练标签采用官方修订版。运行 `python -m scripts.prepare_manifest` 可从本机数据生成清单，并按仓库记录恢复 1805/195 划分、校验预处理后的标签。测试清单通过三模态文件名匹配生成。

---

## Usage

### 半决赛 RF1029-M

```bash
cd code
python -m scripts.prepare_manifest --data-root /path/to/phase3 --out ../runs/phase3-input
python -m scripts.run_rf1029m infer \
    --manifest ../runs/phase3-input/test-manifest.json \
    --weights /path/to/RF1029M-ep008.pt --out ../runs/rf1029m
```

输出包括逐图 TXT、提交 ZIP 和运行摘要。微调、开发集评估和权重校验见 [半决赛复现](docs/semifinal.md)。

### RF48 推理

```bash
cd code
python -m scripts.prepare_manifest --data-root /path/to/test --out ../runs/test-input
python -m src.predict_rfdetr \
    --weights /path/to/best-early-ep67.safetensors \
    --sample-manifest ../runs/test-input/test-manifest.json \
    --out-dir ../runs/raw --side test --imgsz 1280 --conf 0.001
python -m scripts.apply_nms --source ../runs/raw --dest ../runs/final --conf 0.35 --nms 0.6
```

预测先按置信度取全局 top-100。基于相同原始预测，可离线调整更高的置信度阈值和 NMS。

### RF48 训练

在 `code/` 目录执行以下命令。链接中的权重是 RF48 的公开 RF-DETR Medium 初始化权重。

```bash
mkdir -p weights/public
curl -fL https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth \
    -o weights/public/rf-detr-medium.pth
python -m scripts.prepare_manifest --data-root /path/to/train \
    --labels-root /path/to/revised-labels --out runs/formal/rf48
python -m scripts.run_experiment \
    --config configs/rf48c-early1280.yaml --split-manifest runs/formal/rf48/split.json \
    --out ../runs/rf48c
```

| 配置 | 说明 |
|---|---|
| `rf48c-early1280.yaml` | 六通道早融合，1280 画布（主配方） |
| `rf48b-1280.yaml` | 门控晚融合，1280 画布 |
| `rf48d-1536.yaml` / `rf48e-1408.yaml` | 分辨率对照 |
| `rf48f-all2000.yaml` | 全量数据训练（无留出验证集） |
| `rf48g-aug.yaml` | 加强数据增强 |

### 测试

```bash
cd code && python -m pytest -q tests/
```

覆盖提交格式校验、几何反算、六通道编码与零初始化等价性。早融合测试使用上一节的 RF-DETR Medium 初始化权重。

---

## Model Weights

RF48 权重通过 Release 附件提供。RF1029-M 推理和微调使用半决赛作品中的权重，文件名及 SHA256 见 [权重说明](docs/semifinal.md#模型权重)。

**下载 → [Releases · v1.0-weights](https://github.com/Atlas-aqueous/urban-multimodal-detection/releases/tag/v1.0-weights)**

| 文件 | 说明 | SHA‑256 |
|---|---|---|
| `best-early-ep67.pt` | 主模型（早融合 1280 第 67 轮） | `34467cda0e5ff1ec39376ba606d59f1515459a8236ebd8ee39408deb9372af32` |
| `epoch-070.pt` | 全量数据训练，第 70 轮 | `5b320664d532e85477b3fc70536952928450e3bf49bbc95dda13e4371e0d1044` |
| `epoch-080.pt` | 全量数据训练，第 80 轮 | `7d5fbd25faddc91b7617d014839691c330051198aebc440132a211ad3185c13c` |
| `epoch-090.pt` | 全量数据训练，第 90 轮 | `460afdacef942490ce2a94cf731dd3e7d0122b56e15320a797cb042e74d874de` |

检查点内部结构：

```python
{
  "model":    state_dict,
  "identity": {"config": {...}, "preprocess": "native6-linear20m-v1", "classes": [...]},
  "epoch":    int,
  "config":   dict,
}
```

用法见上文推理示例。使用 Release 权重时，可通过 `--weights-sha256` 传入表中的完整摘要。

---

## Engineering Notes

以下笔记记录了 RF48 阶段的实验结果与工程细节。

| 文档 | 内容 |
|---|---|
| `docs/notes/2026-09-19-metric-arbitration.md` | 两种 AP 实现的系统性偏差与标定方法 |
| `docs/notes/2026-09-19-validation-noise.md` | 小验证集下的噪声量化与配对自助法 |
| `docs/notes/2026-09-16-rf48-implementation.md` | 融合结构实现与零初始化等价性验证 |
| `docs/notes/2026-09-04-reproduction-audit.md` | 数据与标签一致性审计 |

### RF48 消融

| 方向 | 结果 |
|---|---|
| **切块推理**（2×2，放大 1.75×） | mAP 62.55 → 42.76 |
| **提高推理分辨率** 1408 / 1536 / 1664 | 预测与 1280 版重合 91.5~92.7% |
| **原生 1536 训练** | 验证集 60.59 vs 1280 的 61.58 |
| **深度编码改对数** | 线性编码 5–95% 跨度 0.759，对数仅 0.207，线性更优 |
| **增加 P3 检测层** | `num_feature_levels` 1→2 改变 `cross_attn` 权重形状 |

[compare_submissions.py](analysis/compare_submissions.py) 可以按框比较两组提交的重合度。

---

## Data & License

- 本项目**不含任何数据**。使用数据集时请遵守其原始授权条款。
- 未使用外部数据；未使用多模型集成、权重平均或 EMA。
- 第三方组件：
  - RF-DETR (Roboflow) — Apache-2.0
  - DINOv2 (Meta AI) — 见模型卡
  - pycocotools — BSD
- 本项目未附许可证文件，默认保留所有权利。

---

## Acknowledgements

RF-DETR 与 DINOv2 作者提供的开源实现与预训练权重。
