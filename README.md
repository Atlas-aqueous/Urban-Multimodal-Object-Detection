# Urban Multimodal Object Detection

面向城市场景的三模态（可见光 / 红外 / 深度）目标检测实现，基于 **RF-DETR** 与 DINOv2 骨干。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
<img alt="torch" src="https://img.shields.io/badge/PyTorch-2.x-orange">
<img alt="classes" src="https://img.shields.io/badge/classes-12-green">
<img alt="license" src="https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey">
</p>

本项目提供一套完整的三模态检测工程：从六通道数据编码、双路融合结构、训练与推理管线，
到**评测口径标定**、离线后处理优化与提交格式校验工具。所有结构改动均保证与纯 RGB
预训练模型在初始化时刻逐元素等价，因此完整保留预训练能力。

---

## Highlights

- **六通道早融合** —— `[R, G, B, IR, Depth, DepthValid]` 直接送入补丁嵌入层，
  新增通道零初始化；骨干自第一层即可利用传感器信息。
- **零初始化双路融合** —— 早融合 + P4 门控残差晚融合两条路径，初始化时刻与纯 RGB 模型
  逐元素等价（实测差 `0.000e+00`）。
- **深度质量加权下采样** —— 避免无效零值污染插值，掩码走最近邻、深度走质量加权。
- **评测口径标定** —— 发现并量化了两种 AP 实现之间 10~13 分的系统性偏差，
  使选型决策从「凭感觉」变为「可标定」。
- **纯后处理的工作点搜索** —— 阈值与 NMS 调整无需重新推理，零 GPU 成本。
- **完整工程配套** —— 自动化测试、数据/标签一致性校验、提交格式校验器。

---

## Architecture

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

两条融合路径在初始化时刻均退化为纯 RGB 模型，新模态只贡献梯度。

---

## Results

在 195 张留出验证集上（评测口径见 `docs/method.md` 2.2 节）：

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
  method.md       方法全文（架构 / 训练 / 评测口径 / 消融）
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
git clone <repo-url>
cd urban-multimodal-detection
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements.txt
```

- Python 3.12 / CUDA 12.x
- 训练建议单卡 ≥24 GB；推理任意 CUDA 显卡即可
- Linux 环境使用 `environment/setup_env.sh` 安装 PyTorch 2.13 / CUDA 12.6。原实验环境见 `environment/environment.txt`。

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

`manifests/` 中的清单记录了样本划分，路径需按实际位置改写；
每条样本带 `label_sha256`，载入时校验标签未被篡改。

---

## Usage

### 推理

```bash
cd code
python -m src.predict_rfdetr \
    --weights <model.safetensors> \
    --sample-manifest ../manifests/train-manifest.json \
    --out-dir ../runs/raw \
    --side val --imgsz 1280 --conf 0.001
```

### 后处理

```bash
python -m scripts.apply_nms \
    --source ../runs/raw --dest ../runs/final \
    --conf 0.35 --nms 0.6
```

> `output_rows` 的语义是「先取全局 top-100 再按 conf 过滤」，因此
> `top100 ∩ {conf ≥ t}` 与「直接用阈值 t 推理」的集合完全相同——
> **调整工作点只需后处理，无需重新推理，也不需要 GPU**。

### 训练

```bash
python -m scripts.run_experiment \
    --config configs/rf48c-early1280.yaml \
    --out ../runs/exp1
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

覆盖提交格式校验、几何反算一致性、六通道编码与零初始化等价性等。

---

## Model Weights

权重不随仓库分发（单文件约 132 MiB，超出 GitHub 单文件 100 MB 限制），
通过 Release 附件提供：

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

推理支持同名的 `.safetensors` 和 `.json` 文件。使用仍兼容的 `.pt` 权重时，可通过 `--weights-sha256` 传入表中的完整摘要。

---

## Engineering Notes

以下几份笔记记录了工程上真正花时间的部分，也是本项目可复用的经验：

| 文档 | 内容 |
|---|---|
| `docs/notes/2026-09-19-metric-arbitration.md` | 两种 AP 实现的系统性偏差与标定方法 |
| `docs/notes/2026-09-19-validation-noise.md` | 小验证集下的噪声量化与配对自助法 |
| `docs/notes/2026-09-16-rf48-implementation.md` | 融合结构实现与零初始化等价性验证 |
| `docs/notes/2026-09-04-reproduction-audit.md` | 数据与标签一致性审计 |

### 已验证无效的方向

记录在此，避免重复踩坑：

| 方向 | 结果 |
|---|---|
| **切块推理**（2×2，放大 1.75×） | mAP 62.55 → 42.76，全类均匀崩塌 |
| **提高推理分辨率** 1408 / 1536 / 1664 | 预测与 1280 版重合 91.5~92.7%，几乎无变化 |
| **原生 1536 训练** | 验证集 60.59 vs 1280 的 61.58 |
| **深度编码改对数** | 线性编码 5–95% 跨度 0.759，对数仅 0.207，线性更优 |
| **增加 P3 检测层** | `num_feature_levels` 1→2 破坏 `cross_attn` 权重形状，不可行 |

### 一条可复用的经验

> **两个模型的预测重合度超过 90% 时，二者的评测分数差通常 <0.2。**

可用 `analysis/compare_submissions.py` 快速评估两个预测集的重合度，
在大量候选中筛掉"几乎等价"的版本。

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
