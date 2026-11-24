# RF48 实施与运行说明

本次实施替代原 4–7 天实验矩阵；只做 RGB 诊断 10 epochs、三模态 RF-DETR Medium 最多 60 epochs，随后双分辨率验证和正式推理。线上最优仍为 E9 **55.5370**，本地任何新分数均不等于线上评测分。

## 已验证的基础工程

- 2000/2000 组原始三模态合成的 legacy5 与旧 TIFF 逐像素相等。
- E9 在 195 张 A 验证图的 raw/TIFF 推理输出逐元素相等，AP 差异 0。
- 新主评估器与 pycocotools 在真实验证预测上的误差不超过 5.6e-17。
- RF-DETR 1.3.0 源码 commit：`03aa832eb07126b8bbd0fff45790b1f2ad45be2b`；公开权重 SHA-256：`749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`。
- 公开权重迁移 481 项，缺失项仅为新建的 12 类分类层；无 unexpected keys。
- 768/batch2 的 100 step 真实数据 AMP 探测通过；峰值 allocated 2.356 GiB、reserved 2.752 GiB，探测吞吐约 8.02 图/秒。这不是完整训练或正式推理的实测总时长。
- 新分支零初始化与 RGB 对照预测完全相等；训练后 RGB encoder、IR、Depth 分支均有有限非零梯度。
- 完整测试集 46 项通过，包含空标注、类别 0/11、严格打包和配置科学计数法读取；代码编译检查通过。

## 评分记录

以下均为 A 划分重新推理并导出 TXT 后的复算；旧工具算法的复算还受独立推理尺寸/补边、TXT 舍入、匹配和并列分数排序影响，不替换历史 validator 记录。

| 配置 | 细则复刻 AP×100 | Ultralytics 算法 TXT 复算×100 |
|---|---:|---:|
| E0，960/conf .40/NMS .55 | 45.0318 | 56.0054 |
| E9，1280/conf .40/NMS .40 | 43.1289 | 54.3232 |
| E9，1280/conf .001/NMS .40 | 46.0778 | 48.8368 |
| E12，960/conf .35/NMS .40 | 42.1904 | 52.5537 |

新候选主参照使用同一低阈值、全图 top100 的 E9=0.4607776962；主指标门槛为 +0.005，或稀有两类平均 +0.03 且总 AP 不低于 -.002。服务器实现未知，不推算官方分数。

## 接入时已修正的问题

RF-DETR valid-ratio 逻辑假定 NestedTensor 的有效区域从左上角开始。若把居中 letterbox 留白标成 tensor padding，会出现有效比例为零、损失有限但主干梯度为零的问题。当前固定正方形画布整体作为网络坐标系，NestedTensor mask 全 false；深度有效性另用独立输入通道，导出时严格逆变换并裁边。

FP16 默认 loss scale 65536 在探测中出现梯度溢出，起始 scale 降为 512 后 100 step 严格检查通过。训练保留动态缩放及溢出步跳过；连续五次异常即终止，不无限吞掉异常。

首次队列启动发现 YAML 读取 JSON 科学计数法时把 `1e-05` 当作字符串，已修复为按文件类型独立解析并加入回归测试。失败启动目录保留在 `setup-fail-*`，当时未完成训练；修复后队列重新启动，状态文件优先于历史日志中的旧报错。UTC 时间以明确时区记录，不能将服务器本地 UTC+8 的显示值当作 UTC。

## 数据与清理

原始训练/测试 ZIP、所有原图及标签、E0/E9/E10/E12 best 权重和提交包均保留。仅删除 2000 张已经逐像素验证的活动 TIFF，合计 **10339329084 bytes（约 9.63 GiB）**，非递归、精确清单删除。保护文件删除前后哈希相同。

审计：`runs/formal/rf48/input-parity.json`、`prediction-parity.json`、`protected-assets.json`、`tiff-retirement.json`。TIFF 可按 retirement 中的 rebuild 命令重建。日常训练/推理使用原始三模态读取，无需重建。

## 运行与恢复

从项目根目录运行：

```bash
screen -L -Logfile runs/formal/rf48/pipeline.log -dmS aic-rf48 .venv-rfdetr/bin/python -u -m scripts.run_rf48
```

关闭 VS Code 不会终止 screen 内队列；关闭实例、重启或进程崩溃仍会中断。队列使用文件锁防止重复 GPU 作业。恢复时运行同一命令，从 epoch 边界的 last.pt 恢复，配置/清单身份不同会拒绝。

状态：`runs/formal/rf48/pipeline-status.json`；当前阶段日志与 RGB/MM 的 `status.json`、`history.jsonl` 可查看进度。所有训练用 A 的 1805 图，195 图只验证；1000 张测试图只在最终推理阶段读取。全量训练、B/C 复核未执行。

结束后查 `runs/formal/rf48/delivery.json` 和 `DELIVERY.md`。提交包位于 `runs/formal/submissions/phase1-rf48-*.zip`，必须通过 strict 检查，缺失 TXT 不会自动补空。没有可信提升时保留 E9 恢复包，不自动上传。

手动预测接口：

```bash
.venv-rfdetr/bin/python -m src.predict_rfdetr --weights runs/formal/rf48/mm-candidate/best.pt --sample-manifest runs/formal/rf48/test-manifest.json --out-dir runs/formal/rf48/manual-test --imgsz 768
```

公开权重来源：<https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth>。依赖与代码快照保存在每个训练目录；现有全局 YOLO 环境未升级。
