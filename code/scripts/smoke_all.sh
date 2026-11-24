#!/usr/bin/env bash
# 端到端冒烟：示例数据 → 两种预处理 → 权重手术 → 1epoch×2 → 推理 → 自检打包
# 用法: bash scripts/smoke_all.sh [data_root]（默认 ./data）
# 面向 Linux 服务器 / 已激活环境（命令用 python）；
# Windows 本地请在项目根目录用 .venv/Scripts/python.exe 逐条执行等效命令（见 README）。
set -e
cd "$(dirname "$0")/.."
DATA=${1:-./data}

python src/prepare_data.py --data-root "$DATA" --out datasets/sample-b --mode 3ch --val-ratio 0.3 --overwrite
python src/prepare_data.py --data-root "$DATA" --out datasets/sample-a --mode 5ch --val-ratio 0.3 --overwrite
python src/make_5ch_weights.py
python src/train.py --data datasets/sample-b/data.yaml --model yolov8m.pt \
  --imgsz 320 --epochs 1 --batch 4 --workers 0 --name smoke-b
python src/train.py --data datasets/sample-a/data.yaml --model runs/pretrain/yolov8m-5ch.pt \
  --imgsz 320 --epochs 1 --batch 4 --workers 0 --name smoke-a
python src/predict.py --weights runs/smoke-a/weights/best.pt \
  --image-dir datasets/sample-a/images/val --out-dir submit_txt --imgsz 320
python src/check_submit.py --pred-dir submit_txt \
  --image-dir datasets/sample-a/images/val --out submit_smoke.zip
echo "=== SMOKE ALL PASS ==="
