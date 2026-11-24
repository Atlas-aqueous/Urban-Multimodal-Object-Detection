#!/usr/bin/env bash
# Linux 服务器一键环境。用法: bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 固定已验证的 PyTorch 版本；驱动须兼容 CUDA 12.1 运行时。
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip freeze > requirements.lock.txt
python - <<'EOF'
import torch, ultralytics, cv2
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| 显存 GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
print("ultralytics", ultralytics.__version__)
EOF
