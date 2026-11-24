#!/usr/bin/env bash
# Install the Linux CUDA environment from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r environment/requirements.txt
python -m pip install --no-deps rfdetr==1.3.0

RFD_SITE=$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("rfdetr").origin).parent.parent)')
RFD_PATCH="$PWD/environment/rfdetr-1.3.0-rectangular.patch"
if ! patch -d "$RFD_SITE" -p1 --reverse --dry-run --silent < "$RFD_PATCH" > /dev/null; then
    patch -d "$RFD_SITE" -p1 --forward < "$RFD_PATCH"
fi
