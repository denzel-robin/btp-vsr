#!/usr/bin/env bash
# Run with bash from any directory. Supply an existing Python 3.10 executable.
set -euo pipefail
cd "$(dirname "$0")/.."
CDA_PYTHON="${CDA_PYTHON:-python3.10}"
"$CDA_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
if [[ ! -d .venv-cda ]]; then
    "$CDA_PYTHON" -m venv .venv-cda
fi
.venv-cda/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
# The bundled pip 23.0.1 misreads normalized names on the PyTorch index.
.venv-cda/bin/python -m pip install --only-binary=:all: 'pip==25.0.1'
.venv-cda/bin/python -m pip install --only-binary=:all: \
    'numpy==1.26.4' 'typing_extensions==4.15.0' 'opencv-python==4.11.0.86'
.venv-cda/bin/python -m pip install --only-binary=:all: \
    'torch==2.1.0+cu121' 'torchvision==0.16.0+cu121' \
    --index-url https://download.pytorch.org/whl/cu121
# Direct official binary: never resolve an MMCV source distribution.
.venv-cda/bin/python -m pip install --only-binary=:all: \
    'numpy==1.26.4' 'opencv-python==4.11.0.86' \
    https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl
.venv-cda/bin/python -m pip check
.venv-cda/bin/python scripts/check_cda_operator.py
