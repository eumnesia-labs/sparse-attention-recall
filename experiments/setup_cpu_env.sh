#!/usr/bin/env bash
# CPU environment for tests, analysis and paper work (no GPU, no cloud credentials).
#   zoology (HazyResearch, pinned 1ad20d1) + experiments/patch_zoology_accum.py  ->  ./zoology
#   Python venv with CPU torch 2.14.0 and the harness's packages                 ->  ./.venv-cpu
# Then: export ZOOLOGY_DIR=$PWD/zoology; run tests one at a time with OMP_NUM_THREADS=4.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"

if [ ! -d zoology/.git ]; then
  git clone -q https://github.com/HazyResearch/zoology.git zoology
fi
git -C zoology checkout -q 1ad20d1
"$PY" experiments/patch_zoology_accum.py zoology

if [ ! -x .venv-cpu/bin/python ]; then
  "$PY" -m venv .venv-cpu
fi
.venv-cpu/bin/pip install -q --upgrade pip
.venv-cpu/bin/pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
  || .venv-cpu/bin/pip install -q torch==2.14.0 torchvision==0.29.0
.venv-cpu/bin/pip install -q einops==0.8.2 pydantic==2.13.5 numpy pandas tqdm pyyaml rich xxhash matplotlib wandb==0.30.0
.venv-cpu/bin/python -c "import torch, einops, pydantic, sys; sys.path.insert(0, 'zoology'); import zoology.model; print('cpu env ok: torch', torch.__version__)"
echo "export ZOOLOGY_DIR=$REPO/zoology"
