#!/usr/bin/env bash
# One-shot environment setup for the UpScaling project on Linux.
#
# Tested on:
#   - Ubuntu 24.04 / Python 3.11
#   - Dev:  RTX 5050 (sm_120 Blackwell), CUDA 13.x, driver 595.x
#   - Prod: NVIDIA A100 80 GB, CUDA 12.x
#
# Usage:
#   ./setup_linux.sh            # full install (creates ./venv)
#   ./setup_linux.sh --no-sage  # skip SageAttention build
#   ./setup_linux.sh --no-bsa   # skip Block-Sparse-Attention build (FlashVSR will be slower)
#
set -euo pipefail

WITH_SAGE=1
WITH_BSA=1
WITH_DOWNLOADS=1
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

for arg in "$@"; do
  case "$arg" in
    --no-sage) WITH_SAGE=0 ;;
    --no-bsa) WITH_BSA=0 ;;
    --no-downloads) WITH_DOWNLOADS=0 ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[setup] python = $($PYTHON_BIN -V 2>&1)"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] ERROR: $PYTHON_BIN not found."
  echo "  Install with:  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv python3.11-dev"
  exit 1
fi

# 1) venv
if [[ ! -d "$ROOT/venv" ]]; then
  "$PYTHON_BIN" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -U pip wheel setuptools

# 2) PyTorch — pick the right wheel for your GPU
#    Detect Blackwell (sm_120) vs Ampere/Hopper.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo unknown)"
echo "[setup] GPU: $GPU_NAME"
case "$GPU_NAME" in
  *RTX\ 50*|*Blackwell*)
    echo "[setup] installing PyTorch nightly cu128 (sm_120 support)"
    pip install --pre torch torchvision \
      --index-url https://download.pytorch.org/whl/nightly/cu128
    ;;
  *A100*|*H100*|*A40*)
    echo "[setup] installing PyTorch stable cu121"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ;;
  *)
    echo "[setup] installing PyTorch stable cu121 (default)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ;;
esac

# 3) Project deps (skip the platform-conditional perf libs; handled below)
pip install -r requirements.txt

# 4) Clone FlashVSR + download weights
if [[ $WITH_DOWNLOADS -eq 1 ]]; then
  python scripts/download_weights.py --model all
else
  echo "[setup] skipping weight downloads (--no-downloads)"
fi

# 5) Block-Sparse-Attention (FlashVSR fast path)
if [[ $WITH_BSA -eq 1 ]]; then
  if ! python -c "import block_sparse_attn" 2>/dev/null; then
    echo "[setup] building Block-Sparse-Attention from source"
    mkdir -p third_party && cd third_party
    if [[ ! -d Block-Sparse-Attention ]]; then
      git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention.git
    fi
    cd Block-Sparse-Attention
    pip install -e . || echo "[setup] Block-Sparse-Attention build failed (FlashVSR will fall back)."
    cd "$ROOT"
  else
    echo "[setup] block_sparse_attn already installed"
  fi
fi

# 6) SageAttention (Linux only)
if [[ $WITH_SAGE -eq 1 ]]; then
  if ! python -c "import sageattention" 2>/dev/null; then
    echo "[setup] installing SageAttention"
    pip install sageattention || \
      (echo "[setup] pypi sageattention failed; trying source build"; \
       cd third_party && \
       git clone --depth 1 https://github.com/thu-ml/SageAttention.git && \
       cd SageAttention && pip install -e . || \
       echo "[setup] SageAttention not installed; --sage-attn will fall back.")
  else
    echo "[setup] sageattention already installed"
  fi
fi

# 7) Smoke check
python - <<'PY'
import importlib, sys
for m in ["torch", "numpy", "imageio", "PIL", "cv2"]:
    try:
        importlib.import_module(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  ??  {m}: {e}")
import torch
print("  cuda:", torch.cuda.is_available(),
      "device_count:", torch.cuda.device_count(),
      "name0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

echo
echo "[setup] DONE. Activate with:  source venv/bin/activate"
echo "[setup] Try:  python -m src.infer --model flashvsr_tiny --input 'Real Test Video/1.mp4' --seconds 3 --pre-resize vga"
