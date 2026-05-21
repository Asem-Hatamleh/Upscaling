#!/usr/bin/env bash
# One-shot environment setup for the realesrgan-lite stack on Linux.
#
# Stack covered:
#   - realesrgan_lite        (SRVGGNetCompact / realesr-general-x4v3)
#   - realesrgan_full        (RRDBNet x4plus)
#   - realesrgan_gfpgan      (Compact bg + GFPGAN-1.4 face restore)
#   - codeformer_compact     (Compact bg + CodeFormer-v1 face restore)
#
# Tested on:
#   - Ubuntu 24.04 / Python 3.11
#   - Dev:  RTX 5050 (sm_120 Blackwell), CUDA 13.x, driver 595.x
#   - Prod: NVIDIA A100 80 GB, CUDA 12.x
#
# Usage:
#   ./setup_linux.sh                    # full install (creates ./venv)
#   ./setup_linux.sh --no-downloads     # skip weight pre-fetch
#   ./setup_linux.sh --python=python3.11
set -euo pipefail

WITH_DOWNLOADS=1
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

for arg in "$@"; do
  case "$arg" in
    --no-downloads) WITH_DOWNLOADS=0 ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    -h|--help)
      sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# /tmp on this machine is tmpfs (~7 GB). pip build-isolation extracts torch+CUDA
# wheels (~10 GB) into TMPDIR and hits "Disk quota exceeded". Redirect to home.
export TMPDIR="${TMPDIR:-$ROOT/.build_tmp}"
mkdir -p "$TMPDIR"
echo "[setup] TMPDIR=$TMPDIR"

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

# 3) Project deps. basicsr/gfpgan/realesrgan need torch present in build env;
#    install them with --no-build-isolation so they don't re-download torch+CUDA
#    into TMPDIR.
pip install --no-cache-dir \
  numpy einops safetensors "huggingface_hub<2" imageio imageio-ffmpeg av Pillow \
  opencv-python tqdm PyYAML \
  "transformers==4.46.2" \
  lpips
pip install --no-cache-dir --no-build-isolation basicsr gfpgan realesrgan facexlib

# 3a) basicsr 1.4.2 still imports the removed torchvision.transforms.functional_tensor.
#     Patch the one offending import so `import realesrgan` works.
BSR_DEG="$(python -c 'import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), "data", "degradations.py"))')"
if [[ -f "$BSR_DEG" ]]; then
  sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|g' "$BSR_DEG"
  echo "[setup] patched $BSR_DEG"
fi

# 4) Pre-fetch model weights (Real-ESRGAN, GFPGAN, CodeFormer, facexlib).
if [[ $WITH_DOWNLOADS -eq 1 ]]; then
  python scripts/download_weights.py --model all
else
  echo "[setup] skipping weight downloads (--no-downloads)"
fi

# 4a) RIFE v6 weights are checked in, but the matching Python source isn't —
#     ECCV2022-RIFE shipped the network code separately. Clone the repo once
#     and copy the inference-time files into RIFE_trained_v6/model/. Only
#     needed if you plan to use `--frame-interp rife`.
if [[ ! -f RIFE_trained_v6/model/RIFE.py ]]; then
  if [[ ! -d third_party/RIFE ]]; then
    git clone --depth 1 https://github.com/megvii-research/ECCV2022-RIFE.git third_party/RIFE
  fi
  mkdir -p RIFE_trained_v6/model RIFE_trained_v6/train_log
  cp third_party/RIFE/model/{IFNet.py,IFNet_m.py,RIFE.py,refine.py,warplayer.py,laplacian.py} \
     RIFE_trained_v6/model/
  # loss.py is referenced at import time but only used during training; ship
  # a stub so Model() can construct without the full training stack.
  cat > RIFE_trained_v6/model/loss.py <<'PYEOF'
import torch
import torch.nn as nn


class EPE(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError


class SOBEL(nn.Module):
    def __init__(self):
        super().__init__()
        kX = torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]).reshape(1,1,3,3)
        self.register_buffer("kernelX", kX)
        self.register_buffer("kernelY", kX.transpose(2,3))
    def forward(self, *a, **k): raise NotImplementedError


class VGGPerceptualLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError
PYEOF
  touch RIFE_trained_v6/__init__.py RIFE_trained_v6/model/__init__.py RIFE_trained_v6/train_log/__init__.py
  echo "[setup] RIFE source files installed in RIFE_trained_v6/model/"
fi

# 5) Smoke check
python - <<'PY'
import importlib
for m in ["torch", "numpy", "imageio", "PIL", "cv2", "realesrgan", "gfpgan",
          "facexlib", "basicsr"]:
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
echo "[setup] Try:  python -m src.infer --model realesrgan_lite --input 'Real Test Video/1.mp4' --seconds 3 --pre-resize 80%"
