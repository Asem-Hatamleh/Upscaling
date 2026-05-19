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

# 3) Project deps. basicsr/gfpgan/realesrgan need torch present in build env;
#    install them with --no-build-isolation so they don't re-download torch+CUDA
#    into TMPDIR.
pip install --no-cache-dir \
  numpy einops safetensors "huggingface_hub<2" imageio imageio-ffmpeg av Pillow \
  opencv-python tqdm PyYAML modelscope \
  diffusers "transformers==4.46.2" "accelerate<2" sentencepiece \
  peft ftfy torchsde torchmetrics pytorch-lightning pandas \
  bitsandbytes
pip install --no-cache-dir --no-build-isolation basicsr gfpgan realesrgan

# 3a) basicsr 1.4.2 still imports the removed torchvision.transforms.functional_tensor.
#     Patch the one offending import so `import realesrgan` works.
BSR_DEG="$(python -c 'import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), "data", "degradations.py"))')"
if [[ -f "$BSR_DEG" ]]; then
  sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|g' "$BSR_DEG"
  echo "[setup] patched $BSR_DEG"
fi

# 4) Clone FlashVSR + download weights
if [[ $WITH_DOWNLOADS -eq 1 ]]; then
  python scripts/download_weights.py --model all
else
  echo "[setup] skipping weight downloads (--no-downloads)"
fi

# 4a) Install FlashVSR's bundled diffsynth fork onto sys.path (editable, no deps —
#     torch is already installed and FlashVSR's pinned torch==2.6.0+cu124 would
#     downgrade our cu128 build).
if [[ -d third_party/FlashVSR ]]; then
  pip install --no-cache-dir --no-deps --no-build-isolation -e third_party/FlashVSR/ \
    || echo "[setup] diffsynth editable install failed; sys.path injection will still work"
fi

# 4b) FlashVSR's wan_video_dit.py hard-imports block_sparse_attn. We don't build
#     the kernel here (needs system CUDA toolkit + nvcc). Patch the import to be
#     optional and the dispatch to fall back to dense SDPA when the kernel
#     isn't present.
WAN_DIT="third_party/FlashVSR/diffsynth/models/wan_video_dit.py"
if [[ -f "$WAN_DIT" ]] && ! grep -q "BSA_AVAILABLE" "$WAN_DIT"; then
  python - "$WAN_DIT" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()
src = src.replace(
    "from block_sparse_attn import block_sparse_attn_func",
    "try:\n"
    "    from block_sparse_attn import block_sparse_attn_func\n"
    "    BSA_AVAILABLE = True\n"
    "except (ImportError, ModuleNotFoundError):\n"
    "    block_sparse_attn_func = None\n"
    "    BSA_AVAILABLE = False",
)
src = src.replace(
    "def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):\n"
    "    if attention_mask is not None:",
    "def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attention_mask=None, return_KV=False):\n"
    "    if attention_mask is not None and not BSA_AVAILABLE:\n"
    "        q_d = rearrange(q, \"b s (n d) -> b n s d\", n=num_heads)\n"
    "        k_d = rearrange(k, \"b s (n d) -> b n s d\", n=num_heads)\n"
    "        v_d = rearrange(v, \"b s (n d) -> b n s d\", n=num_heads)\n"
    "        x = F.scaled_dot_product_attention(q_d, k_d, v_d)\n"
    "        x = rearrange(x, \"b n s d -> b s (n d)\", n=num_heads)\n"
    "        if return_KV:\n"
    "            return x, k, v\n"
    "        return x\n"
    "    if attention_mask is not None:",
)
open(p, "w").write(src)
print("[setup] patched", p)
PY
fi

# 4c) RIFE v6 weights are checked in, but the matching Python source isn't —
#     ECCV2022-RIFE shipped the network code separately. Clone the repo once
#     and copy the inference-time files into RIFE_trained_v6/model/.
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
    pip install --no-cache-dir sageattention || \
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
