#!/usr/bin/env python3
"""Download model weights for FlashVSR Tiny and (optionally) Real-ESRGAN.

Examples:
    python scripts/download_weights.py --model flashvsr
    python scripts/download_weights.py --model realesrgan
    python scripts/download_weights.py --model all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLASHVSR_DIR = ROOT / "third_party" / "FlashVSR"
FLASHVSR_REPO = "https://github.com/OpenImagingLab/FlashVSR.git"
FLASHVSR_HF = "JunhaoZhuang/FlashVSR-v1.1"
FLASHVSR_WEIGHTS_DEST = FLASHVSR_DIR / "examples" / "WanVSR" / "FlashVSR-v1.1"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def clone_flashvsr() -> None:
    if FLASHVSR_DIR.exists():
        print(f"[flashvsr] repo already exists at {FLASHVSR_DIR}")
        return
    FLASHVSR_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", FLASHVSR_REPO, str(FLASHVSR_DIR)])


def download_flashvsr_weights() -> None:
    FLASHVSR_WEIGHTS_DEST.mkdir(parents=True, exist_ok=True)
    needed = [
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model_streaming_dmd.safetensors",
    ]
    missing = [f for f in needed if not (FLASHVSR_WEIGHTS_DEST / f).exists()]
    if not missing:
        print(f"[flashvsr] weights complete in {FLASHVSR_WEIGHTS_DEST}")
        return
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print("[flashvsr] huggingface_hub missing; install with `pip install huggingface_hub`",
              file=sys.stderr)
        sys.exit(2)
    for f in missing:
        print(f"[flashvsr] hf:{FLASHVSR_HF}/{f}")
        local = hf_hub_download(repo_id=FLASHVSR_HF, filename=f)
        shutil.copy2(local, FLASHVSR_WEIGHTS_DEST / f)


def download_realesrgan_weights() -> None:
    dest = ROOT / "weights" / "realesrgan"
    dest.mkdir(parents=True, exist_ok=True)
    urls = [
        ("realesr-general-x4v3.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
        ("realesr-general-wdn-x4v3.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"),
        ("realesr-animevideov3.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"),
    ]
    for name, url in urls:
        out = dest / name
        if out.exists():
            print(f"[realesrgan] already have {name}")
            continue
        print(f"[realesrgan] downloading {url}")
        urllib.request.urlretrieve(url, out)


def download_gfpgan_weights() -> None:
    dest = ROOT / "weights" / "gfpgan"
    dest.mkdir(parents=True, exist_ok=True)
    urls = [
        ("GFPGANv1.4.pth",
         "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"),
    ]
    for name, url in urls:
        out = dest / name
        if out.exists():
            print(f"[gfpgan] already have {name}")
            continue
        print(f"[gfpgan] downloading {url}")
        urllib.request.urlretrieve(url, out)


def download_codeformer_weights() -> None:
    dest = ROOT / "weights" / "codeformer"
    dest.mkdir(parents=True, exist_ok=True)
    urls = [
        ("codeformer.pth",
         "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"),
    ]
    for name, url in urls:
        out = dest / name
        if out.exists():
            print(f"[codeformer] already have {name}")
            continue
        print(f"[codeformer] downloading {url}")
        urllib.request.urlretrieve(url, out)


def download_basicvsrpp_weights() -> None:
    dest = ROOT / "weights" / "basicvsrpp"
    dest.mkdir(parents=True, exist_ok=True)
    urls = [
        ("basicvsr_plusplus_reds4.pth",
         "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
         "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"),
    ]
    for name, url in urls:
        out = dest / name
        if out.exists():
            print(f"[basicvsrpp] already have {name}")
            continue
        print(f"[basicvsrpp] downloading {url}")
        urllib.request.urlretrieve(url, out)
    # facexlib face-detection weight (RetinaFace ResNet50) — auto-downloads
    # on first use, but pre-fetch so the first inference run isn't slow.
    facexlib_dest = ROOT / "weights" / "facexlib"
    facexlib_dest.mkdir(parents=True, exist_ok=True)
    fxlib_urls = [
        ("detection_Resnet50_Final.pth",
         "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
        ("parsing_parsenet.pth",
         "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"),
    ]
    for name, url in fxlib_urls:
        out = facexlib_dest / name
        if out.exists():
            print(f"[facexlib] already have {name}")
            continue
        print(f"[facexlib] downloading {url}")
        urllib.request.urlretrieve(url, out)


def check_rife() -> None:
    rife = ROOT / "RIFE_trained_v6" / "train_log" / "flownet.pkl"
    if rife.exists():
        print(f"[rife] OK -> {rife}")
        return
    print("[rife] WARNING: RIFE_trained_v6/train_log/flownet.pkl missing.")
    print("[rife] Get v6 weights from https://github.com/megvii-research/ECCV2022-RIFE")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["flashvsr", "realesrgan", "gfpgan", "codeformer",
                            "basicvsrpp", "all"],
                   required=True)
    args = p.parse_args()

    if args.model in ("flashvsr", "all"):
        clone_flashvsr()
        download_flashvsr_weights()
    if args.model in ("realesrgan", "all"):
        download_realesrgan_weights()
    if args.model in ("gfpgan", "all"):
        download_gfpgan_weights()
    if args.model in ("codeformer", "all"):
        download_codeformer_weights()
    if args.model in ("basicvsrpp", "all"):
        download_basicvsrpp_weights()
    check_rife()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
