#!/usr/bin/env python3
"""Download model weights for the Real-ESRGAN / GFPGAN / CodeFormer stack.

Examples:
    python scripts/download_weights.py --model realesrgan
    python scripts/download_weights.py --model gfpgan
    python scripts/download_weights.py --model codeformer
    python scripts/download_weights.py --model all
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        ("RealESRGAN_x4plus.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
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


def download_facexlib_weights() -> None:
    """RetinaFace + ParseNet weights are auto-downloaded by facexlib on first
    use; pre-fetch them so the first inference run isn't slow."""
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
                   choices=["realesrgan", "gfpgan", "codeformer",
                            "facexlib", "all"],
                   required=True)
    args = p.parse_args()

    if args.model in ("realesrgan", "all"):
        download_realesrgan_weights()
    if args.model in ("gfpgan", "all"):
        download_gfpgan_weights()
    if args.model in ("codeformer", "all"):
        download_codeformer_weights()
    if args.model in ("facexlib", "all"):
        download_facexlib_weights()
    check_rife()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
