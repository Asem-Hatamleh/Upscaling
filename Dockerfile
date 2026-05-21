# Face-blur pipeline image. A100/Ampere+ ready (CUDA 12.1).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    YOLO_CONFIG_DIR=/tmp \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist;file,http,https,tcp,tls,crypto"

# system deps: ffmpeg (encode/decode), libs for opencv, curl/ca-certs for mediamtx download
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# mediamtx (RTMP/RTSP/HLS/WebRTC relay). Pinned version, Linux amd64.
ARG MEDIAMTX_VERSION=1.9.3
RUN mkdir -p /opt/mediamtx && cd /opt/mediamtx \
    && curl -sSL -o mtx.tgz "https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/mediamtx_v${MEDIAMTX_VERSION}_linux_amd64.tar.gz" \
    && tar -xzf mtx.tgz && rm mtx.tgz \
    && ln -s /opt/mediamtx/mediamtx /usr/local/bin/mediamtx

WORKDIR /app

# python deps (cached layers)
COPY requirements.txt requirements-pipeline.txt /app/
RUN pip install --upgrade pip wheel "setuptools<70" \
    && pip install --no-build-isolation -r /app/requirements.txt \
    && pip install -r /app/requirements-pipeline.txt

# patch basicsr torchvision import (functional_tensor removed)
RUN python -c "import basicsr,os; p=os.path.join(os.path.dirname(basicsr.__file__),'data','degradations.py'); s=open(p).read().replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); open(p,'w').write(s)"

# project files (flat layout: Upscaling repo root)
COPY blur_faces.py pipeline_live.py LiveFeeder.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY RIFE_trained_v6/ ./RIFE_trained_v6/

# default ports (override in compose)
EXPOSE 1935 8554 8888 8889 8189/udp

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "pipeline_live.py", "--help"]
