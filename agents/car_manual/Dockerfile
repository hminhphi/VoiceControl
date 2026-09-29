# syntax=docker/dockerfile:1
# car_manual — installs deps into system site-packages on top of l4t-base,
# which already provides torch 2.8 + numpy/scipy (shared, not duplicated).
#
# NOTE: no uv venv. uv does not honour --system-site-packages, so the old
# `uv venv --system-site-packages && uv sync` reinstalled a full duplicate
# torch/transformers stack (~1-2GB) into a per-agent venv. Plain pip skips
# anything already satisfied by the base.
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY agents/car_manual/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

COPY agents/car_manual/*.py ./
COPY agents/car_manual/data/ ./data/

CMD ["python3", "main.py"]
