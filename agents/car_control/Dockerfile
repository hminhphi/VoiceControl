# syntax=docker/dockerfile:1
# car_control — installs deps into system site-packages on top of l4t-base
# (torch 2.8 + numpy 1.26 already present, shared not duplicated).
#
# NOTE: no uv venv. uv does not honour --system-site-packages, so the old
# `uv venv --system-site-packages && uv sync` reinstalled a duplicate torch
# (~1GB). Plain pip skips anything already satisfied by the base.
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY agents/car_control/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

COPY agents/car_control/*.py ./

CMD ["python3", "main.py"]
