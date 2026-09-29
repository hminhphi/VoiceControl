# syntax=docker/dockerfile:1
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python3 -m pip install uv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.10

COPY agents/infotainment/pyproject.toml ./

RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    uv venv --system-site-packages \
    && uv sync

COPY agents/infotainment/*.py ./

CMD ["uv", "run", "python", "main.py"]
