# syntax=docker/dockerfile:1
# ============================================================
# orchestrator — API service sharing l4t-base layers
# (was python:3.11-slim + pip torch CPU).
#
# Reuses torch 2.8 + numpy/scipy from l4t-base instead of downloading
# a separate CPU torch (~0.5GB, unshared layers). The service still
# runs CPU-only (no GPU request in docker-compose.yml); torch here
# only serves sentence-transformers embeddings.
#
# NOTE: base Python is 3.10 (l4t-jetpack r36.4). Code verified 3.10-clean
# (no tomllib/ExceptionGroup/TaskGroup/asyncio.timeout). If a future
# dependency requires >=3.11, revert to python:3.11-slim + torch CPU.
# ============================================================
ARG BASE_IMAGE=orchestrator-on-edge/l4t-base:r36.4.0-torch2.8
FROM ${BASE_IMAGE}

WORKDIR /app

COPY orchestrator/requirements.txt ./

# Offline wheelhouse (voice_processing/wheels in the build context).
RUN --mount=type=bind,source=voice_processing/wheels,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links /tmp/wheels \
        --upgrade pip \
    && python3 -m pip install --no-index --find-links /tmp/wheels --no-build-isolation \
        -r requirements.txt

COPY orchestrator/*.py ./

CMD ["python3", "main.py"]
