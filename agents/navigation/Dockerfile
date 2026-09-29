FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install uv
ENV UV_LINK_MODE=copy

COPY pyproject.toml ./
RUN uv venv --system-site-packages && uv sync

COPY *.py ./
COPY data/ ./data/

CMD ["uv", "run", "python", "main.py"]
