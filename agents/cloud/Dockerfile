FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
ENV UV_LINK_MODE=copy
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync
COPY *.py ./
CMD ["uv", "run", "python", "main.py"]
