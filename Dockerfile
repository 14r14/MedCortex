FROM python:3.11-slim AS base

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./

# Sync deps
RUN uv sync --frozen --no-dev

# App source
COPY app ./app
COPY README.md ./README.md

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]


