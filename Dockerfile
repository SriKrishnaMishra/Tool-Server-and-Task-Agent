FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY backend ./backend
COPY ai ./ai
COPY frontend ./frontend
COPY tests ./tests
COPY README.md ./

RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["python"]
