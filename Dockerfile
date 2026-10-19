# Inference image for the linear models (TF-IDF / prior). No PyTorch: ~250 MB.
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ARG MODEL_DIR=artifacts/tfidf_logreg
COPY ${MODEL_DIR} /app/model
ENV FIELDCLF_MODEL_DIR=/app/model

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/app/.venv/bin/uvicorn", "fieldclf.serve:app", "--host", "0.0.0.0", "--port", "8000"]
