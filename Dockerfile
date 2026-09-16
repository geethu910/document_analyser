FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Disable virtualenv inside container (important)
ENV POETRY_VENV_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* ./

# copy readme
COPY README.md ./

# Copy application code
COPY document_assistant/ ./document_assistant/

# copy static files if any
COPY static/ ./static/

# Install dependencies
RUN poetry install --only main

# Bake the embedding/cross-encoder models into the image so Cloud Run cold starts don't hit
# Hugging Face over the network. These build args MUST match the EMBED_MODEL/CROSS_ENCODER_MODEL
# env vars set at deploy time — otherwise HF_HOME won't have that model cached and it silently
# falls back to downloading from HF at runtime.
ARG EMBED_MODEL=all-MiniLM-L6-v2
ARG CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache && \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${CROSS_ENCODER_MODEL}')"

# create user
RUN useradd -r -m appuser && \
    chown -R appuser:appuser /app

# Create uploads directory and set ownership
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app/uploads
RUN chown -R appuser:appuser /app/hf_cache

# Switch to non-root user
USER appuser


EXPOSE 8000

CMD ["uvicorn", "document_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]