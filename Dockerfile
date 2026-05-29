FROM python:3.11-slim

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

# Install dependencies
RUN poetry install

# Copy application code
COPY document_assistant/ ./document_assistant/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "document_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]