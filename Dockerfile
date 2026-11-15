# Multi-stage Dockerfile for NutriChatBot (Streamlit)
# Stage 1: Build dependencies with uv
FROM python:3.12-slim AS builder

WORKDIR /app

# Install curl and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification files
COPY pyproject.toml uv.lock ./

# Install dependencies into virtual environment
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Minimal hardened runtime
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root system user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy virtualenv and uv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy application source code and datasets
COPY app.py config.py embeddings.py ./
COPY services/ ./services/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Set file ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0" \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py"]
