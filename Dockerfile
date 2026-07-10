FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY examples/ examples/
COPY eval/ eval/

# Default: run the minimal RAG CLI
# Override with: docker run ... python examples/02_rag_fastapi.py
CMD ["python", "examples/01_min_rag_cli.py", "--help"]
