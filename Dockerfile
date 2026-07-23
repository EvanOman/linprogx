FROM python:3.12-slim

# Install build deps for the C extension + OpenBLAS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and build the package (C extensions compile against OpenBLAS)
COPY pyproject.toml MANIFEST.in LICENSE README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Install API dependencies
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.2 pydantic==2.11.3

# Copy the demo API code
COPY demo/api/ /app/demo/api/

EXPOSE 10000

CMD ["uvicorn", "demo.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
