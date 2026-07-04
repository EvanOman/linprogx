FROM python:3.12-slim

# Install build deps for the C extension + OpenBLAS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package from the sdist (builds C extensions against OpenBLAS)
COPY dist/linprogx-0.1.0.tar.gz /tmp/linprogx-0.1.0.tar.gz
RUN pip install --no-cache-dir /tmp/linprogx-0.1.0.tar.gz

# Install API dependencies
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.2 pydantic==2.11.3

# Copy the demo API code
COPY demo/api/ /app/demo/api/

EXPOSE 10000

CMD ["uvicorn", "demo.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
