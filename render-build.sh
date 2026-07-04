#!/usr/bin/env bash
set -e

# Try to install OpenBLAS for full performance (may fail in sandboxed envs)
if command -v apt-get &>/dev/null; then
    apt-get update && apt-get install -y --no-install-recommends libopenblas-dev 2>/dev/null || true
fi

# Install linprogx from source
pip install .

# Install API dependencies
pip install fastapi==0.115.12 uvicorn==0.34.2 pydantic==2.11.3
