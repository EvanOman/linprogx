set shell := ["bash", "-cu"]

default:
    @just --list

fmt:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check . --fix

type:
    uv run ty check . --exclude "demo/" --exclude "setup.py"

security:
    uv run --extra dev bandit -q --severity-level medium -r src bench.py bench_cycle.py bench_dense.py bench_large.py bench_plots.py bench_sparse_fast.py
    uv run --extra dev pip-audit

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=98

plots:
    uv run python bench_plots.py --repeats 50

large-bench:
    uv run python bench_large.py

dense-bench:
    uv run python bench_dense.py

sparse-fast-bench:
    uv run python bench_sparse_fast.py

cycle-bench:
    uv run python bench_cycle.py

fc: fmt lint-fix lint type test

ci: lint format-check type security test-cov

install:
    uv sync --extra dev
