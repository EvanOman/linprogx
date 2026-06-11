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
    uv run ty check .

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing

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

ci: lint format-check type test

install:
    uv sync --extra dev
