from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_LARGE_PATH = Path(__file__).resolve().parents[1] / "bench_large.py"
spec = importlib.util.spec_from_file_location("bench_large", BENCH_LARGE_PATH)
assert spec is not None
bench_large = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["bench_large"] = bench_large
spec.loader.exec_module(bench_large)

scipy = pytest.importorskip("scipy")
clarabel = pytest.importorskip("clarabel")
del clarabel


def test_large_benchmark_clarabel_formulation_solves_feasible_eq_bounds_lp() -> None:
    problem_data = {
        "A_scipy": scipy.sparse.csc_matrix([[1.0, 1.0]]),
        "b": np.array([3.0]),
        "c": np.array([1.0, 2.0]),
        "lo": np.array([0.0, 0.0]),
        "hi": np.array([2.0, 3.0]),
    }

    result = bench_large._run_clarabel(problem_data)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(4.0, abs=1e-6)
