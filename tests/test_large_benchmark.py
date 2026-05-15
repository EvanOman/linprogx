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


def test_large_benchmark_clarabel_formulation_solves_feasible_eq_bounds_lp() -> None:
    problem_data = {
        "A_scipy": scipy.sparse.csc_matrix([[1.0, 1.0]]),
        "b": np.array([3.0]),
        "c": np.array([1.0, 2.0]),
        "lo": np.array([0.0, 0.0]),
        "hi": np.array([2.0, 3.0]),
    }
    captured: dict[str, np.ndarray] = {}
    real_solver = clarabel.DefaultSolver

    class CapturingSolver:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self._solver = real_solver(*args, **kwargs)

        def solve(self):  # type: ignore[no-untyped-def]
            result = self._solver.solve()
            captured["x"] = np.array(result.x, dtype=float)
            return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(clarabel, "DefaultSolver", CapturingSolver)
    try:
        result = bench_large._run_clarabel(problem_data)
    finally:
        monkeypatch.undo()

    x = captured["x"]
    equality_residual = problem_data["A_scipy"] @ x - problem_data["b"]
    lower_residual = problem_data["lo"] - x
    upper_residual = x - problem_data["hi"]

    assert result.status == "optimal"
    assert result.objective == pytest.approx(4.0, abs=1e-6)
    assert np.max(np.abs(equality_residual)) <= 1e-7
    assert np.max(np.maximum(lower_residual, 0.0)) <= 1e-7
    assert np.max(np.maximum(upper_residual, 0.0)) <= 1e-7


@pytest.mark.parametrize(
    ("clarabel_status", "expected"),
    [
        ("Solved", "optimal"),
        ("AlmostSolved", "optimal"),
        ("DualInfeasible", "reported_dual_infeasible"),
        ("AlmostDualInfeasible", "reported_dual_infeasible"),
        ("PrimalInfeasible", "reported_primal_infeasible"),
        ("AlmostPrimalInfeasible", "reported_primal_infeasible"),
        ("MaxIterations", "maxiterations"),
    ],
)
def test_large_benchmark_clarabel_status_mapping(clarabel_status: str, expected: str) -> None:
    assert bench_large._clarabel_status(clarabel_status) == expected
