from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from linprogx.sparse import SparseLPProblem, SparseSolver

BENCH_LARGE_PATH = Path(__file__).resolve().parents[1] / "bench_large.py"
spec = importlib.util.spec_from_file_location("bench_large", BENCH_LARGE_PATH)
assert spec is not None
bench_large = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["bench_large"] = bench_large
spec.loader.exec_module(bench_large)

BENCH_CYCLE_PATH = Path(__file__).resolve().parents[1] / "bench_cycle.py"
cycle_spec = importlib.util.spec_from_file_location("bench_cycle", BENCH_CYCLE_PATH)
assert cycle_spec is not None
bench_cycle = importlib.util.module_from_spec(cycle_spec)
assert cycle_spec.loader is not None
sys.modules["bench_cycle"] = bench_cycle
cycle_spec.loader.exec_module(bench_cycle)

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


def test_cycle_sparse_pdhg_reaches_scaled_feasibility_guardrail() -> None:
    problem_data = bench_cycle.load_cycle(bench_cycle.DATA_PATH)
    result = SparseSolver(
        algorithm="pdhg",
        max_iterations=50_000,
        eps=2e-5,
        check_interval=50_000,
    ).solve(
        SparseLPProblem(
            c=problem_data["c"].tolist(),
            A_eq=problem_data["A"],
            b_eq=problem_data["b"].tolist(),
            objective="min",
            bounds=bench_cycle._bounds(problem_data),
            name="cycle",
        )
    )

    x = np.array(result.solution.x, dtype=float)
    max_residual = float(np.max(np.abs(problem_data["A_scipy"] @ x - problem_data["b"])))

    assert max_residual <= 1e-2


def test_cycle_sparse_pdhg_untuned_reaches_benchmark_quality() -> None:
    problem_data = bench_cycle.load_cycle(bench_cycle.DATA_PATH)
    result = SparseSolver(
        algorithm="pdhg",
        max_iterations=50_000,
        eps=2e-5,
        check_interval=50_000,
    ).solve(
        SparseLPProblem(
            c=problem_data["c"].tolist(),
            A_eq=problem_data["A"],
            b_eq=problem_data["b"].tolist(),
            objective="min",
            bounds=bench_cycle._bounds(problem_data),
            name="cycle",
        )
    )

    x = np.array(result.solution.x, dtype=float)
    objective = float(problem_data["c"] @ x)
    max_residual = float(np.max(np.abs(problem_data["A_scipy"] @ x - problem_data["b"])))

    assert result.solution.status.value == "optimal"
    assert result.solution.iterations <= 50_000
    assert max_residual <= 2e-5
    assert abs(objective - bench_cycle.EXPECTED_CYCLE_OBJECTIVE) <= 1e-2


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
