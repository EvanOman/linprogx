"""End-to-end integration tests for the sparse solver portfolio."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from scipy import optimize as scipy_optimize

from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix
from linprogx.types import Status


def _random_feasible_problem(
    seed: int, rows: int, cols: int
) -> tuple[SparseLPProblem, list[float]]:
    """A bounded feasible LP: pick an interior point and set b = A x*."""
    rng = random.Random(seed)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for _ in range(rows):
        width = rng.randint(2, 5)
        cols_in_row = sorted(rng.sample(range(cols), width))
        indices.extend(cols_in_row)
        data.extend(rng.uniform(-2.0, 2.0) for _ in cols_in_row)
        indptr.append(len(indices))
    x_star = [rng.uniform(1.0, 9.0) for _ in range(cols)]
    b = [
        sum(data[p] * x_star[indices[p]] for p in range(indptr[i], indptr[i + 1]))
        for i in range(rows)
    ]
    c = [rng.uniform(-1.0, 1.0) for _ in range(cols)]
    matrix = csr_matrix(rows, cols, indptr, indices, data)
    problem = SparseLPProblem(
        c,
        A_eq=matrix,
        b_eq=b,
        objective="min",
        bounds=[(0.0, 10.0)] * cols,
    )
    return problem, b


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_auto_solver_matches_scipy_on_random_feasible_lps(seed: int) -> None:
    rows, cols = 25, 60
    problem, b = _random_feasible_problem(seed, rows, cols)

    result = SparseSolver(algorithm="auto", eps=1e-9, max_iterations=50_000).solve(problem)
    assert result.solution.status == Status.OPTIMAL

    assert problem.A_eq is not None
    indptr, indices, data = problem.A_eq.to_components()
    dense = np.zeros((rows, cols))
    for i in range(rows):
        for p in range(indptr[i], indptr[i + 1]):
            dense[i, indices[p]] = data[p]
    reference = scipy_optimize.linprog(
        problem.c, A_eq=dense, b_eq=b, bounds=[(0.0, 10.0)] * cols, method="highs"
    )
    assert reference.success

    assert result.solution.objective_value == pytest.approx(
        float(reference.fun), rel=1e-6, abs=1e-6
    )
    x = np.array(result.solution.x)
    assert float(np.max(np.abs(dense @ x - np.array(b)))) < 1e-6
    assert float(np.min(x)) >= -1e-8
    assert float(np.max(x)) <= 10.0 + 1e-8


@pytest.mark.parametrize("algorithm", ["ipm", "pdhg", "auto"])
def test_solver_is_deterministic(algorithm: str) -> None:
    problem, _ = _random_feasible_problem(3, 12, 30)

    chosen = cast('Literal["ipm", "pdhg", "auto"]', algorithm)
    solver = SparseSolver(algorithm=chosen, eps=1e-8, max_iterations=50_000)
    first = solver.solve(problem)
    second = solver.solve(problem)

    assert first.solution.status == second.solution.status
    assert first.solution.iterations == second.solution.iterations
    assert first.solution.objective_value == second.solution.objective_value
    assert first.solution.x == second.solution.x


def test_presolve_on_and_off_agree() -> None:
    problem, _ = _random_feasible_problem(11, 15, 35)

    with_presolve = SparseSolver(algorithm="auto", eps=1e-9, presolve=True).solve(problem)
    without_presolve = SparseSolver(algorithm="auto", eps=1e-9, presolve=False).solve(problem)

    assert with_presolve.solution.status == Status.OPTIMAL
    assert without_presolve.solution.status == Status.OPTIMAL
    assert with_presolve.solution.objective_value == pytest.approx(
        without_presolve.solution.objective_value, abs=1e-6
    )


def _load_cycle_module():
    path = Path(__file__).resolve().parents[1] / "bench_cycle.py"
    spec = importlib.util.spec_from_file_location("bench_cycle_integration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_cycle_integration"] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_auto_routes_to_ipm_with_high_accuracy() -> None:
    bench_cycle = _load_cycle_module()
    problem_data = bench_cycle.load_cycle(bench_cycle.DATA_PATH)

    result = SparseSolver(algorithm="auto", eps=2e-5, max_iterations=50_000).solve(
        SparseLPProblem(
            c=problem_data["c"].tolist(),
            A_eq=problem_data["A"],
            b_eq=problem_data["b"].tolist(),
            objective="min",
            bounds=bench_cycle._bounds(problem_data),
            name="cycle",
        )
    )

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.iterations <= 100

    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(problem_data["A_scipy"] @ x - problem_data["b"])))
    assert residual <= 1e-9
    assert abs(result.solution.objective_value - bench_cycle.EXPECTED_CYCLE_OBJECTIVE) <= 1e-5
