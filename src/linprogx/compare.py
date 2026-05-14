from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from linprogx.solver import Solver
from linprogx.types import LPProblem, Solution, Status


@dataclass(frozen=True)
class Comparison:
    name: str
    linprogx_status: str
    solver_status: str
    linprogx_objective: float | None
    solver_objective: float | None
    objective_delta: float | None
    linprogx_seconds: float
    solver_seconds: float
    solver_name: str


def compare_with_scipy(problem: LPProblem, *, repeats: int = 1) -> Comparison:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover
        msg = "SciPy is required for comparison: uv sync --extra dev"
        raise RuntimeError(msg) from exc

    own_result: Solution | None = None
    own_start = time.perf_counter()
    for _ in range(repeats):
        own_result = Solver().solve(problem)
    own_seconds = (time.perf_counter() - own_start) / repeats
    assert own_result is not None

    c = list(problem.c)
    if problem.objective == "max":
        c = [-value for value in c]

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for constraint in problem.constraints:
        if constraint.sense == "<=":
            a_ub.append(constraint.coefficients)
            b_ub.append(constraint.rhs)
        elif constraint.sense == ">=":
            a_ub.append([-value for value in constraint.coefficients])
            b_ub.append(-constraint.rhs)
        else:
            a_eq.append(constraint.coefficients)
            b_eq.append(constraint.rhs)

    scipy_result = None
    scipy_start = time.perf_counter()
    for _ in range(repeats):
        scipy_result = linprog(
            c,
            A_ub=a_ub or None,
            b_ub=b_ub or None,
            A_eq=a_eq or None,
            b_eq=b_eq or None,
            bounds=problem.bounds or [(0, None)] * len(problem.c),
            method="highs",
        )
    scipy_seconds = (time.perf_counter() - scipy_start) / repeats
    assert scipy_result is not None

    scipy_status = _scipy_status(int(scipy_result.status))
    scipy_objective = None
    if scipy_status == "optimal":
        scipy_objective = float(scipy_result.fun)
        if problem.objective == "max":
            scipy_objective = -scipy_objective

    delta = None
    if own_result.objective_value is not None and scipy_objective is not None:
        delta = abs(own_result.objective_value - scipy_objective)

    return Comparison(
        name=problem.name,
        linprogx_status=own_result.status.value,
        solver_status=scipy_status,
        linprogx_objective=own_result.objective_value,
        solver_objective=scipy_objective,
        objective_delta=delta,
        linprogx_seconds=own_seconds,
        solver_seconds=scipy_seconds,
        solver_name="scipy-highs",
    )


def compare_with_clarabel(problem: LPProblem, *, repeats: int = 1) -> Comparison:
    try:
        import clarabel
        import numpy as np
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover
        msg = "Clarabel and SciPy are required for comparison: uv sync --extra dev"
        raise RuntimeError(msg) from exc

    own_result: Solution | None = None
    own_start = time.perf_counter()
    for _ in range(repeats):
        own_result = Solver().solve(problem)
    own_seconds = (time.perf_counter() - own_start) / repeats
    assert own_result is not None

    rows, rhs, zero_count, nonnegative_count = _clarabel_rows(problem)
    n = len(problem.c)
    q = np.array(
        [-value if problem.objective == "max" else value for value in problem.c], dtype=float
    )
    p = sparse.csc_matrix((n, n), dtype=float)
    a = sparse.csc_matrix(np.array(rows, dtype=float)) if rows else sparse.csc_matrix((0, n))
    b = np.array(rhs, dtype=float)
    clarabel_api = vars(clarabel)
    zero_cone = clarabel_api["ZeroConeT"]
    nonnegative_cone = clarabel_api["NonnegativeConeT"]
    default_settings = clarabel_api["DefaultSettings"]
    default_solver = clarabel_api["DefaultSolver"]
    cones = []
    if zero_count:
        cones.append(zero_cone(zero_count))
    if nonnegative_count:
        cones.append(nonnegative_cone(nonnegative_count))
    settings: Any = default_settings()
    settings.verbose = False

    clarabel_result: Any = None
    clarabel_start = time.perf_counter()
    for _ in range(repeats):
        solver = default_solver(p, q, a, b, cones, settings)
        clarabel_result = solver.solve()
    clarabel_seconds = (time.perf_counter() - clarabel_start) / repeats
    assert clarabel_result is not None

    clarabel_status = _clarabel_status(str(clarabel_result.status))
    clarabel_objective = None
    if clarabel_status == Status.OPTIMAL.value:
        clarabel_objective = float(clarabel_result.obj_val)
        if problem.objective == "max":
            clarabel_objective = -clarabel_objective

    delta = None
    if own_result.objective_value is not None and clarabel_objective is not None:
        delta = abs(own_result.objective_value - clarabel_objective)

    return Comparison(
        name=problem.name,
        linprogx_status=own_result.status.value,
        solver_status=clarabel_status,
        linprogx_objective=own_result.objective_value,
        solver_objective=clarabel_objective,
        objective_delta=delta,
        linprogx_seconds=own_seconds,
        solver_seconds=clarabel_seconds,
        solver_name="clarabel",
    )


def assert_matches_scipy(problem: LPProblem, *, objective_tol: float = 1e-7) -> None:
    comparison = compare_with_scipy(problem)
    if comparison.solver_status == Status.ITERATION_LIMIT.value:
        return
    assert comparison.linprogx_status == comparison.solver_status
    if comparison.solver_status == Status.OPTIMAL.value:
        assert comparison.objective_delta is not None
        assert comparison.objective_delta <= objective_tol


def assert_matches_clarabel(
    problem: LPProblem, *, objective_tol: float = 1e-5, relative_tol: float = 1e-9
) -> None:
    comparison = compare_with_clarabel(problem)
    if comparison.solver_status == Status.ITERATION_LIMIT.value:
        return
    assert comparison.linprogx_status == comparison.solver_status
    if comparison.solver_status == Status.OPTIMAL.value:
        assert comparison.objective_delta is not None
        scale = max(1.0, abs(comparison.linprogx_objective or 0.0))
        assert comparison.objective_delta <= max(objective_tol, relative_tol * scale)


def _clarabel_rows(problem: LPProblem) -> tuple[list[list[float]], list[float], int, int]:
    equality_rows: list[list[float]] = []
    equality_rhs: list[float] = []
    inequality_rows: list[list[float]] = []
    inequality_rhs: list[float] = []

    for constraint in problem.constraints:
        if constraint.sense == "=":
            equality_rows.append(list(constraint.coefficients))
            equality_rhs.append(constraint.rhs)
        elif constraint.sense == "<=":
            inequality_rows.append(list(constraint.coefficients))
            inequality_rhs.append(constraint.rhs)
        else:
            inequality_rows.append([-value for value in constraint.coefficients])
            inequality_rhs.append(-constraint.rhs)

    bounds = problem.bounds or [(0, None)] * len(problem.c)
    for index, (lower, upper) in enumerate(bounds):
        if lower is not None:
            row = [0.0] * len(problem.c)
            row[index] = -1.0
            inequality_rows.append(row)
            inequality_rhs.append(-float(lower))
        if upper is not None:
            row = [0.0] * len(problem.c)
            row[index] = 1.0
            inequality_rows.append(row)
            inequality_rhs.append(float(upper))

    return (
        equality_rows + inequality_rows,
        equality_rhs + inequality_rhs,
        len(equality_rows),
        len(inequality_rows),
    )


def _scipy_status(status: int) -> str:
    return {
        0: Status.OPTIMAL.value,
        1: Status.ITERATION_LIMIT.value,
        2: Status.INFEASIBLE.value,
        3: Status.UNBOUNDED.value,
    }.get(status, f"scipy_status_{status}")


def _clarabel_status(status: str) -> str:
    normalized = status.lower()
    if normalized in {"solved", "almostsolved"}:
        return Status.OPTIMAL.value
    if normalized in {"primalinfeasible", "almostprimalinfeasible"}:
        return Status.INFEASIBLE.value
    if normalized in {"dualinfeasible", "almostdualinfeasible"}:
        return Status.UNBOUNDED.value
    if normalized in {"maxiterations", "maxtime"}:
        return Status.ITERATION_LIMIT.value
    return f"clarabel_{normalized}"
