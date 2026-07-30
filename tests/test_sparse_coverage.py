"""Focused coverage for sparse Python validation, routing, and simplex helpers."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

import linprogx.sparse as sparse
from linprogx import SparseLPProblem, SparseSolver, Status, csr_matrix
from linprogx.sparse import _SparseTableau


class _Matrix:
    """Small scripted equality matrix for Python routing characterization."""

    shape = (1, 1)
    nnz = 1

    def __init__(
        self,
        *,
        ipm: list[dict[str, Any]] | None = None,
        dual_simplex: dict[str, Any] | None = None,
        pdhg: dict[str, Any] | None = None,
        matvec: Callable[[list[float]], list[float]] | None = None,
    ) -> None:
        self.ipm_results = list(ipm or [])
        self.dual_simplex_result = dual_simplex
        self.pdhg_result = pdhg
        self._matvec = matvec or (lambda x: list(x))
        self.ipm_kwargs: list[dict[str, Any]] = []

    def matvec(self, x: list[float]) -> list[float]:
        return self._matvec(x)

    def solve_eq_box_ipm(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        self.ipm_kwargs.append(kwargs)
        return self.ipm_results.pop(0)

    def solve_eq_box_dual_simplex(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        assert self.dual_simplex_result is not None
        return self.dual_simplex_result

    def solve_eq_box_pdhg(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        assert self.pdhg_result is not None
        return self.pdhg_result


def _result(
    status: str,
    x: list[float],
    *,
    objective: float = 0.0,
    iterations: int = 3,
    ipm_slice_us: object | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "x": x,
        "objective": objective,
        "iterations": iterations,
    }
    if ipm_slice_us is not None:
        result["ipm_slice_us"] = ipm_slice_us
    return result


def _problem(matrix: Any, *, b: float = 1.0) -> SparseLPProblem:
    return SparseLPProblem([1.0], A_eq=matrix, b_eq=[b], bounds=[(0.0, None)])


def _reduction(matrix: Any | None) -> SimpleNamespace:
    return SimpleNamespace(
        _matrix=matrix,
        _reduction_counts={},
        rows=1,
        cols=1,
        indptr=[0, 1],
        indices=[0],
        data=[1.0],
        c=[1.0],
        b=[1.0],
        lo=[0.0],
        hi=[float("inf")],
        removed_rows=0,
        removed_cols=0,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SparseLPProblem([]), "objective must contain"),
        (
            lambda: SparseLPProblem([1.0], A_eq=csr_matrix(1, 2, [0, 1], [0], [1.0]), b_eq=[1.0]),
            "A_eq column count",
        ),
        (
            lambda: SparseLPProblem([1.0], G_ub=csr_matrix(1, 2, [0, 1], [0], [1.0]), h_ub=[1.0]),
            "G_ub column count",
        ),
        (
            lambda: SparseLPProblem([1.0], G_ub=csr_matrix(1, 1, [0, 1], [0], [1.0]), h_ub=[]),
            "h_ub length",
        ),
        (lambda: SparseLPProblem([1.0], bounds=[]), "bounds width"),
        (lambda: SparseSolver(threads=-1), "threads must be nonnegative"),
    ],
)
def test_sparse_public_validation_errors(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_csr_matrix_reports_missing_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse, "CSRMatrix", None)

    with pytest.raises(RuntimeError, match="extension is not available"):
        sparse.csr_matrix(0, 0, [0], [], [])


def test_simplex_handles_constraint_free_models() -> None:
    unbounded = SparseSolver().solve(SparseLPProblem([1.0], objective="max"))
    bounded = SparseSolver().solve(SparseLPProblem([2.0], objective="min", bounds=[(3.0, None)]))

    assert unbounded.solution.status == Status.UNBOUNDED
    assert bounded.solution.status == Status.OPTIMAL
    assert bounded.solution.x == [3.0]
    assert bounded.solution.objective_value == 6.0


def test_simplex_reports_phase_one_iteration_limit_and_infeasibility() -> None:
    equality = csr_matrix(1, 1, [0, 1], [0], [1.0])
    iteration_limit = SparseSolver(max_iterations=0).solve(
        SparseLPProblem([0.0], A_eq=equality, b_eq=[1.0])
    )
    infeasible = SparseSolver().solve(
        SparseLPProblem([0.0], A_eq=equality, b_eq=[1.0], bounds=[(0.0, 0.0)])
    )

    assert iteration_limit.solution.status == Status.ITERATION_LIMIT
    assert iteration_limit.solution.message == "phase I hit the iteration limit"
    assert infeasible.solution.status == Status.INFEASIBLE
    assert infeasible.solution.message == "phase I found no feasible basis"


def test_simplex_reports_phase_two_unbounded() -> None:
    equality = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, -1.0])

    result = SparseSolver().solve(
        SparseLPProblem([1.0, 0.0], A_eq=equality, b_eq=[0.0], objective="max")
    )

    assert result.solution.status == Status.UNBOUNDED
    assert result.solution.message == "objective is unbounded"


def test_simplex_rejects_reversed_bounds_and_prepares_free_upper_bound() -> None:
    solver = SparseSolver()

    with pytest.raises(ValueError, match="upper bound is lower"):
        solver.solve(SparseLPProblem([1.0], bounds=[(2.0, 1.0)]))

    prepared = solver._prepare(SparseLPProblem([1.0], bounds=[(None, 4.0)]))
    assert prepared.c_max == [-1.0, 1.0]
    assert prepared.rows == [{0: 1.0, 1: -1.0}]
    assert prepared.rhs == [4.0]


def test_simplex_builds_surplus_and_artificial_columns_for_negative_rhs() -> None:
    matrix = csr_matrix(1, 1, [0, 1], [0], [1.0])
    solver = SparseSolver()
    prepared = solver._prepare(SparseLPProblem([0.0], G_ub=matrix, h_ub=[-1.0], objective="max"))

    tableau = solver._build_tableau(prepared)

    assert tableau.basis == [2]
    assert tableau.artificial == {2}
    assert tableau.rows[0] == {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0}


def test_simplex_pivot_rejects_zero_and_removes_artificial_rows() -> None:
    solver = SparseSolver()
    zero_pivot = _SparseTableau([{0: 0.0, 1: 1.0}, {1: 0.0}], [0], set(), 1, 1)

    with pytest.raises(ZeroDivisionError, match="pivot value is too close"):
        solver._pivot(zero_pivot, 0, 0)

    tableau = _SparseTableau(
        [
            {0: 2.0, 2: 1.0, 3: 4.0},
            {2: 1.0, 3: 0.0},
            {3: 0.0},
        ],
        [2, 2],
        {2},
        3,
        1,
    )
    solver._remove_artificial_columns(tableau)

    assert tableau.artificial == set()
    assert tableau.basis == [0]
    assert tableau.rows == [{0: 1.0, 3: 2.0}, {3: 0.0}]


def test_presolve_rebuilds_matrix_from_components(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _Matrix()
    pdhg_result = _result("optimal", [1.0], objective=1.0)
    pdhg_result["objective_scale"] = 1.0
    rebuilt = _Matrix(pdhg=pdhg_result)
    reduction = _reduction(None)
    seen: list[tuple[object, ...]] = []

    monkeypatch.setattr(sparse, "presolve_matrix", lambda *args, **kwargs: reduction)
    monkeypatch.setattr(sparse, "postsolve_x", lambda x, reduction: x)

    def fake_csr(*args: object) -> _Matrix:
        seen.append(args)
        return rebuilt

    monkeypatch.setattr(sparse, "csr_matrix", fake_csr)

    result = SparseSolver(algorithm="pdhg").solve(_problem(original))

    assert result.solution.status == Status.OPTIMAL
    assert seen == [(1, 1, [0, 1], [0], [1.0])]


def test_ipm_retries_raw_infeasible_optimum_with_floored_kernel() -> None:
    matrix = _Matrix(
        ipm=[
            _result("optimal", [0.0]),
            _result(
                "optimal",
                [1.0],
                objective=1.0,
                iterations=7,
                ipm_slice_us={"factor": 2},
            ),
        ]
    )

    result = SparseSolver(algorithm="ipm", presolve=False, eps=1e-8).solve(_problem(matrix))

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.x == [1.0]
    assert "floored retry" in result.solution.message
    assert result.ipm_slice_us == {"factor": 2.0}
    assert matrix.ipm_kwargs[1]["blas"] is False


def test_ipm_uses_unpresolved_retry_after_reduced_retry_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _Matrix(ipm=[_result("optimal", [1.0], objective=9.0, iterations=11)])
    reduced = _Matrix(
        ipm=[
            _result("iteration_limit", [0.0]),
            _result("optimal", [0.0]),
        ]
    )
    reduction = _reduction(reduced)
    postsolve_calls: list[list[float]] = []

    monkeypatch.setattr(sparse, "presolve_matrix", lambda *args, **kwargs: reduction)

    def postsolve(x: list[float], reduction: object) -> list[float]:
        postsolve_calls.append(x)
        return x

    monkeypatch.setattr(sparse, "postsolve_x", postsolve)

    result = SparseSolver(algorithm="ipm", eps=1e-8).solve(_problem(original))

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == 9.0
    assert "unpresolved floored retry" in result.solution.message
    assert postsolve_calls == [[0.0], [0.0]]
    assert original.ipm_kwargs == [
        {
            "max_iter": 200,
            "tol": 1e-09,
            "threads": 0,
            "blas": False,
            "feas_tol": 1e-08,
        }
    ]


def test_auto_uses_dual_simplex_rescue_after_ipm_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _Matrix(ipm=[_result("iteration_limit", [0.0])])
    reduced = _Matrix(
        ipm=[
            _result("iteration_limit", [0.0]),
            _result("iteration_limit", [0.0]),
        ],
        dual_simplex=_result("optimal", [1.0], objective=1.0, iterations=5),
    )
    reduction = _reduction(reduced)
    monkeypatch.setattr(sparse, "presolve_matrix", lambda *args, **kwargs: reduction)
    monkeypatch.setattr(sparse, "postsolve_x", lambda x, reduction: x)

    result = SparseSolver(algorithm="auto", eps=1e-8).solve(_problem(original))

    assert result.backend == "native-c-sparse-dual-simplex"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.x == [1.0]
    assert "certified after the IPM stalled" in result.solution.message


def test_ipm_keeps_earlier_feasible_candidate_when_final_residual_is_worse() -> None:
    residual_calls = 0

    def changing_matvec(x: list[float]) -> list[float]:
        nonlocal residual_calls
        residual_calls += 1
        return [1.0] if residual_calls == 1 else [3.0]

    matrix = _Matrix(
        ipm=[
            _result("iteration_limit", [1.0], objective=1.0, iterations=4),
            _result("iteration_limit", [1.0], objective=1.0, iterations=4),
        ],
        matvec=changing_matvec,
    )

    result = SparseSolver(algorithm="ipm", presolve=False, eps=1e-8).solve(_problem(matrix))

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.ITERATION_LIMIT
    assert result.solution.x == [1.0]
    assert "best feasible IPM candidate after fallback failed" in result.solution.message
