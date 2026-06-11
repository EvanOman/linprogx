from __future__ import annotations

import pytest

from linprogx.presolve import postsolve_x, presolve_eq_box
from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix
from linprogx.types import Status

INF = float("inf")


def test_presolve_returns_none_when_nothing_reduces() -> None:
    # One dense 3-nonzero row: no empty, singleton, or doubleton reductions.
    reduction = presolve_eq_box(
        1,
        3,
        [0, 3],
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        [3.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [INF, INF, INF],
    )

    assert reduction is None


def test_presolve_fixes_singleton_rows_and_cascades() -> None:
    # x0 = 2 (singleton); the second row then becomes a singleton for x1.
    reduction = presolve_eq_box(
        2,
        2,
        [0, 1, 3],
        [0, 0, 1],
        [1.0, 1.0, 1.0],
        [2.0, 5.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [INF, INF],
    )

    assert reduction is not None
    assert reduction.rows == 0
    assert reduction.cols == 0
    assert reduction.objective_offset == pytest.approx(5.0)

    x = postsolve_x([], reduction)
    assert x == pytest.approx([2.0, 3.0])


def test_presolve_doubleton_maps_bounds_and_objective() -> None:
    # x0 + x1 = 3 with x0 in [0, 2]: eliminating x0 forces x1 in [1, 3].
    reduction = presolve_eq_box(
        1,
        2,
        [0, 2],
        [0, 1],
        [1.0, 1.0],
        [3.0],
        [1.0, 2.0],
        [0.0, 0.0],
        [2.0, 3.0],
    )

    assert reduction is not None
    assert reduction.rows == 0
    assert reduction.cols == 1
    assert reduction.lo == pytest.approx([1.0])
    assert reduction.hi == pytest.approx([3.0])
    # c1' = c1 - c0 = 1 and offset = c0 * 3 = 3 keep the objective identical.
    assert reduction.c == pytest.approx([1.0])
    assert reduction.objective_offset == pytest.approx(3.0)

    x = postsolve_x([1.0], reduction)
    assert x == pytest.approx([2.0, 1.0])


def test_sparse_pdhg_presolve_matches_unpresolved_solution() -> None:
    # Chain with a doubleton head: x0 + x1 = 4, x1 + x2 + x3 = 6.
    problem = SparseLPProblem(
        c=[3.0, 1.0, 1.0, 2.0],
        A_eq=csr_matrix(2, 4, [0, 2, 5], [0, 1, 1, 2, 3], [1.0, 1.0, 1.0, 1.0, 1.0]),
        b_eq=[4.0, 6.0],
        objective="min",
        bounds=[(0.0, 3.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0)],
        name="presolve-roundtrip",
    )

    solved = {}
    for presolve in (True, False):
        result = SparseSolver(
            algorithm="pdhg",
            eps=1e-6,
            max_iterations=20_000,
            check_interval=20_000,
            presolve=presolve,
        ).solve(problem)
        assert result.solution.status == Status.OPTIMAL
        solved[presolve] = result.solution

    assert solved[True].objective_value == pytest.approx(solved[False].objective_value, abs=1e-4)
    assert solved[True].x == pytest.approx(solved[False].x, abs=1e-3)
    assert "presolve removed" in solved[True].message


def test_sparse_pdhg_presolve_handles_fully_reduced_problem() -> None:
    # Two singleton rows determine both variables; nothing is left to solve.
    problem = SparseLPProblem(
        c=[1.0, 1.0],
        A_eq=csr_matrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0]),
        b_eq=[1.5, 4.0],
        objective="min",
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        name="fully-reduced",
    )

    result = SparseSolver(
        algorithm="pdhg",
        eps=1e-8,
        max_iterations=100,
        check_interval=100,
    ).solve(problem)

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.x == pytest.approx([1.5, 2.0])
    assert result.solution.objective_value == pytest.approx(3.5)


def test_presolve_doubleton_positive_alpha_intersects_bounds() -> None:
    # x0 - x1 = 0 means x0 = x1, so x1 inherits x0's tighter upper bound.
    reduction = presolve_eq_box(
        1,
        2,
        [0, 2],
        [0, 1],
        [1.0, -1.0],
        [0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [2.0, 5.0],
    )

    assert reduction is not None
    assert reduction.rows == 0
    assert reduction.cols == 1
    assert reduction.lo == pytest.approx([0.0])
    assert reduction.hi == pytest.approx([2.0])

    x = postsolve_x([1.5], reduction)
    assert x == pytest.approx([1.5, 1.5])


def test_presolve_skips_extreme_coefficient_ratios() -> None:
    # ratio 1e6 in both orientations exceeds the guard; row must survive.
    reduction = presolve_eq_box(
        1,
        2,
        [0, 2],
        [0, 1],
        [1.0, 1e6],
        [1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [10.0, 10.0],
    )

    assert reduction is None


def test_presolve_respects_fill_limit() -> None:
    # x0 appears in 7 other rows: eliminating it via the doubleton would
    # exceed max_fill=5, so with the default the doubleton row survives;
    # with a raised limit it is eliminated.
    indptr = [0, 2]
    indices = [0, 1]
    data = [1.0, 1.0]
    extra_rows = 7
    for r in range(extra_rows):
        indices.extend([0, 2 + r])
        data.extend([1.0, 1.0])
        indptr.append(len(indices))
    rows = 1 + extra_rows
    cols = 2 + extra_rows
    b = [1.0] * rows
    c = [1.0] * cols
    lo = [0.0] * cols
    hi = [10.0] * cols

    limited = presolve_eq_box(
        rows, cols, list(indptr), list(indices), list(data), list(b), list(c), list(lo), list(hi)
    )
    raised = presolve_eq_box(
        rows,
        cols,
        list(indptr),
        list(indices),
        list(data),
        list(b),
        list(c),
        list(lo),
        list(hi),
        max_fill=20,
    )

    # x1 has degree 1 so the doubleton eliminates x1 even under the limit,
    # but x0 (degree 8) must never be the eliminated side at max_fill=5.
    assert limited is not None
    assert raised is not None
    assert limited.cols <= cols - 1
    assert raised.cols <= limited.cols


def test_presolve_cascade_doubleton_creates_singleton() -> None:
    # Row 0: x0 + x1 = 4 (doubleton). Row 1: x0 + x1 + x2 = 6.
    # Eliminating the doubleton turns row 1 into a singleton fixing x2.
    reduction = presolve_eq_box(
        2,
        3,
        [0, 2, 5],
        [0, 1, 0, 1, 2],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [4.0, 6.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction.rows == 0
    assert reduction.cols == 1

    x = postsolve_x([2.5], reduction)
    assert x[0] + x[1] == pytest.approx(4.0)
    assert x[0] + x[1] + x[2] == pytest.approx(6.0)
    assert x[2] == pytest.approx(2.0)


def test_presolve_drops_consistent_empty_rows() -> None:
    reduction = presolve_eq_box(
        2,
        2,
        [0, 0, 2],
        [0, 1],
        [1.0, 1.0],
        [0.0, 3.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [5.0, 5.0],
    )

    assert reduction is not None
    assert reduction.removed_rows >= 1
    assert reduction.rows <= 1
