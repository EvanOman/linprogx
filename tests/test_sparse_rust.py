"""Rust-backend mirror of the C sparse PDHG tests.

These exercise ``linprogx._rsparse.CSRMatrix`` through ``csr_matrix_rust`` so the
experimental Rust path stays drop-in compatible with the existing C extension.
"""
from __future__ import annotations

import pytest

from linprogx import (
    SparseLPProblem,
    SparseSolver,
    Status,
    csr_matrix_rust,
    solve_sparse_canonical,
)


def test_rust_csr_matrix_operations() -> None:
    matrix = csr_matrix_rust(
        3,
        4,
        [0, 2, 3, 5],
        [0, 2, 1, 0, 3],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    assert matrix.shape == (3, 4)
    assert matrix.nnz == 5
    assert matrix.density() == pytest.approx(5 / 12)
    assert matrix.matvec([1, 2, 3, 4]) == pytest.approx([7, 6, 24])
    assert matrix.transpose_matvec([1, 2, 3]) == pytest.approx([13, 6, 2, 15])
    assert matrix.to_dense() == [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 5.0],
    ]


@pytest.mark.parametrize(
    ("indptr", "match"),
    [
        ([0, 1], "indptr must contain 3 entries"),
        ([1, 1, 1], "indptr must start with 0"),
        ([0, 2, 1], "indptr must be nondecreasing"),
        ([0, 0, 0], r"indptr\[-1\] must equal nnz"),
    ],
)
def test_rust_csr_matrix_rejects_bad_indptr(indptr: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        csr_matrix_rust(2, 2, indptr, [0], [1.0])


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_rust_csr_matrix_rejects_column_indices_outside_width(bad_index: int) -> None:
    with pytest.raises(ValueError):
        csr_matrix_rust(1, 2, [0, 1], [bad_index], [1.0])


def test_rust_sparse_solver_min_canonical() -> None:
    a_eq = csr_matrix_rust(1, 2, [0, 2], [0, 1], [1.0, 1.0])
    g_ub = csr_matrix_rust(3, 2, [0, 1, 2, 3], [0, 1, 0], [-1.0, -1.0, 1.0])

    result = solve_sparse_canonical(
        [1.0, 2.0],
        a_eq,
        [3.0],
        g_ub,
        [0.0, 0.0, 2.0],
    )

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0)
    assert result.solution.x == pytest.approx([2.0, 1.0])


def test_rust_sparse_pdhg_equality_bounds_path() -> None:
    a_eq = csr_matrix_rust(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = SparseSolver(
        algorithm="pdhg",
        eps=1e-5,
        max_iterations=5_000,
        objective_scale=1.0,
        check_interval=5_000,
    ).solve(
        SparseLPProblem(
            [1.0, 2.0],
            A_eq=a_eq,
            b_eq=[3.0],
            objective="min",
            bounds=[(0.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0, abs=1e-3)
    assert result.solution.x == pytest.approx([2.0, 1.0], abs=1e-3)


def test_rust_sparse_pdhg_respects_active_lower_bound() -> None:
    a_eq = csr_matrix_rust(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = SparseSolver(
        algorithm="pdhg",
        eps=1e-5,
        max_iterations=5_000,
        objective_scale=1.0,
        check_interval=5_000,
    ).solve(
        SparseLPProblem(
            [2.0, 1.0],
            A_eq=a_eq,
            b_eq=[3.0],
            objective="min",
            bounds=[(1.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0, abs=1e-3)
    assert result.solution.x == pytest.approx([1.0, 2.0], abs=1e-3)
