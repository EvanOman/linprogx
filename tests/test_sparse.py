from __future__ import annotations

import pytest

from linprogx import SparseLPProblem, Status, csr_matrix, solve_sparse, solve_sparse_canonical


def test_csr_matrix_operations() -> None:
    matrix = csr_matrix(
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


def test_sparse_solver_min_canonical() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
    g_ub = csr_matrix(3, 2, [0, 1, 2, 3], [0, 1, 0], [-1.0, -1.0, 1.0])

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


def test_sparse_problem_validation() -> None:
    matrix = csr_matrix(1, 2, [0, 1], [0], [1.0])

    with pytest.raises(ValueError, match="b_eq length"):
        SparseLPProblem([1.0, 2.0], matrix, [])


def test_solve_sparse_max_with_bounds() -> None:
    g_ub = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
    result = solve_sparse(
        SparseLPProblem(
            [3.0, 2.0],
            G_ub=g_ub,
            h_ub=[4.0],
            objective="max",
            bounds=[(0.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(10.0)
    assert result.solution.x == pytest.approx([2.0, 2.0])
