from __future__ import annotations

import pytest

from linprogx import (
    SparseLPProblem,
    SparseSolver,
    Status,
    csr_matrix,
    solve_sparse,
    solve_sparse_canonical,
)


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


@pytest.mark.parametrize(
    ("indptr", "match"),
    [
        ([0, 1], "indptr must contain 3 entries"),
        ([1, 1, 1], "indptr must start with 0"),
        ([0, 2, 1], "indptr must be nondecreasing"),
        ([0, 0, 0], r"indptr\[-1\] must equal nnz"),
    ],
)
def test_csr_matrix_rejects_bad_indptr(indptr: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        csr_matrix(2, 2, indptr, [0], [1.0])


@pytest.mark.parametrize("bad_index", [-1, 2])
def test_csr_matrix_rejects_column_indices_outside_width(bad_index: int) -> None:
    with pytest.raises(ValueError, match="column index out of range"):
        csr_matrix(1, 2, [0, 1], [bad_index], [1.0])


def test_csr_matrix_rejects_matvec_vector_width_mismatch() -> None:
    matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 2.0])

    with pytest.raises(ValueError, match="vector length must match matrix column count"):
        matrix.matvec([1.0])


def test_csr_matrix_rejects_transpose_matvec_vector_height_mismatch() -> None:
    matrix = csr_matrix(2, 1, [0, 1, 1], [0], [1.0])

    with pytest.raises(ValueError, match="vector length must match matrix row count"):
        matrix.transpose_matvec([1.0])


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


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        (
            SparseLPProblem(
                [1.0],
                A_eq=csr_matrix(1, 1, [0, 1], [0], [1.0]),
                b_eq=[1.0],
                objective="max",
            ),
            "expects minimization",
        ),
        (
            SparseLPProblem([1.0], bounds=[(0.0, 1.0)]),
            "expects equality constraints",
        ),
        (
            SparseLPProblem(
                [1.0],
                A_eq=csr_matrix(1, 1, [0, 1], [0], [1.0]),
                b_eq=[1.0],
                G_ub=csr_matrix(1, 1, [0, 1], [0], [1.0]),
                h_ub=[1.0],
            ),
            "expects bounds instead of G_ub",
        ),
    ],
)
def test_sparse_pdhg_rejects_unsupported_problem_shapes(
    problem: SparseLPProblem, message: str
) -> None:
    result = SparseSolver(algorithm="pdhg").solve(problem)

    assert result.solution.status == Status.INFEASIBLE
    assert message in result.solution.message


def test_sparse_pdhg_equality_bounds_path() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

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


def test_sparse_pdhg_respects_active_lower_bound() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

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


def test_sparse_pdhg_zero_iteration_uses_projected_zero_start() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, -1.0])

    result = SparseSolver(
        algorithm="pdhg",
        eps=1e-8,
        max_iterations=0,
        objective_scale=1.0,
        check_interval=1,
    ).solve(
        SparseLPProblem(
            [1.0, 1.0],
            A_eq=a_eq,
            b_eq=[0.0],
            objective="min",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
    )

    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(0.0)
    assert result.solution.x == pytest.approx([0.0, 0.0])


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


def test_normal_equations_solve_matches_dense_reference() -> None:
    # A = [[1, 0, 2], [0, 3, 1]], d = [1, 2, 0.5], delta = 1e-3.
    matrix = csr_matrix(2, 3, [0, 2, 4], [0, 2, 1, 2], [1.0, 2.0, 3.0, 1.0])
    d = [1.0, 2.0, 0.5]
    rhs = [1.0, -2.0]
    delta = 1e-3

    x = matrix.normal_equations_solve(d, rhs, delta)

    # dense ADA' = [[1*1+0.5*4, 0.5*2], [0.5*2, 2*9+0.5*1]] + delta I
    a11 = 3.0 + delta
    a12 = 1.0
    a22 = 18.5 + delta
    det = a11 * a22 - a12 * a12
    expected = [
        (a22 * rhs[0] - a12 * rhs[1]) / det,
        (a11 * rhs[1] - a12 * rhs[0]) / det,
    ]
    assert x == pytest.approx(expected, rel=1e-12)


def test_min_degree_returns_permutation() -> None:
    import importlib

    _csparse = importlib.import_module("linprogx._csparse")

    # arrow matrix pattern: dense first row/col plus diagonal
    indptr = [0, 5, 7, 9, 11, 13]
    indices = [0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4]
    order = _csparse.min_degree(indptr, indices)

    assert sorted(order) == [0, 1, 2, 3, 4]
    # the dense hub must not be eliminated while leaves remain cheaper
    assert order[0] != 0


def test_sparse_ipm_equality_bounds_path() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = SparseSolver(algorithm="ipm", eps=1e-9).solve(
        SparseLPProblem(
            [1.0, 2.0],
            A_eq=a_eq,
            b_eq=[3.0],
            objective="min",
            bounds=[(0.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0, abs=1e-6)
    assert result.solution.x == pytest.approx([2.0, 1.0], abs=1e-6)


def test_sparse_auto_routes_small_problems_to_ipm() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, -1.0])

    result = SparseSolver(algorithm="auto", eps=1e-9).solve(
        SparseLPProblem(
            [1.0, 1.0],
            A_eq=a_eq,
            b_eq=[0.0],
            objective="min",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
    )

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(0.0, abs=1e-7)
