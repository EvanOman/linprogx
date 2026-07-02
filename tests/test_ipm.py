from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.sparse

from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix, from_scipy_sparse
from linprogx.types import Status

_csparse = importlib.import_module("linprogx._csparse")


def _dense_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Tiny Gaussian elimination reference for the Cholesky tests."""
    size = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for r in range(size):
            if r == col:
                continue
            factor = augmented[r][col] / pivot
            for k in range(col, size + 1):
                augmented[r][k] -= factor * augmented[col][k]
    return [augmented[i][size] / augmented[i][i] for i in range(size)]


def _normal_matrix(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    d: list[float],
    delta: float,
) -> list[list[float]]:
    dense = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for p in range(indptr[i], indptr[i + 1]):
            dense[i][indices[p]] = data[p]
    out = [[0.0] * rows for _ in range(rows)]
    for i in range(rows):
        for j in range(rows):
            out[i][j] = sum(dense[i][t] * d[t] * dense[j][t] for t in range(cols))
        out[i][i] += delta
    return out


class TestNormalEquationsSolve:
    def test_diagonal_matrix(self) -> None:
        matrix = csr_matrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])

        x = matrix.normal_equations_solve([1.0, 1.0], [2.0, 8.0], 0.0)

        assert x == pytest.approx([2.0, 2.0], rel=1e-12)

    def test_matches_dense_reference_on_overlapping_rows(self) -> None:
        rows, cols = 3, 5
        indptr = [0, 3, 6, 9]
        indices = [0, 1, 4, 1, 2, 3, 0, 3, 4]
        data = [1.0, -2.0, 0.5, 3.0, 1.5, -1.0, 2.0, 4.0, -0.5]
        d = [1.0, 0.5, 2.0, 1.5, 3.0]
        rhs = [1.0, -2.0, 3.0]
        delta = 1e-6
        matrix = csr_matrix(rows, cols, indptr, indices, data)

        x = matrix.normal_equations_solve(d, rhs, delta)

        reference = _dense_solve(_normal_matrix(rows, cols, indptr, indices, data, d, delta), rhs)
        assert x == pytest.approx(reference, rel=1e-10)

    def test_regularization_handles_rank_deficiency(self) -> None:
        # duplicate rows make A D A' singular without the delta term
        matrix = csr_matrix(2, 2, [0, 2, 4], [0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0])
        delta = 1e-6

        x = matrix.normal_equations_solve([1.0, 1.0], [1.0, 1.0], delta)

        reference = _dense_solve([[2.0 + delta, 2.0], [2.0, 2.0 + delta]], [1.0, 1.0])
        assert x == pytest.approx(reference, rel=1e-8)

    def test_repeated_calls_are_deterministic(self) -> None:
        matrix = csr_matrix(2, 3, [0, 2, 3], [0, 1, 2], [1.0, 2.0, 3.0])

        first = matrix.normal_equations_solve([1.0, 2.0, 3.0], [1.0, 2.0], 1e-8)
        second = matrix.normal_equations_solve([1.0, 2.0, 3.0], [1.0, 2.0], 1e-8)

        assert first == second


class TestMinDegree:
    def test_chain_pattern(self) -> None:
        # tridiagonal chain of 5 nodes
        indptr = [0, 2, 5, 8, 11, 13]
        indices = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]

        order = _csparse.min_degree(indptr, indices)

        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_complete_graph(self) -> None:
        size = 4
        indptr = [size * i for i in range(size + 1)]
        indices = [j for _ in range(size) for j in range(size)]

        order = _csparse.min_degree(indptr, indices)

        assert sorted(order) == list(range(size))

    def test_single_node(self) -> None:
        assert _csparse.min_degree([0, 1], [0]) == [0]

    def test_empty_pattern(self) -> None:
        assert _csparse.min_degree([0], []) == []

    def test_diagonal_only(self) -> None:
        indptr = [0, 1, 2, 3]
        indices = [0, 1, 2]

        order = _csparse.min_degree(indptr, indices)

        assert sorted(order) == [0, 1, 2]


class TestSolveEqBoxIpm:
    def test_equality_bounds_known_solution(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm(
            [1.0, 2.0], [3.0], [0.0, 0.0], [2.0, 3.0], max_iter=60, tol=1e-9
        )

        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(4.0, abs=1e-6)
        assert result["x"] == pytest.approx([2.0, 1.0], abs=1e-6)
        assert result["max_primal_residual"] < 1e-8

    def test_supernodal_kwarg_uses_ipm_path(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm(
            [1.0, 2.0],
            [3.0],
            [0.0, 0.0],
            [2.0, 3.0],
            max_iter=60,
            tol=1e-9,
            supernodal=True,
        )

        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(4.0, abs=1e-6)
        assert result["x"] == pytest.approx([2.0, 1.0], abs=1e-6)

    def test_respects_active_lower_bound(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm(
            [2.0, 1.0], [3.0], [1.0, 0.0], [2.0, 3.0], max_iter=60, tol=1e-9
        )

        assert result["status"] == "optimal"
        assert result["x"] == pytest.approx([1.0, 2.0], abs=1e-6)

    def test_free_variable_follows_constraint(self) -> None:
        # min x1 subject to x0 + x1 = 2 with x0 in [0, 1] and x1 free.
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
        inf = float("inf")

        result = matrix.solve_eq_box_ipm(
            [0.0, 1.0], [2.0], [0.0, -inf], [1.0, inf], max_iter=60, tol=1e-9
        )

        assert result["status"] == "optimal"
        assert result["x"] == pytest.approx([1.0, 1.0], abs=1e-6)

    def test_upper_bound_only_variable(self) -> None:
        # max x0 (min -x0) subject to x0 + x1 = 4, x0 <= 3, x1 in [0, 5].
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
        inf = float("inf")

        result = matrix.solve_eq_box_ipm(
            [-1.0, 0.0], [4.0], [-inf, 0.0], [3.0, 5.0], max_iter=60, tol=1e-9
        )

        assert result["status"] == "optimal"
        assert result["x"] == pytest.approx([3.0, 1.0], abs=1e-6)

    def test_zero_width_box_is_pinned(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm(
            [1.0, 1.0], [3.0], [1.0, 0.0], [1.0, 5.0], max_iter=60, tol=1e-9
        )

        assert result["status"] == "optimal"
        assert result["x"] == pytest.approx([1.0, 2.0], abs=1e-6)

    def test_dual_vector_for_simple_equality(self) -> None:
        # min x0 subject to x0 = 5: the equality multiplier is 1.
        matrix = csr_matrix(1, 1, [0, 1], [0], [1.0])

        result = matrix.solve_eq_box_ipm([1.0], [5.0], [0.0], [float("inf")])

        assert result["status"] == "optimal"
        assert result["x"] == pytest.approx([5.0], abs=1e-6)
        assert result["y"] == pytest.approx([1.0], abs=1e-6)

    def test_iteration_limit_status(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm(
            [1.0, 2.0], [3.0], [0.0, 0.0], [2.0, 3.0], max_iter=1, tol=1e-12
        )

        assert result["status"] == "iteration_limit"

    def test_result_dict_keys(self) -> None:
        matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

        result = matrix.solve_eq_box_ipm([1.0, 2.0], [3.0], [0.0, 0.0], [2.0, 3.0])

        expected = {
            "status",
            "objective",
            "max_primal_residual",
            "rel_primal_residual",
            "rel_dual_residual",
            "mu",
            "iterations",
            "x",
            "y",
        }
        assert expected <= set(result)


class TestSolverRouting:
    def test_ipm_backend_reported(self) -> None:
        result = SparseSolver(algorithm="ipm", eps=1e-9).solve(
            SparseLPProblem(
                [1.0, 2.0],
                A_eq=csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0]),
                b_eq=[3.0],
                objective="min",
                bounds=[(0.0, 2.0), (0.0, 3.0)],
            )
        )

        assert result.backend == "native-c-sparse-ipm"
        assert result.solution.status == Status.OPTIMAL

    def test_auto_routes_large_problems_to_pdhg(self) -> None:
        size = SparseSolver.AUTO_IPM_MAX_ROWS + 1
        matrix = csr_matrix(size, size, list(range(size + 1)), list(range(size)), [1.0] * size)
        result = SparseSolver(
            algorithm="auto",
            eps=1e-6,
            max_iterations=2_000,
            presolve=False,
        ).solve(
            SparseLPProblem(
                [1.0] * size,
                A_eq=matrix,
                b_eq=[1.0] * size,
                objective="min",
                bounds=[(0.0, None)] * size,
            )
        )

        assert result.backend == "native-c-sparse-pdhg"
        assert result.solution.status == Status.OPTIMAL
        assert result.solution.objective_value == pytest.approx(size, rel=1e-4)

    def test_auto_falls_back_to_pdhg_when_ipm_hits_limit(self) -> None:
        result = SparseSolver(
            algorithm="auto",
            eps=1e-9,
            max_iterations=1,
            presolve=False,
        ).solve(
            SparseLPProblem(
                [1.0, 2.0],
                A_eq=csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0]),
                b_eq=[3.0],
                objective="min",
                bounds=[(0.0, 2.0), (0.0, 3.0)],
            )
        )

        assert result.backend == "native-c-sparse-pdhg"

    def test_ipm_rejects_unsupported_shapes_like_pdhg(self) -> None:
        result = SparseSolver(algorithm="ipm").solve(SparseLPProblem([1.0], bounds=[(0.0, 1.0)]))

        assert result.solution.status == Status.INFEASIBLE
        assert "expects equality constraints" in result.solution.message

    def test_ipm_handles_fully_presolved_problem(self) -> None:
        # two singleton rows fix both variables; nothing reaches the IPM
        result = SparseSolver(algorithm="ipm", eps=1e-9).solve(
            SparseLPProblem(
                [1.0, 1.0],
                A_eq=csr_matrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0]),
                b_eq=[1.5, 4.0],
                objective="min",
                bounds=[(0.0, 5.0), (0.0, 5.0)],
            )
        )

        assert result.solution.status == Status.OPTIMAL
        assert result.solution.x == pytest.approx([1.5, 2.0])


class TestAlgorithmEquivalence:
    @pytest.mark.parametrize(
        ("c", "indptr", "indices", "data", "rows", "cols", "b", "bounds"),
        [
            (
                [1.0, 2.0, 0.5],
                [0, 3, 5],
                [0, 1, 2, 1, 2],
                [1.0, 1.0, 1.0, 1.0, -1.0],
                2,
                3,
                [4.0, 0.0],
                [(0.0, 3.0), (0.0, 3.0), (0.0, 3.0)],
            ),
            (
                [3.0, 1.0, 1.0, 2.0],
                [0, 2, 5],
                [0, 1, 1, 2, 3],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                2,
                4,
                [4.0, 6.0],
                [(0.0, 3.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0)],
            ),
        ],
    )
    def test_ipm_pdhg_and_simplex_agree(
        self,
        c: list[float],
        indptr: list[int],
        indices: list[int],
        data: list[float],
        rows: int,
        cols: int,
        b: list[float],
        bounds: list[tuple[float | None, float | None]],
    ) -> None:
        problem = SparseLPProblem(
            c,
            A_eq=csr_matrix(rows, cols, indptr, indices, data),
            b_eq=b,
            objective="min",
            bounds=bounds,
        )

        objectives = {}
        for algorithm in ("ipm", "pdhg", "simplex"):
            result = SparseSolver(
                algorithm=algorithm,
                eps=1e-7 if algorithm != "simplex" else 1e-9,
                max_iterations=50_000,
                check_interval=5_000,
            ).solve(problem)
            assert result.solution.status == Status.OPTIMAL, algorithm
            objectives[algorithm] = result.solution.objective_value

        assert objectives["ipm"] == pytest.approx(objectives["simplex"], abs=1e-5)
        assert objectives["pdhg"] == pytest.approx(objectives["simplex"], abs=1e-4)


class TestOrderingBudget:
    def test_min_degree_budget_abort_returns_none(self) -> None:
        # complete graph on 60 nodes with a 100-op budget: must abort
        size = 60
        indptr = [size * i for i in range(size + 1)]
        indices = [j for _ in range(size) for j in range(size)]

        assert _csparse.min_degree(indptr, indices, 100) is None

    def test_min_degree_unlimited_budget_completes(self) -> None:
        size = 60
        indptr = [size * i for i in range(size + 1)]
        indices = [j for _ in range(size) for j in range(size)]

        order = _csparse.min_degree(indptr, indices, 0)

        assert order is not None
        assert sorted(order) == list(range(size))


def test_auto_routes_medium_sparse_problems_to_ipm() -> None:
    # 5000-row staircase: above the old 4000-row cap, trivially cheap factor.
    size = 5_000
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(size):
        if i == 0:
            indices.append(0)
            data.append(1.0)
        else:
            indices.extend([i - 1, i])
            data.extend([0.5, 1.0])
        indptr.append(len(indices))
    matrix = csr_matrix(size, size, indptr, indices, data)

    result = SparseSolver(algorithm="auto", eps=1e-9, max_iterations=2_000, presolve=False).solve(
        SparseLPProblem(
            [1.0] * size,
            A_eq=matrix,
            b_eq=[1.0] * size,
            objective="min",
            bounds=[(0.0, None)] * size,
        )
    )

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.OPTIMAL


def test_normal_equations_solve_with_dense_column() -> None:
    # 80 rows; one column touching every row triggers the dense-column
    # splitting path; verify against the plain dense reference.
    m = 80
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(m):
        # diagonal sparse column i, plus the shared dense column m
        indices.extend([i, m])
        data.extend([1.0 + 0.01 * i, 0.5 + 0.001 * i])
        indptr.append(len(indices))
    matrix = csr_matrix(m, m + 1, indptr, indices, data)
    d = [1.0 + 0.005 * j for j in range(m + 1)]
    rhs = [((j * 7919) % 13) - 6.0 for j in range(m)]
    delta = 1e-8

    x = matrix.normal_equations_solve(d, rhs, delta)

    dense = [[0.0] * (m + 1) for _ in range(m)]
    for i in range(m):
        for p in range(indptr[i], indptr[i + 1]):
            dense[i][indices[p]] = data[p]
    reference = _dense_solve(_normal_matrix(m, m + 1, indptr, indices, data, d, delta), rhs)
    assert x == pytest.approx(reference, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("m", "n", "density"), [(200, 400, 0.25), (300, 600, 0.15)])
def test_normal_equations_dense_tail_matches_dense_reference(
    m: int, n: int, density: float
) -> None:
    # systems large/dense enough that the dense-tail factorization path
    # activates (tail >= 64 columns clearing the flop break-even)
    rng = np.random.default_rng(9)
    A = (
        scipy.sparse.random(
            m, n, density=density, random_state=rng, data_rvs=lambda s: rng.uniform(-2, 2, s)
        ).tocsr()
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    )
    matrix = from_scipy_sparse(scipy.sparse.csr_matrix(A))
    d = rng.uniform(0.5, 2.0, n)
    rhs = rng.uniform(-1, 1, m)
    delta = 1e-8
    out = np.array(matrix.normal_equations_solve(d.tolist(), rhs.tolist(), delta))
    G = (A @ scipy.sparse.diags(d) @ A.T).toarray() + delta * np.eye(m)
    residual = np.max(np.abs(G @ out - rhs))
    assert residual <= 1e-10


def test_normal_equations_supernodal_factor_matches_dense_reference() -> None:
    rng = np.random.default_rng(15)
    m, n = 48, 120
    A = (
        scipy.sparse.random(
            m,
            n,
            density=0.08,
            random_state=rng,
            data_rvs=lambda size: rng.uniform(-1.5, 1.5, size),
        ).tocsr()
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    )
    matrix = from_scipy_sparse(scipy.sparse.csr_matrix(A))
    d = rng.uniform(0.25, 2.0, n)
    rhs = rng.uniform(-1, 1, m)
    delta = 1e-7

    out = np.array(matrix.normal_equations_solve(d.tolist(), rhs.tolist(), delta, True))

    G = (A @ scipy.sparse.diags(d) @ A.T).toarray() + delta * np.eye(m)
    residual = np.max(np.abs(G @ out - rhs))
    assert residual <= 1e-10


def test_supernodal_profile_reports_single_thread_blas(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(16)
    m, n = 48, 120
    A = (
        scipy.sparse.random(
            m,
            n,
            density=0.08,
            random_state=rng,
            data_rvs=lambda size: rng.uniform(-1.5, 1.5, size),
        ).tocsr()
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    )
    matrix = from_scipy_sparse(scipy.sparse.csr_matrix(A))
    d = rng.uniform(0.25, 2.0, n)
    rhs = rng.uniform(-1, 1, m)
    monkeypatch.setenv("LINPROGX_SUPERNODAL_PROFILE", "1")

    matrix.normal_equations_solve(d.tolist(), rhs.tolist(), 1e-7, True)

    captured = capfd.readouterr()
    assert "supernodal profile:" in captured.err
    assert "blas_threads=1" in captured.err


def test_ipm_threads_kwarg_bit_identical() -> None:
    # the threaded tail GEMM partitions rows in 4-aligned chunks so each
    # output element is computed wholly by one thread in the same order
    rng = np.random.default_rng(21)
    m, n = 220, 440
    A = (
        scipy.sparse.random(
            m, n, density=0.2, random_state=rng, data_rvs=lambda s: rng.uniform(-2, 2, s)
        ).tocsr()
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    )
    matrix = from_scipy_sparse(scipy.sparse.csr_matrix(A))
    b = (A @ rng.uniform(0.5, 1.5, n)).tolist()
    c = rng.uniform(0.5, 2.0, n).tolist()
    lo = [0.0] * n
    hi = [float("inf")] * n
    r1 = matrix.solve_eq_box_ipm(c, b, lo, hi, max_iter=60, tol=1e-9, threads=1)
    r4 = matrix.solve_eq_box_ipm(c, b, lo, hi, max_iter=60, tol=1e-9, threads=4)
    assert r1["x"] == r4["x"]
    assert r1["y"] == r4["y"]


def test_ipm_blas_and_floored_tail_agree_on_residual() -> None:
    # the BLAS dpotrf tail and the floored hand-kernel tail factor the
    # same system; both must reach the same objective regardless of
    # which kernel runs (summation order differs, so not bitwise equal).
    # Sparse columns keep the normal equations factorable.
    m, n = 900, 1800
    rows: list[int] = list(range(m))
    cols: list[int] = list(range(m))
    data: list[float] = [1.0] * m
    state = 999
    for j in range(n):
        for _ in range(3):
            state = (state * 1103515245 + 12345) % 2**31
            rows.append(state % m)
            cols.append(j)
            state = (state * 1103515245 + 12345) % 2**31
            data.append(0.2 + (state % 100) / 200.0)
    A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    matrix = from_scipy_sparse(A)
    rng = np.random.default_rng(33)
    b = (A @ rng.uniform(0.5, 1.5, n)).tolist()
    c = rng.uniform(0.5, 2.0, n).tolist()
    lo = [0.0] * n
    hi = [float("inf")] * n
    r_blas = matrix.solve_eq_box_ipm(c, b, lo, hi, max_iter=200, tol=1e-9, blas=True)
    r_hand = matrix.solve_eq_box_ipm(c, b, lo, hi, max_iter=200, tol=1e-9, blas=False)
    assert r_blas["status"] == r_hand["status"]
    if r_blas["status"] == "optimal":
        assert r_blas["objective"] == pytest.approx(r_hand["objective"], rel=1e-6, abs=1e-6)


def _build_sparse_csr(m: int, n: int, per_col: int, seed: int) -> scipy.sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    state = seed
    for j in range(n):
        for _ in range(per_col):
            state = (state * 1103515245 + 12345) % 2**31
            rows.append(state % m)
            cols.append(j)
            state = (state * 1103515245 + 12345) % 2**31
            data.append(1.0 + (state % 50) / 50.0)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n))


def _reference_supernode_symbolic(
    A: scipy.sparse.csr_matrix,
) -> list[tuple[int, list[int], list[tuple[int, list[int], list[int], list[int], list[int]]]]]:
    G = (A @ A.T).tocsc()
    G.setdiag(1.0)
    G.eliminate_zeros()
    perm = np.array(_csparse.min_degree(G.indptr.tolist(), G.indices.tolist()))
    Gp = G[perm][:, perm].tocsc()
    m = Gp.shape[0]
    parent = np.full(m, -1, dtype=np.int64)
    ancestor = np.full(m, -1, dtype=np.int64)
    for k in range(m):
        for p in range(Gp.indptr[k], Gp.indptr[k + 1]):
            i = int(Gp.indices[p])
            while i != -1 and i < k:
                nxt = int(ancestor[i])
                ancestor[i] = k
                if nxt == -1:
                    parent[i] = k
                i = nxt

    col_rows: list[np.ndarray] = []
    below = [set[int]() for _ in range(m)]
    for k in range(m):
        reach: set[int] = set()
        for p in range(Gp.indptr[k], Gp.indptr[k + 1]):
            i = int(Gp.indices[p])
            while i < k and i not in reach:
                reach.add(i)
                i = int(parent[i])
        for i in reach:
            below[i].add(k)
        below[k].add(k)
    for rows in below:
        col_rows.append(np.array(sorted(rows), dtype=np.int64))

    nchild = np.zeros(m, dtype=np.int64)
    for p in parent:
        if p >= 0:
            nchild[p] += 1
    starts: list[int] = []
    for j in range(m):
        merges_previous = (
            j > 0
            and parent[j - 1] == j
            and nchild[j] == 1
            and len(col_rows[j - 1]) == len(col_rows[j]) + 1
        )
        if not merges_previous:
            starts.append(j)
    starts.append(m)

    out: list[
        tuple[int, list[int], list[tuple[int, list[int], list[int], list[int], list[int]]]]
    ] = []
    snode_rows = [col_rows[starts[s]].tolist() for s in range(len(starts) - 1)]
    for s, rows_s_list in enumerate(snode_rows):
        j0, j1 = starts[s], starts[s + 1]
        rows_s = np.array(rows_s_list, dtype=np.int64)
        updates: list[tuple[int, list[int], list[int], list[int], list[int]]] = []
        for source in range(s):
            k0, k1 = starts[source], starts[source + 1]
            width = k1 - k0
            rows_k = np.array(snode_rows[source], dtype=np.int64)
            pivot_source_positions: list[int] = []
            pivot_columns: list[int] = []
            target_source_positions: list[int] = []
            target_row_positions: list[int] = []
            for pos in range(width, len(rows_k)):
                row = int(rows_k[pos])
                if j0 <= row < j1:
                    pivot_source_positions.append(pos)
                    pivot_columns.append(row - j0)
                target_pos = int(np.searchsorted(rows_s, row))
                if target_pos < len(rows_s) and int(rows_s[target_pos]) == row:
                    target_source_positions.append(pos)
                    target_row_positions.append(target_pos)
            if pivot_source_positions and target_source_positions:
                updates.append(
                    (
                        source,
                        pivot_source_positions,
                        pivot_columns,
                        target_source_positions,
                        target_row_positions,
                    )
                )
        out.append((j0, rows_s_list, updates))
    return out


def test_supernode_partition_covers_all_columns() -> None:
    # the fundamental supernode sizes must tile the columns exactly
    A = _build_sparse_csr(40, 160, 4, seed=11)
    sizes = from_scipy_sparse(A).supernode_sizes()
    assert sum(sizes) == A.shape[0]
    assert all(w >= 1 for w in sizes)


def test_supernode_partition_block_diagonal_doubles() -> None:
    # two independent blocks: every column still accounted for, and the
    # partition does not merge across the disconnected blocks
    block = _build_sparse_csr(16, 60, 4, seed=7)
    blk = scipy.sparse.bmat([[block, None], [None, block]]).tocsr()
    sizes = from_scipy_sparse(blk).supernode_sizes()
    assert sum(sizes) == blk.shape[0] == 32
    single = from_scipy_sparse(block).supernode_sizes()
    # the doubled system has at least as many supernodes as one block
    assert len(sizes) >= len(single)


def test_supernode_partition_on_cre_a_fixture() -> None:
    # a real instance: every column tiled, and amalgamation actually
    # happens (some supernode wider than one column)
    from pathlib import Path

    from scipy.io import loadmat

    path = Path(__file__).parent / "data" / "lp_cre_a.mat"
    raw = loadmat(path)["Problem"][0, 0]
    A = raw["A"].tocsr().astype(float)
    sizes = from_scipy_sparse(A).supernode_sizes()
    assert sum(sizes) == A.shape[0]
    assert max(sizes) > 1


def test_supernode_symbolic_updates_match_reference() -> None:
    rng = np.random.default_rng(1)
    m, n = 32, 96
    random_part = scipy.sparse.random(
        m,
        n,
        density=0.06,
        random_state=rng,
        data_rvs=lambda size: rng.uniform(0.5, 2.0, size),
    ).tocsr()
    A = (
        random_part
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    ).tocsr()

    expected = _reference_supernode_symbolic(A)
    actual = from_scipy_sparse(A).supernode_symbolic_structure()

    assert actual == expected
    assert sum(len(updates) for _, _, updates in actual) > 0
