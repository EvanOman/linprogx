from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from linprogx import (
    SparseLPProblem,
    SparseSolver,
    Status,
    csr_matrix,
    solve_sparse,
    solve_sparse_canonical,
)
from linprogx.sparse import _ipm_stall_risk


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


def test_sparse_pdhg_accepts_public_threads_option() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = SparseSolver(
        algorithm="pdhg",
        eps=1e-5,
        max_iterations=5_000,
        objective_scale=1.0,
        check_interval=5_000,
        threads=4,
    ).solve(
        SparseLPProblem(
            [1.0, 2.0],
            A_eq=a_eq,
            b_eq=[3.0],
            objective="min",
            bounds=[(0.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.backend == "native-c-sparse-pdhg"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0, abs=1e-3)


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


class _FallbackMatrix:
    shape = (1, 1)
    nnz = 1

    def __init__(self) -> None:
        self.pdhg_calls = 0
        self.dual_simplex_calls = 0

    def solve_eq_box_dual_simplex(self, *args: object, **kwargs: object) -> dict[str, object]:
        # the auto route tries a dual simplex rescue before keeping the
        # feasible IPM candidate; model it failing to certify
        self.dual_simplex_calls += 1
        return {
            "status": "iteration_limit",
            "objective": 0.0,
            "max_primal_residual": 1.0,
            "iterations": 5,
            "x": [0.0],
            "y": [0.0],
        }

    def solve_eq_box_ipm(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {
            "status": "iteration_limit",
            "objective": 1.0,
            "max_primal_residual": 0.0,
            "rel_primal_residual": 0.0,
            "rel_dual_residual": 1e-4,
            "mu": 1e-8,
            "iterations": 12,
            "x": [1.0],
            "y": [0.0],
        }

    def solve_eq_box_pdhg(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.pdhg_calls += 1
        return {
            "status": "iteration_limit",
            "objective": 0.0,
            "max_primal_residual": 9.0,
            "iterations": 50,
            "objective_scale": 1.0,
            "x": [10.0],
            "y": [0.0],
        }

    def matvec(self, x: list[float]) -> list[float]:
        return [x[0]]


class _PdhgThreadMatrix:
    shape = (1, 1)
    nnz = 1

    def __init__(self) -> None:
        self.threads: int | None = None

    def solve_eq_box_pdhg(self, *args: object, **kwargs: object) -> dict[str, object]:
        threads = kwargs["threads"]
        assert isinstance(threads, int)
        self.threads = threads
        return {
            "status": "optimal",
            "objective": 1.0,
            "max_primal_residual": 0.0,
            "iterations": 1,
            "objective_scale": 1.0,
            "x": [1.0],
            "y": [0.0],
        }

    def matvec(self, x: list[float]) -> list[float]:
        return [x[0]]


def test_auto_skips_pdhg_when_ipm_candidate_is_feasible_but_uncertified() -> None:
    matrix = _FallbackMatrix()

    result = SparseSolver(algorithm="auto", eps=1e-6, presolve=False).solve(
        SparseLPProblem(
            [1.0],
            A_eq=matrix,
            b_eq=[1.0],
            objective="min",
            bounds=[(0.0, None)],
        )
    )

    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status == Status.ITERATION_LIMIT
    assert result.solution.x == [1.0]
    assert "best feasible IPM candidate" in result.solution.message
    assert matrix.pdhg_calls == 0
    assert matrix.dual_simplex_calls == 1


def test_pdhg_public_route_defaults_to_auto_threads() -> None:
    matrix = _PdhgThreadMatrix()

    result = SparseSolver(algorithm="pdhg", eps=1e-6, presolve=False).solve(
        SparseLPProblem(
            [1.0],
            A_eq=matrix,
            b_eq=[1.0],
            objective="min",
            bounds=[(0.0, None)],
        )
    )

    assert result.solution.status == Status.OPTIMAL
    # 0 = auto: the C side sizes the worker pool to the physical-core
    # estimate (logical cores / 2, capped at the pool maximum).
    assert matrix.threads == 0


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


def test_sparse_dual_simplex_equality_bounds_path() -> None:
    a_eq = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = SparseSolver(algorithm="dual_simplex", eps=1e-9).solve(
        SparseLPProblem(
            [1.0, 2.0],
            A_eq=a_eq,
            b_eq=[3.0],
            objective="min",
            bounds=[(0.0, 2.0), (0.0, 3.0)],
        )
    )

    assert result.backend == "native-c-sparse-dual_simplex"
    assert result.solution.status == Status.OPTIMAL
    assert result.solution.objective_value == pytest.approx(4.0, abs=1e-6)
    assert result.solution.x == pytest.approx([2.0, 1.0], abs=1e-6)


def test_sparse_dual_simplex_matches_ipm_on_random_lp() -> None:
    import numpy as np
    import scipy.sparse

    from linprogx.sparse import from_scipy_sparse

    rng = np.random.default_rng(3)
    m, n = 12, 30
    dense = (
        scipy.sparse.random(
            m, n, density=0.3, random_state=rng, data_rvs=lambda s: rng.uniform(-2, 2, s)
        ).tocsr()
        + scipy.sparse.hstack(
            [scipy.sparse.identity(m), scipy.sparse.csr_matrix((m, n - m))]
        ).tocsr()
    )
    lo = np.zeros(n)
    hi = rng.uniform(0.5, 3.0, n)
    x0 = lo + (hi - lo) * rng.uniform(0, 1, n)
    b = (dense @ x0).tolist()
    c = rng.uniform(-2, 2, n).tolist()
    problem = SparseLPProblem(
        c,
        A_eq=from_scipy_sparse(scipy.sparse.csr_matrix(dense)),
        b_eq=b,
        objective="min",
        bounds=[(float(a), float(z)) for a, z in zip(lo, hi, strict=True)],
    )

    ds = SparseSolver(algorithm="dual_simplex", eps=1e-9).solve(problem)
    ipm = SparseSolver(algorithm="ipm", eps=1e-9).solve(problem)

    assert ds.solution.status == Status.OPTIMAL
    assert ipm.solution.status == Status.OPTIMAL
    assert ds.solution.objective_value == pytest.approx(
        ipm.solution.objective_value, rel=1e-6, abs=1e-6
    )


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


def test_pdhg_result_dict_exposes_diagnostics() -> None:
    matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = matrix.solve_eq_box_pdhg(
        [1.0, 2.0], [3.0], [0.0, 0.0], [2.0, 3.0], max_iter=5_000, tol=1e-6
    )

    expected = {
        "status",
        "objective",
        "max_primal_residual",
        "l2_primal_residual",
        "iterations",
        "operator_norm",
        "step_size",
        "objective_scale",
        "primal_weight",
        "dual_residual",
        "gap",
        "restarts",
        "step_trials",
        "plateau_exit",
        "x",
        "y",
    }
    assert expected <= set(result)
    assert result["status"] == "optimal"
    assert len(result["y"]) == 1


def test_pdhg_experiment_knobs_are_accepted() -> None:
    matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])

    result = matrix.solve_eq_box_pdhg(
        [1.0, 2.0],
        [3.0],
        [0.0, 0.0],
        [2.0, 3.0],
        max_iter=5_000,
        tol=1e-6,
        adaptive_weight=0,
        plateau_window=0,
        eval_interval_override=32,
        restart_sufficient=0.25,
        restart_necessary=0.75,
        restart_artificial=0.4,
    )

    assert result["status"] == "optimal"
    assert result["objective"] == pytest.approx(4.0, abs=1e-3)


def test_pdhg_threads_kwarg_bit_identical() -> None:
    # the threaded kernels write disjoint output ranges and sum
    # reductions in canonical order, so any thread count must produce
    # bit-identical iterates to the serial path
    m, n = 30, 60
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    state = 12345
    for row in range(m):
        col_set = {row}
        for _ in range(5):
            state = (state * 1103515245 + 12345) % 2**31
            col_set.add(state % n)
        for col in sorted(col_set):
            state = (state * 1103515245 + 12345) % 2**31
            indices.append(col)
            data.append(0.5 + (state % 1000) / 500.0)
        indptr.append(len(indices))
    matrix = csr_matrix(m, n, indptr, indices, data)
    x_feas = [0.5 + (j % 7) / 7.0 for j in range(n)]
    b = [
        sum(data[p] * x_feas[indices[p]] for p in range(indptr[i], indptr[i + 1])) for i in range(m)
    ]
    c = [1.0 + (j % 5) / 3.0 for j in range(n)]
    kwargs = dict(max_iter=300, tol=1e-12, check_interval=10**6)
    r1 = matrix.solve_eq_box_pdhg(c, b, [0.0] * n, [float("inf")] * n, **kwargs, threads=1)
    r4 = matrix.solve_eq_box_pdhg(c, b, [0.0] * n, [float("inf")] * n, **kwargs, threads=4)
    assert r1["iterations"] == r4["iterations"]
    assert r1["x"] == r4["x"]
    assert r1["y"] == r4["y"]


def test_pdhg_profile_env_emits_timing_summary(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
    monkeypatch.setenv("LINPROGX_PDHG_PROFILE", "1")

    result = matrix.solve_eq_box_pdhg(
        [1.0, 2.0],
        [3.0],
        [0.0, 0.0],
        [2.0, 3.0],
        max_iter=4,
        tol=1e-12,
        check_interval=4,
    )

    captured = capfd.readouterr()
    assert result["iterations"] == 4
    assert "pdhg profile:" in captured.err
    assert "iterations=4" in captured.err
    assert "trial_primal=" in captured.err
    assert "trial_dual=" in captured.err


def test_pdhg_thread_pool_grows_and_reports_capacity(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    matrix = csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0])
    monkeypatch.setenv("LINPROGX_PDHG_PROFILE", "1")
    kwargs = dict(max_iter=4, tol=1e-12, check_interval=4)

    matrix.solve_eq_box_pdhg(
        [1.0, 2.0],
        [3.0],
        [0.0, 0.0],
        [2.0, 3.0],
        **kwargs,
        threads=2,
    )
    capfd.readouterr()

    matrix.solve_eq_box_pdhg(
        [1.0, 2.0],
        [3.0],
        [0.0, 0.0],
        [2.0, 3.0],
        **kwargs,
        threads=4,
    )

    captured = capfd.readouterr()
    assert "threads=4" in captured.err
    # The pool is process-global and only ever grows: another solve in the
    # same process (e.g. an auto-threaded run on a many-core machine) may
    # already have grown it past this request, so assert capacity covers
    # the request rather than matching it exactly.
    match = re.search(r"pool_threads=(\d+)", captured.err)
    assert match is not None
    assert int(match.group(1)) >= 4


def test_pdhg_cleanup_stops_early_when_certificate_is_close() -> None:
    import numpy as np
    import scipy.sparse

    from linprogx.sparse import from_scipy_sparse

    rng = np.random.default_rng(0)
    rows, cols = 50, 160
    matrix_data = scipy.sparse.random(
        rows,
        cols,
        density=0.06,
        random_state=rng,
        data_rvs=lambda size: rng.uniform(-2.0, 2.0, size),
    ).tocsr()
    matrix_data = (
        matrix_data
        + scipy.sparse.hstack(
            [scipy.sparse.identity(rows), scipy.sparse.csr_matrix((rows, cols - rows))]
        ).tocsr()
    )
    x_feas = rng.uniform(0.0, 2.0, cols)
    x_feas[rng.random(cols) < 0.35] = 0.0
    b = matrix_data @ x_feas
    c = rng.uniform(0.1, 2.0, cols)
    matrix = from_scipy_sparse(matrix_data)

    result = matrix.solve_eq_box_pdhg(
        c.tolist(),
        b.tolist(),
        [0.0] * cols,
        [float("inf")] * cols,
        max_iter=900,
        tol=1e-5,
        check_interval=64,
        threads=1,
    )

    assert result["status"] == "optimal"
    assert result["iterations"] <= 640
    assert result["max_primal_residual"] <= 1e-5


# ---------------------------------------------------------------------------
# _ipm_stall_risk unit tests
# ---------------------------------------------------------------------------


class TestIpmStallRisk:
    """Unit tests for the structural stall-prediction signal.

    The signal fires when >= 50% of columns are one-sided with a cost
    that does not resist movement toward the infinite side (c_j <= 0
    for lo-only, c_j >= 0 for hi-only) AND the average column nnz is
    in [5, 8) -- network-ish sparsity with enough coupling.
    """

    def test_fires_on_zero_cost_lo_only_coupled(self) -> None:
        # 100 columns: 60 lo-only with c=0 (60% >= 50%), avg nnz 6 in [5,8)
        cols = 100
        c = [0.0] * 60 + [1.0] * 40
        lo = [0.0] * 100
        hi = [float("inf")] * 60 + [10.0] * 40
        nnz = 600  # avg 6.0
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is True

    def test_fires_on_negative_cost_lo_only_coupled(self) -> None:
        # 100 columns: 55 lo-only with c<0 (55% >= 50%), avg nnz 5.5
        cols = 100
        c = [-1.0] * 55 + [1.0] * 45
        lo = [0.0] * 100
        hi = [float("inf")] * 55 + [10.0] * 45
        nnz = 550  # avg 5.5
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is True

    def test_fires_on_hi_only_zero_cost_coupled(self) -> None:
        # 100 columns: 60 hi-only with c=0 (60% >= 50%), avg nnz 7
        cols = 100
        c = [0.0] * 60 + [-1.0] * 40
        lo = [float("-inf")] * 60 + [0.0] * 40
        hi = [10.0] * 60 + [float("inf")] * 40
        nnz = 700  # avg 7.0
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is True

    def test_does_not_fire_when_all_columns_boxed(self) -> None:
        # All columns have both finite bounds -> no at-risk columns
        cols = 100
        c = [0.0] * 100
        lo = [0.0] * 100
        hi = [10.0] * 100
        nnz = 600
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False

    def test_does_not_fire_when_at_risk_below_threshold(self) -> None:
        # 100 columns: 40 at-risk (40% < 50%), avg nnz 6
        cols = 100
        c = [0.0] * 40 + [1.0] * 60
        lo = [0.0] * 100
        hi = [float("inf")] * 40 + [10.0] * 60
        nnz = 600
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False

    def test_does_not_fire_when_avg_col_nnz_too_high(self) -> None:
        # at-risk 60% but avg nnz = 10 >= 8
        cols = 100
        c = [0.0] * 60 + [1.0] * 40
        lo = [0.0] * 100
        hi = [float("inf")] * 60 + [10.0] * 40
        nnz = 1000  # avg 10.0
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False

    def test_does_not_fire_when_avg_col_nnz_too_low(self) -> None:
        # at-risk 60% but avg nnz = 3 < 5 -- too sparse for coupling
        cols = 100
        c = [0.0] * 60 + [1.0] * 40
        lo = [0.0] * 100
        hi = [float("inf")] * 60 + [10.0] * 40
        nnz = 300  # avg 3.0
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False

    def test_does_not_fire_when_cost_resists_infinity(self) -> None:
        # lo-only with c>0 -> cost resists movement toward infinity
        cols = 100
        c = [1.0] * 100
        lo = [0.0] * 100
        hi = [float("inf")] * 100
        nnz = 600
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False

    def test_empty_columns_returns_false(self) -> None:
        assert _ipm_stall_risk([], [], [], 0, 0) is False

    def test_exact_50_percent_boundary(self) -> None:
        # Exactly 50% at-risk = 50 out of 100
        cols = 100
        c = [0.0] * 50 + [1.0] * 50
        lo = [0.0] * 100
        hi = [float("inf")] * 50 + [10.0] * 50
        nnz = 600
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is True

    def test_just_below_50_percent(self) -> None:
        # 49 at-risk out of 100 = 49% < 50%
        cols = 100
        c = [0.0] * 49 + [1.0] * 51
        lo = [0.0] * 100
        hi = [float("inf")] * 49 + [10.0] * 51
        nnz = 600
        assert _ipm_stall_risk(c, lo, hi, nnz, cols) is False


# ---------------------------------------------------------------------------
# Stall-predictor routing test
# ---------------------------------------------------------------------------


class _StallPredictorMatrix:
    """Mock matrix that triggers the stall predictor signal.

    Models a 50x200 problem with avg col nnz = 6.0 (in the [5,8) coupling
    band) and >= 50% at-risk one-sided columns.
    """

    shape = (50, 200)
    nnz = 1200  # avg col nnz = 6.0

    def __init__(self) -> None:
        self.ipm_calls = 0
        self.ds_calls = 0
        self.ds2_calls = 0
        self.pdhg_calls = 0

    def solve_eq_box_ds2(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.ds2_calls += 1
        return {
            "status": "optimal",
            "objective": -5.0,
            "max_primal_residual": 0.0,
            "iterations": 100,
            "x": [1.0] * 200,
            "y": [0.0] * 50,
        }

    def solve_eq_box_dual_simplex(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.ds_calls += 1
        return {
            "status": "optimal",
            "objective": -5.0,
            "max_primal_residual": 0.0,
            "iterations": 100,
            "x": [1.0] * 200,
            "y": [0.0] * 50,
        }

    def solve_eq_box_ipm(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.ipm_calls += 1
        return {
            "status": "iteration_limit",
            "objective": -4.0,
            "max_primal_residual": 1e-3,
            "rel_primal_residual": 1e-3,
            "rel_dual_residual": 1e-3,
            "mu": 1e-6,
            "iterations": 200,
            "x": [0.5] * 200,
            "y": [0.0] * 50,
        }

    def solve_eq_box_pdhg(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.pdhg_calls += 1
        return {
            "status": "iteration_limit",
            "objective": 0.0,
            "max_primal_residual": 1.0,
            "iterations": 50,
            "objective_scale": 1.0,
            "x": [0.0] * 200,
            "y": [0.0] * 50,
        }

    def matvec(self, x: list[float]) -> list[float]:
        # Identity-like: return first 50 elements as Ax
        return list(x[:50])

    def to_components(self) -> tuple[list[int], list[int], list[float]]:
        # Minimal stub; not used in routing
        return [], [], []


def test_stall_predictor_routes_to_ds_before_ipm() -> None:
    """Without a qualifying aggregation, the shipped DS rescue is preserved."""
    matrix = _StallPredictorMatrix()

    # Build a problem that triggers the signal: 120 lo-only columns with c=0
    # (60% of 200 columns >= 50% threshold), avg col nnz 6.0 in [5,8).
    n = 200
    m = 50
    c = [0.0] * 120 + [1.0] * 80  # 120 at-risk lo-only (c<=0)
    lo = [0.0] * n
    hi = [float("inf")] * 120 + [10.0] * 80
    bounds: list[tuple[float | None, float | None]] = [
        (lo[j], hi[j] if hi[j] != float("inf") else None) for j in range(n)
    ]
    b = [1.0] * m

    result = SparseSolver(algorithm="auto", eps=1e-6, presolve=False).solve(
        SparseLPProblem(c, A_eq=matrix, b_eq=b, objective="min", bounds=bounds)
    )

    assert result.backend == "native-c-sparse-dual-simplex"
    assert result.solution.status == Status.OPTIMAL
    assert "stall predictor" in result.solution.message
    assert matrix.ds2_calls == 0
    assert matrix.ds_calls == 1
    assert matrix.ipm_calls == 0
    assert matrix.pdhg_calls == 0


def test_qualifying_aggressive_aggregation_routes_to_ds2(monkeypatch) -> None:
    matrix = _StallPredictorMatrix()
    seen = {"aggressive": False}
    n = 200
    c = [0.0] * 120 + [1.0] * 80
    lo = [0.0] * n
    hi = [float("inf")] * 120 + [10.0] * 80
    reduction = SimpleNamespace(
        _matrix=matrix,
        _reduction_counts={},
        c=c,
        b=[1.0] * 50,
        lo=lo,
        hi=hi,
        removed_rows=0,
        removed_cols=0,
    )

    def fake_presolve(*args: object, **kwargs: object) -> object:
        return reduction

    def fake_aggressive(value: object) -> object:
        assert value is reduction
        seen["aggressive"] = True
        return reduction

    monkeypatch.setattr("linprogx.sparse.presolve_matrix", fake_presolve)
    monkeypatch.setattr("linprogx.sparse.aggressive_aggregate_for_ds2", fake_aggressive)
    monkeypatch.setattr("linprogx.sparse.postsolve_x", lambda x, _reduction: x)
    problem = SparseLPProblem(
        c, A_eq=matrix, b_eq=[1.0] * 50, bounds=list(zip(lo, hi, strict=True))
    )

    SparseSolver(algorithm="auto", eps=1e-6).solve(problem)

    assert seen["aggressive"] is True
    assert matrix.ds2_calls == 1
    assert matrix.ds_calls == 0


class _StallPredictorFailMatrix(_StallPredictorMatrix):
    """Like _StallPredictorMatrix but DS returns iteration_limit, so
    the code should fall through to the normal IPM path."""

    def solve_eq_box_dual_simplex(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.ds_calls += 1
        return {
            "status": "iteration_limit",
            "objective": 0.0,
            "max_primal_residual": 1.0,
            "iterations": 50000,
            "x": [0.0] * 200,
            "y": [0.0] * 50,
        }


def test_stall_predictor_falls_through_when_ds_fails() -> None:
    """When DS fails to certify, the normal IPM path runs."""
    matrix = _StallPredictorFailMatrix()

    n = 200
    m = 50
    c = [0.0] * 120 + [1.0] * 80
    lo = [0.0] * n
    hi = [float("inf")] * 120 + [10.0] * 80
    bounds: list[tuple[float | None, float | None]] = [
        (lo[j], hi[j] if hi[j] != float("inf") else None) for j in range(n)
    ]
    b = [1.0] * m

    SparseSolver(algorithm="auto", eps=1e-6, presolve=False).solve(
        SparseLPProblem(c, A_eq=matrix, b_eq=b, objective="min", bounds=bounds)
    )

    # DS was tried first but failed; IPM should have been attempted
    assert matrix.ds_calls >= 1
    assert matrix.ipm_calls >= 1
