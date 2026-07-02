"""Tests for the IPM's min-norm dual cleanup stage.

Degenerate instances can exit the IPM with an excellent primal point
whose Lagrangian certificate fails on a small set of wrong-signed
reduced costs. The cleanup applies a min-norm dual correction that
zeroes those signs; it may gain a certificate but must never fake one.

The cre_a fixture (tests/data/lp_cre_a.mat) is the LPnetlib instance
from sparse.tamu.edu (public benchmark data); it is the smallest
real-world instance known to exercise the cleanup path.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import loadmat

from linprogx.presolve import postsolve_x, presolve_eq_box
from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix, from_scipy_sparse

CRE_A_PATH = Path(__file__).parent / "data" / "lp_cre_a.mat"
CRE_A_PUBLISHED_OBJECTIVE = 23595407.06  # Gurobi at 1e-8 (github.com/SkyLiu0/NETLIB)


def load_cre_a():
    raw = loadmat(CRE_A_PATH)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return (
        raw["A"].tocsr().astype(float),
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def test_dual_cleanup_certifies_cre_a_ipm() -> None:
    A, b, c, lo, hi = load_cre_a()
    reduction = presolve_eq_box(
        A.shape[0],
        A.shape[1],
        A.indptr.tolist(),
        A.indices.tolist(),
        A.data.tolist(),
        b.tolist(),
        c.tolist(),
        lo.tolist(),
        hi.tolist(),
    )
    assert reduction is not None
    matrix = csr_matrix(
        reduction.rows, reduction.cols, reduction.indptr, reduction.indices, reduction.data
    )
    result = matrix.solve_eq_box_ipm(
        reduction.c,
        reduction.b,
        reduction.lo,
        reduction.hi,
        max_iter=200,
        tol=1e-9,
        feas_tol=2e-5,
    )

    assert result["status"] == "optimal"
    assert result["dual_cleanup_rounds"] >= 1
    # Dual cleanup should fire in-loop once residuals are small enough;
    # waiting for post-exit cleanup burns several extra factorizations.
    assert result["iterations"] <= 36
    x = np.array(postsolve_x([float(value) for value in result["x"]], reduction))
    rel_err = abs(float(c @ x) - CRE_A_PUBLISHED_OBJECTIVE) / (1.0 + abs(CRE_A_PUBLISHED_OBJECTIVE))
    assert rel_err <= 1e-5
    assert float(np.max(np.abs(A @ x - b))) <= 2e-5

    # independent soundness audit of the returned dual point: any
    # wrong-signed reduced cost must be tiny (the certificate's 1e-9
    # scaled tolerance maps to <=1e-6 relative in raw units), and the
    # Lagrangian bound built from the well-signed terms must close the
    # gap to the primal objective
    y = np.array(result["y"])
    reduced_a = sp.csr_matrix(
        (reduction.data, reduction.indices, reduction.indptr),
        shape=(reduction.rows, reduction.cols),
    )
    r = np.array(reduction.c) - reduced_a.T @ y
    cscale = 1.0 + np.abs(reduction.c)
    reduced_lo = np.array(reduction.lo)
    reduced_hi = np.array(reduction.hi)
    inf_hi = reduced_hi >= 1e308
    inf_lo = reduced_lo <= -1e308
    bad_neg = (r < 0) & inf_hi
    bad_pos = (r > 0) & inf_lo
    assert np.all(np.abs(r[bad_neg]) <= 1e-6 * cscale[bad_neg])
    assert np.all(np.abs(r[bad_pos]) <= 1e-6 * cscale[bad_pos])
    dobj = float(np.array(reduction.b) @ y)
    dobj += float(np.sum(np.where((r > 0) & ~inf_lo, r * reduced_lo, 0.0)))
    dobj += float(np.sum(np.where((r < 0) & ~inf_hi, r * reduced_hi, 0.0)))
    pobj = float(np.array(reduction.c) @ np.array(result["x"]))
    assert abs(pobj - dobj) / (1.0 + abs(pobj)) <= 2e-5


def test_dual_cleanup_certifies_cre_a_auto_path() -> None:
    A, b, c, lo, hi = load_cre_a()
    bounds = [
        (lb if lb > -1e308 else None, ub if ub < 1e308 else None)
        for lb, ub in zip(lo, hi, strict=True)
    ]
    result = SparseSolver(algorithm="auto", eps=2e-5).solve(
        SparseLPProblem(
            c=c.tolist(),
            A_eq=from_scipy_sparse(A),
            b_eq=b.tolist(),
            objective="min",
            bounds=bounds,
            name="cre_a",
        )
    )
    assert result.solution.status.value == "optimal"
    obj = float(c @ np.array(result.solution.x))
    assert abs(obj - CRE_A_PUBLISHED_OBJECTIVE) / (1.0 + abs(CRE_A_PUBLISHED_OBJECTIVE)) <= 1e-4


def test_auto_ipm_polishes_raw_feasibility_before_falling_back() -> None:
    A, b, c, lo, hi = load_cre_a()
    bounds = [
        (lb if lb > -1e308 else None, ub if ub < 1e308 else None)
        for lb, ub in zip(lo, hi, strict=True)
    ]
    result = SparseSolver(algorithm="auto", eps=1e-9).solve(
        SparseLPProblem(
            c=c.tolist(),
            A_eq=from_scipy_sparse(A),
            b_eq=b.tolist(),
            objective="min",
            bounds=bounds,
            name="cre_a",
        )
    )

    x = np.array(result.solution.x)
    residual = float(np.max(np.abs(A @ x - b)))
    assert result.backend == "native-c-sparse-ipm"
    assert result.solution.status.value == "optimal"
    assert residual <= 1e-9


def test_dual_cleanup_idle_on_clean_problem() -> None:
    # a well-conditioned LP converges without the cleanup stage firing
    rng = np.random.default_rng(7)
    m, n = 20, 40
    A = (
        sp.random(m, n, density=0.4, random_state=rng).tocsr()
        + sp.hstack([sp.identity(m), sp.csr_matrix((m, n - m))]).tocsr() * 0.5
    )
    x_feas = rng.uniform(0.5, 1.5, n)
    b = A @ x_feas
    c = rng.uniform(0.5, 2.0, n)
    matrix = from_scipy_sparse(sp.csr_matrix(A))
    result = matrix.solve_eq_box_ipm(
        c.tolist(),
        b.tolist(),
        [0.0] * n,
        [np.inf] * n,
        max_iter=200,
        tol=1e-9,
    )
    assert result["status"] == "optimal"
    assert result["dual_cleanup_rounds"] == 0


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_dual_cleanup_never_fakes_optimality_on_degenerate_lps(seed: int) -> None:
    # duplicated columns with shared costs create a degenerate optimal
    # face; whenever the IPM claims optimal, the objective must agree
    # with an independent solver
    from scipy.optimize import linprog

    rng = np.random.default_rng(seed)
    m, k = 25, 20
    base = sp.random(
        m, k, density=0.35, random_state=rng, data_rvs=lambda s: rng.uniform(-2, 2, s)
    ).tocsc()
    A = sp.hstack([base, base[:, : k // 2], sp.identity(m)], format="csr")
    n = A.shape[1]
    cb = rng.uniform(0.1, 1.0, k)
    c = np.concatenate([cb, cb[: k // 2], np.full(m, 5.0)])
    b = A @ np.abs(rng.uniform(0.1, 1.0, n))
    matrix = from_scipy_sparse(A)
    result = matrix.solve_eq_box_ipm(
        c.tolist(), b.tolist(), [0.0] * n, [np.inf] * n, max_iter=200, tol=1e-9
    )
    reference = linprog(c, A_eq=A.toarray(), b_eq=b, bounds=[(0, None)] * n)
    assert reference.status == 0
    if result["status"] == "optimal":
        assert result["objective"] == pytest.approx(reference.fun, rel=1e-5, abs=1e-7)


def test_ipm_debug_kwarg_accepted() -> None:
    # debug=True streams per-iteration diagnostics to stderr; the call
    # must succeed and produce the same result shape
    rng = np.random.default_rng(3)
    m, n = 10, 20
    A = sp.random(m, n, density=0.5, random_state=rng).tocsr() + sp.eye(m, n) * 0.5
    b = A @ rng.uniform(0.5, 1.5, n)
    c = rng.uniform(0.5, 2.0, n)
    matrix = from_scipy_sparse(sp.csr_matrix(A))
    result = matrix.solve_eq_box_ipm(
        c.tolist(), b.tolist(), [0.0] * n, [np.inf] * n, max_iter=50, tol=1e-9, debug=True
    )
    assert result["status"] == "optimal"
    assert "dual_cleanup_rounds" in result


@pytest.mark.parametrize("seed", [2, 17])
def test_ipm_returns_finite_iterates_on_ill_conditioned_lps(seed: int) -> None:
    # late Newton steps can overflow on ill-conditioned instances; the
    # solver must bail to its best iterate and never return NaN/inf
    rng = np.random.default_rng(seed)
    m, n = 30, 60
    A = (
        sp.random(
            m, n, density=0.3, random_state=rng, data_rvs=lambda s: rng.uniform(1e-6, 1e6, s)
        ).tocsr()
        + sp.hstack([sp.identity(m) * 1e-4, sp.csr_matrix((m, n - m))]).tocsr()
    )
    b = A @ rng.uniform(0.5, 1.5, n)
    c = rng.uniform(1e-3, 1e3, n)
    matrix = from_scipy_sparse(sp.csr_matrix(A))
    result = matrix.solve_eq_box_ipm(
        c.tolist(), b.tolist(), [0.0] * n, [np.inf] * n, max_iter=200, tol=1e-9
    )
    assert np.all(np.isfinite(result["x"]))
    assert np.all(np.isfinite(result["y"]))
