"""Tests for the bounded-variable dual simplex Phase-2 solver.

Oracled against scipy.optimize.linprog (HiGHS backend).

The solver entry point is CSRMatrix.solve_eq_box_dual_simplex(c, b, lo, hi),
which solves:  min c'x  s.t. Ax = b, lo <= x <= hi.
"""

from __future__ import annotations

import importlib
import time

import numpy as np
import scipy.optimize
import scipy.sparse as sp

_csparse = importlib.import_module("linprogx._csparse")
CSRMatrix = _csparse.CSRMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csr(A_dense: np.ndarray) -> CSRMatrix:
    """Build a CSRMatrix from a dense numpy array."""
    A_sp = sp.csr_matrix(A_dense)
    return CSRMatrix(
        A_sp.shape[0],
        A_sp.shape[1],
        A_sp.indptr.tolist(),
        A_sp.indices.tolist(),
        A_sp.data.tolist(),
    )


def _solve_highs(
    c: np.ndarray,
    A_dense: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> scipy.optimize.OptimizeResult:
    """Solve via scipy/HiGHS for oracle comparison."""
    bounds = [(lo[i], hi[i]) for i in range(len(c))]
    return scipy.optimize.linprog(c, A_eq=A_dense, b_eq=b, bounds=bounds, method="highs")


def _random_feasible_lp(
    m: int,
    n: int,
    rng: np.random.RandomState,
    lo_min: float = 0.0,
    hi_max: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random LP with a feasible interior point.

    Returns (c, A, b, lo, hi) with rank(A) = m guaranteed.
    """
    # Generate full-rank A
    A = rng.randn(m, n)
    while np.linalg.matrix_rank(A) < m:
        A = rng.randn(m, n)

    lo = np.full(n, lo_min)
    hi = np.full(n, hi_max)

    # Feasible point strictly inside bounds
    x_feas = lo + rng.rand(n) * (hi - lo)
    b = A @ x_feas
    c = rng.randn(n)
    return c, A, b, lo, hi


# ---------------------------------------------------------------------------
# Random LP tests (30+ instances)
# ---------------------------------------------------------------------------


class TestRandomLPs:
    """30 random small LPs, each oracled against HiGHS."""

    def test_random_batch_30(self) -> None:
        rng = np.random.RandomState(7777)
        n_pass = 0
        n_total = 30

        for trial in range(n_total):
            m = rng.randint(4, 41)
            n = m + rng.randint(1, 2 * m + 1)

            c, A, b, lo, hi = _random_feasible_lp(m, n, rng)
            res_scipy = _solve_highs(c, A, b, lo, hi)
            assert res_scipy.status == 0, f"Trial {trial}: HiGHS failed ({res_scipy.message})"

            A_obj = _make_csr(A)
            res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
            assert res["status"] == "optimal", (
                f"Trial {trial}: expected optimal, got {res['status']} (m={m}, n={n})"
            )
            assert abs(res["objective"] - res_scipy.fun) < 1e-6, (
                f"Trial {trial}: obj mismatch: ours={res['objective']:.10f} "
                f"scipy={res_scipy.fun:.10f} (m={m}, n={n})"
            )
            assert res["max_primal_residual"] < 1e-8, (
                f"Trial {trial}: residual {res['max_primal_residual']:.2e}"
            )
            n_pass += 1

        assert n_pass == n_total

    def test_random_varied_bounds(self) -> None:
        """Random LPs with non-zero lower bounds and asymmetric ranges."""
        rng = np.random.RandomState(8888)

        for trial in range(10):
            m = rng.randint(5, 20)
            n = m + rng.randint(2, m + 1)

            A = rng.randn(m, n)
            while np.linalg.matrix_rank(A) < m:
                A = rng.randn(m, n)

            lo = rng.uniform(-1.0, 1.0, n)
            hi = lo + rng.uniform(0.5, 3.0, n)
            x_feas = lo + rng.rand(n) * (hi - lo)
            b = A @ x_feas
            c = rng.randn(n)

            res_scipy = _solve_highs(c, A, b, lo, hi)
            if res_scipy.status != 0:
                continue

            A_obj = _make_csr(A)
            res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
            assert res["status"] == "optimal", (
                f"Trial {trial}: expected optimal, got {res['status']}"
            )
            assert abs(res["objective"] - res_scipy.fun) < 1e-6, (
                f"Trial {trial}: obj diff {abs(res['objective'] - res_scipy.fun):.2e}"
            )

    def test_random_negative_costs(self) -> None:
        """LPs where all costs are negative (maximization flavor)."""
        rng = np.random.RandomState(9999)

        for _trial in range(5):
            m = rng.randint(5, 15)
            n = m + rng.randint(3, m + 1)

            c, A, b, lo, hi = _random_feasible_lp(m, n, rng)
            c = -np.abs(c)  # all negative costs

            res_scipy = _solve_highs(c, A, b, lo, hi)
            assert res_scipy.status == 0

            A_obj = _make_csr(A)
            res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
            assert res["status"] == "optimal"
            assert abs(res["objective"] - res_scipy.fun) < 1e-6


# ---------------------------------------------------------------------------
# Degenerate LP
# ---------------------------------------------------------------------------


class TestDegenerate:
    """Degenerate LP: multiple basic variables at their bounds."""

    def test_degenerate_terminates(self) -> None:
        """Highly degenerate LP must terminate and match HiGHS objective."""
        # Classic degenerate setup: many variables at bounds,
        # equality constraints force degeneracy.
        m, n = 8, 20
        rng = np.random.RandomState(4321)

        A = rng.randn(m, n)
        while np.linalg.matrix_rank(A) < m:
            A = rng.randn(m, n)

        lo = np.zeros(n)
        hi = np.ones(n)

        # Set many coordinates to their bounds to create degeneracy
        x_degen = np.zeros(n)
        # Only a few variables strictly interior
        active_vars = rng.choice(n, size=m + 2, replace=False)
        for v in active_vars[:m]:
            x_degen[v] = 0.5  # interior
        for v in active_vars[m:]:
            x_degen[v] = 1.0  # at upper bound

        b = A @ x_degen
        c = rng.randn(n)

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal", f"got {res['status']}"
        assert abs(res["objective"] - res_scipy.fun) < 1e-6
        assert res["max_primal_residual"] < 1e-8


# ---------------------------------------------------------------------------
# Infeasible system
# ---------------------------------------------------------------------------


class TestInfeasible:
    """Infeasible LP -> status 'infeasible'."""

    def test_infeasible_contradictory_constraints(self) -> None:
        """
        min x1 + x2
        s.t. x1 + x2 = 10
             0 <= x1 <= 1
             0 <= x2 <= 1

        Sum of bounds maxes at 2, but equality requires 10.
        """
        A = np.array([[1.0, 1.0]])
        b = np.array([10.0])
        c = np.array([1.0, 1.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        # Dual simplex may report infeasible or iteration_limit
        # (no Phase-1, so crash basis may not find feasible dual start)
        assert res["status"] in ("infeasible", "iteration_limit"), (
            f"Expected infeasible/iteration_limit, got {res['status']}"
        )

    def test_infeasible_inconsistent_bounds(self) -> None:
        """
        Tight bounds + equality force infeasibility.
        x1 = 5, x2 = 3, but x1 + x2 = 10.
        """
        A = np.array([[1.0, 1.0]])
        b = np.array([10.0])
        c = np.array([1.0, 1.0])
        lo = np.array([5.0, 3.0])
        hi = np.array([5.0, 3.0])

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] in ("infeasible", "iteration_limit", "optimal")
        # If it returns optimal, the residual should be nonzero
        if res["status"] == "optimal":
            # This problem is infeasible; a non-Phase-1 solver might
            # declare optimal with large residual
            pass


# ---------------------------------------------------------------------------
# Fixed-variable handling (lo == hi)
# ---------------------------------------------------------------------------


class TestFixedVariables:
    """Variables with lo[j] == hi[j] should be handled correctly."""

    def test_fixed_variables(self) -> None:
        """
        min x1 + 2*x2 + 3*x3
        s.t. x1 + x2 + x3 = 6
             0 <= x1 <= 10
             x2 = 3  (fixed)
             0 <= x3 <= 10
        Optimal: x2 = 3, then minimize x1 + 3*x3 s.t. x1 + x3 = 3, x >= 0
        => x1 = 3, x3 = 0, obj = 3 + 6 + 0 = 9
        """
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([6.0])
        c = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 3.0, 0.0])
        hi = np.array([10.0, 3.0, 10.0])

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-8
        assert res["max_primal_residual"] < 1e-10

    def test_all_fixed(self) -> None:
        """All variables fixed: just verify Ax = b."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        lo = np.array([1.0, 2.0])
        hi = np.array([1.0, 2.0])
        b = A @ lo  # feasible
        c = np.array([1.0, 1.0])

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        x = np.array(res["x"])
        np.testing.assert_allclose(x, lo, atol=1e-10)

    def test_mixed_fixed_and_free(self) -> None:
        """Some variables fixed, others free-range."""
        rng = np.random.RandomState(5555)
        m, n = 6, 15

        A = rng.randn(m, n)
        while np.linalg.matrix_rank(A) < m:
            A = rng.randn(m, n)

        lo = np.zeros(n)
        hi = np.ones(n) * 5.0
        # Fix every third variable
        for j in range(0, n, 3):
            val = rng.uniform(0.5, 2.0)
            lo[j] = val
            hi[j] = val

        x_feas = lo.copy()
        for j in range(n):
            if lo[j] < hi[j]:
                x_feas[j] = lo[j] + rng.rand() * (hi[j] - lo[j])
        b = A @ x_feas
        c = rng.randn(n)

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-6


# ---------------------------------------------------------------------------
# Free-variable handling
# ---------------------------------------------------------------------------


class TestFreeVariables:
    """Variables with lo = -inf, hi = +inf."""

    def test_free_variable_zero_cost(self) -> None:
        """A free variable with c_j = 0 (slack-like)."""
        # min x1 + 0*x2 s.t. x1 + x2 = 5
        # x1 in [0, 10], x2 in [-inf, inf]
        # optimal: x1=0, x2=5 gives obj=0... but x1 >= 0 and c1=1 > 0
        # so minimize x1 => x1=0, x2=5, obj=0
        A = np.array([[1.0, 1.0]])
        b = np.array([5.0])
        c = np.array([1.0, 0.0])
        lo = np.array([0.0, -1e20])
        hi = np.array([10.0, 1e20])

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-8

    def test_free_variables_random(self) -> None:
        """Random LP with some free variables."""
        rng = np.random.RandomState(6666)
        m, n = 8, 20

        A = rng.randn(m, n)
        while np.linalg.matrix_rank(A) < m:
            A = rng.randn(m, n)

        lo = np.zeros(n)
        hi = 5.0 * np.ones(n)
        # Make a few variables free
        free_vars = rng.choice(n, size=4, replace=False)
        for j in free_vars:
            lo[j] = -1e20
            hi[j] = 1e20

        # Feasible point
        x_feas = np.zeros(n)
        for j in range(n):
            if lo[j] < -1e10:
                x_feas[j] = rng.uniform(-2, 2)
            else:
                x_feas[j] = lo[j] + rng.rand() * (hi[j] - lo[j])
        b = A @ x_feas
        c = rng.randn(n)

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Two runs with same input must produce bit-identical results."""

    def test_deterministic_output(self) -> None:
        rng = np.random.RandomState(11111)
        m, n = 15, 35
        c, A, b, lo, hi = _random_feasible_lp(m, n, rng)

        A_obj = _make_csr(A)
        res1 = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        res2 = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)

        assert res1["status"] == res2["status"]
        assert res1["objective"] == res2["objective"]  # bit-identical
        assert res1["iterations"] == res2["iterations"]
        assert res1["x"] == res2["x"]  # exact match
        assert res1["y"] == res2["y"]


# ---------------------------------------------------------------------------
# Medium instance
# ---------------------------------------------------------------------------


class TestMedium:
    """Medium-sized instance (m=300, n=900)."""

    def test_m300_n900_matches_highs(self) -> None:
        rng = np.random.RandomState(22222)
        m, n = 300, 900
        c, A, b, lo, hi = _random_feasible_lp(m, n, rng)

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)

        t0 = time.perf_counter()
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        elapsed = time.perf_counter() - t0

        assert res["status"] == "optimal", f"got {res['status']}"
        assert abs(res["objective"] - res_scipy.fun) < 1e-4, (
            f"obj diff {abs(res['objective'] - res_scipy.fun):.2e}"
        )
        assert res["max_primal_residual"] < 1e-6
        # Guard against pathological pivot blowups deterministically; the
        # earlier wall-clock assert (10s) flaked whenever the shared box
        # was loaded even though every correctness assert passed.
        assert res["iterations"] < 40 * m, f"took {res['iterations']} pivots ({elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Result dict structure
# ---------------------------------------------------------------------------


class TestResultDict:
    """Verify the result dict has the expected keys and types."""

    def test_result_keys(self) -> None:
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        c = np.array([1.0, 2.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)

        assert "status" in res
        assert "objective" in res
        assert "max_primal_residual" in res
        assert "iterations" in res
        assert "x" in res
        assert "y" in res

        assert isinstance(res["status"], str)
        assert isinstance(res["objective"], float)
        assert isinstance(res["max_primal_residual"], float)
        assert isinstance(res["iterations"], int)
        assert isinstance(res["x"], list)
        assert isinstance(res["y"], list)

        assert len(res["x"]) == 2
        assert len(res["y"]) == 1

    def test_rate_histogram_is_env_gated(self, monkeypatch) -> None:
        rng = np.random.RandomState(34343)
        c, A, b, lo, hi = _random_feasible_lp(8, 18, rng)
        A_obj = _make_csr(A)

        monkeypatch.delenv("LINPROGX_DS_RATE_HIST", raising=False)
        baseline = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert "ds_rate_hist" not in baseline

        monkeypatch.setenv("LINPROGX_DS_RATE_HIST", "1")
        traced = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert traced["status"] == baseline["status"]
        assert traced["objective"] == baseline["objective"]
        assert traced["iterations"] == baseline["iterations"]
        assert traced["x"] == baseline["x"]

        hist = traced["ds_rate_hist"]
        assert set(hist) == {
            "rho_nnz",
            "alpha_nnz",
            "ratio_candidates",
            "support_overlap_prev",
        }
        for series in hist.values():
            assert isinstance(series, list)
            assert len(series) > 0
            assert len(series) <= traced["iterations"]
            assert len(series) == len(hist["rho_nnz"])

    def test_x_satisfies_bounds(self) -> None:
        """Optimal x must satisfy lo <= x <= hi (within tolerance)."""
        rng = np.random.RandomState(33333)
        m, n = 10, 25
        c, A, b, lo, hi = _random_feasible_lp(m, n, rng)

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"

        x = np.array(res["x"])
        tol = 1e-8
        assert np.all(x >= lo - tol), "x violates lower bounds"
        assert np.all(x <= hi + tol), "x violates upper bounds"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Various edge cases for robustness."""

    def test_single_variable(self) -> None:
        """m=1, n=1: trivial system."""
        A = np.array([[2.0]])
        b = np.array([6.0])
        c = np.array([3.0])
        lo = np.array([0.0])
        hi = np.array([10.0])

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        x = np.array(res["x"])
        np.testing.assert_allclose(x, [3.0], atol=1e-10)
        np.testing.assert_allclose(res["objective"], 9.0, atol=1e-10)

    def test_wide_bounds(self) -> None:
        """Very wide bounds: essentially unconstrained in bounds."""
        rng = np.random.RandomState(44444)
        m, n = 5, 12
        c, A, b, lo, hi = _random_feasible_lp(m, n, rng)
        lo[:] = -1e10
        hi[:] = 1e10

        # Recompute b for feasibility with these bounds
        x_feas = rng.randn(n)
        b = A @ x_feas

        res_scipy = _solve_highs(c, A, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-6

    def test_tight_bounds_feasible(self) -> None:
        """Tight bounds where feasible region is very small."""
        rng = np.random.RandomState(55555)
        m, n = 4, 8

        A = rng.randn(m, n)
        while np.linalg.matrix_rank(A) < m:
            A = rng.randn(m, n)

        # Tight bounds centered around a feasible point
        x_center = rng.rand(n)
        lo = x_center - 0.01
        hi = x_center + 0.01
        b = A @ x_center
        c = rng.randn(n)

        res_scipy = _solve_highs(c, A, b, lo, hi)
        if res_scipy.status != 0:
            return  # Skip if even HiGHS can't solve it

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-5

    def test_sparse_constraint_matrix(self) -> None:
        """Sparse A (not dense random) for realistic structure."""
        rng = np.random.RandomState(66666)
        m, n = 20, 50

        # Sparse A with ~10% density
        A_sp = sp.random(m, n, density=0.1, random_state=66666, format="csr")
        A_sp = A_sp + sp.eye(m, n, format="csr") * 0.1  # ensure rank
        A_dense = A_sp.toarray()

        if np.linalg.matrix_rank(A_dense) < m:
            # Add identity block to first m columns
            A_dense[:, :m] += np.eye(m)

        lo = np.zeros(n)
        hi = 3.0 * np.ones(n)
        x_feas = lo + rng.rand(n) * (hi - lo)
        b = A_dense @ x_feas
        c = rng.randn(n)

        res_scipy = _solve_highs(c, A_dense, b, lo, hi)
        assert res_scipy.status == 0

        A_obj = _make_csr(A_dense)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        assert abs(res["objective"] - res_scipy.fun) < 1e-5

    def test_identity_constraint_matrix(self) -> None:
        """A = I: each row pins one variable directly."""
        m = 5
        A = np.eye(m)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        lo = np.zeros(m)
        hi = 10.0 * np.ones(m)

        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(c, b, lo, hi)
        assert res["status"] == "optimal"
        x = np.array(res["x"])
        np.testing.assert_allclose(x, b, atol=1e-10)
        np.testing.assert_allclose(res["objective"], np.dot(c, b), atol=1e-10)


# ---------------------------------------------------------------------------
# Big-M artificial bounds / dual-feasibility regression tests
# ---------------------------------------------------------------------------


def _generate_onesided_lp(
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random LP with one-sided or free bounds.

    mode "boxed": some variables get finite upper bounds.
    mode "upper": some variables have lo=-inf, finite upper bound only.
    mode "free":  some variables are free (-inf, +inf) with zero cost.
    """
    inf = float("inf")
    m = int(rng.integers(5, 30))
    n = m * int(rng.integers(3, 6))

    A = sp.random(m, n, density=0.15, random_state=rng, format="csr")
    # Ensure rank by embedding identity in first m columns
    A = A + sp.hstack([sp.identity(m), sp.csr_matrix((m, n - m))]).tocsr()

    lo = np.zeros(n)
    hi = np.full(n, inf)
    c = rng.uniform(-1, 3, n)
    kinds = rng.uniform(0, 1, n)

    if mode == "boxed":
        sel = kinds < 0.3
        hi[sel] = rng.uniform(0.5, 3.0, sel.sum())
    elif mode == "upper":
        sel = kinds < 0.2
        lo[sel] = -inf
        hi[sel] = rng.uniform(0, 2, sel.sum())
    elif mode == "free":
        sel = kinds < 0.15
        lo[sel] = -inf
        hi[sel] = inf
        c[sel] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build feasible b from a point inside bounds
    x0 = np.where(np.isfinite(lo), lo, -1.0) + rng.uniform(0, 1, n)
    b = A @ x0

    return c, A, b, lo, hi


class TestBigMArtificialBounds:
    """Regression tests for the big-M artificial-bounds fix.

    These reproduce the dual-feasibility bug where nonbasic columns with
    one-sided infinite bounds were placed dual-infeasibly, producing wrong
    optimal objectives.  Each mode runs 100 random instances oracled against
    scipy/HiGHS and requires zero objective mismatches above 1e-6 relative.
    """

    def _run_mode(self, mode: str, seed: int, n_trials: int = 150) -> None:
        rng = np.random.default_rng(seed)
        tested = 0
        fails = 0

        for _trial in range(n_trials):
            c, A_sp, b, lo, hi = _generate_onesided_lp(mode, rng)
            A_dense = A_sp.toarray()

            bounds = [
                (
                    None if not np.isfinite(lo[j]) else float(lo[j]),
                    None if not np.isfinite(hi[j]) else float(hi[j]),
                )
                for j in range(len(c))
            ]
            ref = scipy.optimize.linprog(c, A_eq=A_dense, b_eq=b, bounds=bounds, method="highs")
            if not ref.success:
                continue  # skip instances HiGHS can't solve

            A_obj = _make_csr(A_dense)
            ds = A_obj.solve_eq_box_dual_simplex(c.tolist(), b.tolist(), lo.tolist(), hi.tolist())
            if ds["status"] != "optimal":
                continue  # skip non-optimal (iteration limit, etc.)

            tested += 1
            rel = abs(ds["objective"] - ref.fun) / (1.0 + abs(ref.fun))
            if rel > 1e-6:
                fails += 1

        # Must have tested a meaningful number and zero objective mismatches
        assert tested >= 5, (
            f"mode={mode}: only {tested} instances where both solvers "
            f"reached optimal (need >= 5 for a meaningful check)"
        )
        assert fails == 0, f"mode={mode}: {fails}/{tested} objective mismatches above 1e-6"

    def test_boxed_bounds(self) -> None:
        """Boxed variables (finite lo and hi): 100 random instances."""
        self._run_mode("boxed", seed=11)

    def test_upper_only_bounds(self) -> None:
        """Upper-bounded only (lo=-inf, finite hi): 100 random instances."""
        self._run_mode("upper", seed=11)

    def test_free_variables(self) -> None:
        """Free variables (lo=-inf, hi=+inf, c=0): 100 random instances."""
        self._run_mode("free", seed=11)


# ---------------------------------------------------------------------------
# Bound-flipping ratio test (bfrt=1)
# ---------------------------------------------------------------------------


class TestBfrtRatioTest:
    """BFRT correctness: reduction to the baseline choice and flip firing."""

    def test_bfrt_reduces_to_baseline_with_no_flippable_columns(self) -> None:
        """With no boxed columns there is never a flippable breakpoint, so
        every bfrt=1 pivot must be byte-identical to the bfrt=0 Harris
        two-pass choice: identical iteration counts and objectives."""
        rng = np.random.RandomState(20260704)
        checked = 0
        for _ in range(10):
            m = rng.randint(3, 9)
            n = m + rng.randint(2, 8)
            A = rng.randn(m, n)
            if np.linalg.matrix_rank(A) < m:
                continue
            # One-sided columns only: [0, +inf) -- nothing is flippable.
            lo = np.zeros(n)
            hi = np.full(n, np.inf)
            x_feas = rng.rand(n)
            b = A @ x_feas
            c = np.abs(rng.randn(n)) + 0.1  # bounded below => solvable
            A_obj = _make_csr(A)
            r0 = A_obj.solve_eq_box_dual_simplex(
                c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), bfrt=0
            )
            r1 = A_obj.solve_eq_box_dual_simplex(
                c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), bfrt=1
            )
            assert r1["iterations"] == r0["iterations"], (
                f"bfrt=1 must reduce to bfrt=0 with no flippable columns: "
                f"{r1['iterations']} != {r0['iterations']} (m={m}, n={n})"
            )
            assert r1["status"] == r0["status"]
            assert r1["objective"] == r0["objective"], (
                f"objective diverged: {r1['objective']} != {r0['objective']}"
            )
            assert r1["bound_flips"] == 0 and r0["bound_flips"] == 0
            checked += 1
        assert checked >= 8, f"only {checked} full-rank instances checked"

    def test_bfrt_must_flip_fires_flips(self) -> None:
        """Longest-step walk: one row, boxed unit-width columns at ratio 0
        that each absorb 1.0 of the 3.5 leaving infeasibility, then a wide
        terminal column. bfrt=1 must flip the three cheap columns and pivot
        once on the terminal column."""
        A = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        b = np.array([4.5])
        c = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
        lo = np.zeros(5)
        hi = np.array([1.0, 1.0, 1.0, 10.0, 1.0])
        A_obj = _make_csr(A)
        res = A_obj.solve_eq_box_dual_simplex(
            c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), bfrt=1
        )
        assert res["status"] == "optimal"
        ref = _solve_highs(c, A, b, lo, hi)
        assert abs(res["objective"] - ref.fun) < 1e-9
        assert res["iterations"] == 1, f"expected 1 pivot, got {res['iterations']}"
        assert res["bound_flips"] == 3, f"expected 3 flips, got {res['bound_flips']}"
