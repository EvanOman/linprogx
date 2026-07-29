"""Tests for the DS2 dual-simplex rewrite (gated, additive).

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

DS2 is reached two ways and both are covered here:
  - directly, via CSRMatrix.solve_eq_box_ds2(c, b, lo, hi)
  - through the gate, via LINPROGX_DS2=1 on solve_eq_box_dual_simplex

The point of these tests is the CONTRACT, not the trajectory: every accepted
answer must be optimal for the true problem, oracled against scipy/HiGHS, and
the shipped path must be untouched when the gate is off.  Per AGENTS.md the
oracle imports are unconditional -- a missing SciPy must fail loudly, never
skip.
"""

from __future__ import annotations

import importlib

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from test_dual_simplex import _generate_onesided_lp, _make_csr, _random_feasible_lp

_csparse = importlib.import_module("linprogx._csparse")
CSRMatrix = _csparse.CSRMatrix


def _solve_highs(c, A_dense, b, lo, hi):
    bounds = [
        (
            None if not np.isfinite(lo[j]) else float(lo[j]),
            None if not np.isfinite(hi[j]) else float(hi[j]),
        )
        for j in range(len(c))
    ]
    return scipy.optimize.linprog(c, A_eq=A_dense, b_eq=b, bounds=bounds, method="highs")


# ---------------------------------------------------------------------------
# Random LPs, oracled
# ---------------------------------------------------------------------------


class TestDs2RandomLPs:
    def test_random_batch_30(self) -> None:
        rng = np.random.RandomState(7777)
        for trial in range(30):
            m = rng.randint(4, 41)
            n = m + rng.randint(1, 2 * m + 1)
            c, A, b, lo, hi = _random_feasible_lp(m, n, rng)
            ref = _solve_highs(c, A, b, lo, hi)
            assert ref.status == 0, f"trial {trial}: HiGHS failed"

            res = _make_csr(A).solve_eq_box_ds2(c, b, lo, hi)
            assert res["status"] == "optimal", f"trial {trial}: got {res['status']} (m={m}, n={n})"
            assert abs(res["objective"] - ref.fun) < 1e-6, (
                f"trial {trial}: ours={res['objective']!r} highs={ref.fun!r}"
            )
            assert res["max_primal_residual"] < 1e-8

    def test_random_varied_bounds(self) -> None:
        rng = np.random.RandomState(4242)
        for trial in range(15):
            m = rng.randint(5, 25)
            n = m + rng.randint(2, 3 * m)
            c, A, b, lo, hi = _random_feasible_lp(m, n, rng, lo_min=-3.0, hi_max=5.0)
            ref = _solve_highs(c, A, b, lo, hi)
            assert ref.status == 0
            res = _make_csr(A).solve_eq_box_ds2(c, b, lo, hi)
            assert res["status"] == "optimal", f"trial {trial}: {res['status']}"
            assert abs(res["objective"] - ref.fun) < 1e-6


# ---------------------------------------------------------------------------
# Component-A integration: BFRT changes bounds as well as the entering column,
# so a standalone ratio-test replay is not enough to pin the core seam.
# ---------------------------------------------------------------------------


def test_bfrt_integrates_with_signed_reduced_cost_updates(monkeypatch) -> None:
    monkeypatch.setenv("LINPROGX_DS2_BFRT", "1")
    rng = np.random.RandomState(11037)
    for trial in range(10):
        m = rng.randint(5, 18)
        n = m + rng.randint(3, 2 * m)
        c, A, b, lo, hi = _random_feasible_lp(
            m, n, rng, lo_min=-2.0, hi_max=4.0
        )
        ref = _solve_highs(c, A, b, lo, hi)
        assert ref.status == 0, f"trial {trial}: HiGHS failed"
        res = _make_csr(A).solve_eq_box_ds2(c, b, lo, hi, max_iter=20_000)
        assert res["status"] == "optimal", f"trial {trial}: {res['status']}"
        assert abs(res["objective"] - ref.fun) / (1.0 + abs(ref.fun)) < 1e-6
        assert res["max_primal_residual"] < 1e-7


# ---------------------------------------------------------------------------
# The structures DS2 changes: one-sided and free columns have NO big-M bound,
# so dual feasibility has to come from the phase-1 bound substitution.
# ---------------------------------------------------------------------------


class TestDs2OneSidedAndFree:
    def _run_mode(self, mode: str, seed: int, n_trials: int = 200) -> None:
        rng = np.random.default_rng(seed)
        tested = 0
        fails = []
        for _ in range(n_trials):
            c, A_sp, b, lo, hi = _generate_onesided_lp(mode, rng)
            A_dense = A_sp.toarray()
            ref = _solve_highs(c, A_dense, b, lo, hi)
            if not ref.success:
                continue
            ds = _make_csr(A_dense).solve_eq_box_ds2(
                c.tolist(), b.tolist(), lo.tolist(), hi.tolist()
            )
            if ds["status"] != "optimal":
                continue
            tested += 1
            rel = abs(ds["objective"] - ref.fun) / (1.0 + abs(ref.fun))
            if rel > 1e-6:
                fails.append(rel)
        assert tested >= 5, f"mode={mode}: only {tested} comparable instances"
        assert not fails, f"mode={mode}: {len(fails)}/{tested} objective mismatches: {fails[:3]}"

    def test_boxed(self) -> None:
        self._run_mode("boxed", seed=11)

    def test_upper_only(self) -> None:
        self._run_mode("upper", seed=11)

    def test_free(self) -> None:
        self._run_mode("free", seed=11)


# ---------------------------------------------------------------------------
# Status semantics
# ---------------------------------------------------------------------------


class TestDs2Statuses:
    def test_infeasible_is_certified_not_fudged(self) -> None:
        """x1 + x2 = 10 with both boxed in [0, 1] is primal infeasible.

        DS2 has no artificial bounds, so an empty dual ratio test is a genuine
        infeasibility certificate rather than the shipped path's non-committal
        downgrade.  It must not be reported as optimal.
        """
        A = np.array([[1.0, 1.0]])
        res = _make_csr(A).solve_eq_box_ds2([1.0, 1.0], [10.0], [0.0, 0.0], [1.0, 1.0])
        assert res["status"] in ("infeasible", "iteration_limit"), res["status"]

    def test_all_fixed_variables(self) -> None:
        A = np.array([[1.0, 1.0], [1.0, -1.0]])
        b = np.array([8.0, 2.0])
        res = _make_csr(A).solve_eq_box_ds2([1.0, 1.0], b, [5.0, 3.0], [5.0, 3.0])
        assert res["status"] == "optimal", res["status"]
        assert abs(res["objective"] - 8.0) < 1e-9

    def test_fixed_and_free_mixed(self) -> None:
        rng = np.random.RandomState(31337)
        m, n = 12, 30
        c, A, b, lo, hi = _random_feasible_lp(m, n, rng, lo_min=-2.0, hi_max=4.0)
        lo[0] = hi[0] = 1.0
        lo[1] = -np.inf
        hi[1] = np.inf
        c[1] = 0.0
        b = A @ np.clip(np.zeros(n), lo, np.where(np.isfinite(hi), hi, 1.0))
        ref = _solve_highs(c, A, b, lo, hi)
        if ref.status != 0:
            return
        res = _make_csr(A).solve_eq_box_ds2(c, b, lo, hi)
        assert res["status"] == "optimal", res["status"]
        assert abs(res["objective"] - ref.fun) < 1e-6

    def test_result_dict_keys(self) -> None:
        rng = np.random.RandomState(5)
        c, A, b, lo, hi = _random_feasible_lp(6, 15, rng)
        res = _make_csr(A).solve_eq_box_ds2(c, b, lo, hi)
        for key in (
            "status",
            "objective",
            "max_primal_residual",
            "iterations",
            "x",
            "y",
            "refactorizations",
            "bound_flips",
            "degenerate_pivots",
            "banned_rows",
            "audit_rounds",
            "cost_shifts",
            "phase1_iterations",
            "phase1_dual_objective",
        ):
            assert key in res, f"missing key {key}"
        assert len(res["x"]) == A.shape[1]
        assert len(res["y"]) == A.shape[0]

    def test_deterministic(self) -> None:
        rng = np.random.RandomState(99)
        c, A, b, lo, hi = _random_feasible_lp(15, 40, rng)
        matrix = _make_csr(A)
        first = matrix.solve_eq_box_ds2(c, b, lo, hi)
        for _ in range(3):
            again = matrix.solve_eq_box_ds2(c, b, lo, hi)
            assert again["status"] == first["status"]
            assert again["iterations"] == first["iterations"]
            assert again["objective"] == first["objective"]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestDs2Gate:
    """LINPROGX_DS2 must be default OFF and must route when set."""

    def _problem(self):
        rng = np.random.RandomState(2026)
        c, A, b, lo, hi = _random_feasible_lp(20, 55, rng)
        return _make_csr(A), c, b, lo, hi

    def test_gate_off_by_default_keeps_the_shipped_path(self, monkeypatch) -> None:
        monkeypatch.delenv("LINPROGX_DS2", raising=False)
        matrix, c, b, lo, hi = self._problem()
        res = matrix.solve_eq_box_dual_simplex(c, b, lo, hi)
        # Diagnostics only the shipped implementation produces.
        assert "dual_progress" in res
        assert "phase_us" in res
        assert "phase1_dual_objective" not in res

    def test_gate_on_routes_to_ds2(self, monkeypatch) -> None:
        matrix, c, b, lo, hi = self._problem()
        shipped = matrix.solve_eq_box_dual_simplex(c, b, lo, hi)
        monkeypatch.setenv("LINPROGX_DS2", "1")
        gated = matrix.solve_eq_box_dual_simplex(c, b, lo, hi)
        direct = matrix.solve_eq_box_ds2(c, b, lo, hi)

        assert "phase1_dual_objective" in gated
        assert "dual_progress" not in gated
        assert gated["status"] == direct["status"] == "optimal"
        assert gated["iterations"] == direct["iterations"]
        assert gated["objective"] == direct["objective"]
        # Same LP, same optimum, whichever implementation answers.
        assert shipped["status"] == "optimal"
        assert abs(shipped["objective"] - gated["objective"]) < 1e-6

    def test_starting_basis_switch_agrees_on_the_optimum(self, monkeypatch) -> None:
        """The logical-basis start (B = I) is a different trajectory to the
        triangular crash, and must reach the same optimal value."""
        matrix, c, b, lo, hi = self._problem()
        crash = matrix.solve_eq_box_ds2(c, b, lo, hi)
        monkeypatch.setenv("LINPROGX_DS2_LOGICAL_BASIS", "1")
        logical = matrix.solve_eq_box_ds2(c, b, lo, hi)
        assert crash["status"] == logical["status"] == "optimal"
        assert abs(crash["objective"] - logical["objective"]) < 1e-6

    def test_phase1_can_be_disabled(self, monkeypatch) -> None:
        """With phase 1 off DS2 has no way to establish dual feasibility on a
        one-sided model, so it must not claim optimality it cannot certify."""
        rng = np.random.default_rng(11)
        c, A_sp, b, lo, hi = _generate_onesided_lp("upper", rng)
        matrix = _make_csr(A_sp.toarray())
        monkeypatch.setenv("LINPROGX_DS2_PHASE1", "0")
        res = matrix.solve_eq_box_ds2(
            c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), max_iter=5000
        )
        assert res["status"] in (
            "optimal",
            "dual_infeasible",
            "infeasible",
            "iteration_limit",
            "numerical_error",
        )
        if res["status"] == "optimal":
            ref = _solve_highs(c, A_sp.toarray(), b, lo, hi)
            if ref.success:
                assert abs(res["objective"] - ref.fun) / (1.0 + abs(ref.fun)) < 1e-6


# ---------------------------------------------------------------------------
# A medium instance, to exercise refactorization cadence and the phase boundary
# ---------------------------------------------------------------------------


class TestDs2Medium:
    def test_m200_n600_matches_highs(self) -> None:
        rng = np.random.RandomState(20260726)
        m, n = 200, 600
        A = sp.random(m, n, density=0.05, random_state=rng, format="csr")
        A = (A + sp.hstack([sp.identity(m), sp.csr_matrix((m, n - m))])).tocsr()
        A_dense = A.toarray()
        lo = np.zeros(n)
        hi = np.full(n, np.inf)
        hi[: n // 3] = 2.0
        x0 = rng.rand(n) * 0.5
        b = A_dense @ x0
        c = rng.randn(n)

        ref = _solve_highs(c, A_dense, b, lo, hi)
        assert ref.status == 0

        res = _make_csr(A_dense).solve_eq_box_ds2(
            c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), max_iter=50000
        )
        assert res["status"] == "optimal", res["status"]
        assert abs(res["objective"] - ref.fun) / (1.0 + abs(ref.fun)) < 1e-6
        assert res["max_primal_residual"] < 1e-6
