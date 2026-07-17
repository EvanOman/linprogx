"""Characterization tests for the second-fixpoint presolve re-stage.

The V2 opportunity gate in :func:`presolve_matrix` is scored on the raw problem,
before the classic singleton/doubleton cascade creates new fixed/forcing/empty/
column-singleton opportunities. ``LINPROGX_PRESOLVE_FIXPOINT`` (default on)
re-evaluates the gate on the classic-reduced problem and, when classic made
meaningful progress, iterates the combined reduction to its fixpoint by composing
a second native V2 reduction onto the classic one.

These tests pin:
  * the OFF path is byte-identical to the pre-change reduction (golden shapes),
  * the ON path reaches the measured second-fixpoint shapes,
  * reconstruction (postsolve) of the composed reduction is feasible and matches
    the external HiGHS oracle objective,
  * the gate stays CLOSED on the OSA negative controls (classic reduces nothing),
  * the composition/remap algebra is correct in isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.sparse import csr_matrix as scipy_csr

import linprogx.presolve as P
from linprogx.presolve import (
    PresolveResult,
    _ColumnSingleton,
    _compose_reductions,
    _Doubleton,
    _DuplicateColumn,
    _empty_reduction_counts,
    _FixedVar,
    _fixpoint_worth_restage,
    _remap_record,
    postsolve_x,
    presolve_matrix,
)
from linprogx.sparse import csr_matrix, from_scipy_sparse

INF = float("inf")
SUITE = Path("/tmp/lpsuite")
REPO_CRE_A = Path(__file__).parent / "data" / "lp_cre_a.mat"

# Pre-change reduced shapes (rows, cols, nnz). OFF path must reproduce these.
GOLDEN_OFF = {
    "80bau3b": (2079, 11878, 22923),
    "cre_a": (3041, 6861, 17274),
    "cre_b": (5277, 36323, 111954),
    "cre_d": (4068, 28567, 86566),
    "d2q06c": (1996, 5656, 32714),
    "degen3": (1470, 2571, 25366),
    "fit2p": None,
    "greenbea": (1525, 3868, 23274),
    "ken_07": (947, 2123, 5014),
    "ken_11": (5729, 12384, 29808),
    "ken_13": (11042, 25069, 59376),
    "osa_14": None,
    "pds_10": (14438, 47812, 103230),
    "pilot87": (2016, 6666, 74917),
    "qap12": None,
    "stocfor3": (14633, 21499, 68419),
    "truss": None,
    "woodw": (707, 5363, 19807),
}
# ON path second-fixpoint shapes; unchanged instances stay at OFF golden. Only
# instances whose second (V2) reduction removes >= 2% of the reduced shape keep
# it; d2q06c/ken_*/pds_10 have a tiny second reduction (< ~1.2%) that the
# acceptance gate discards because keeping it regresses the solve. pilot87
# (< 2% classic reduction) and degen3 (no V2 candidates) also stay at OFF.
EXPECT_ON = dict(GOLDEN_OFF)
EXPECT_ON.update(
    {
        "80bau3b": (1992, 11155, 21798),
        "cre_a": (2951, 6649, 16734),
        "stocfor3": (13864, 20730, 59964),
    }
)
# Instances that trigger the second pass (classic >= 2%) but whose second
# reduction is discarded by the acceptance gate: ON shape must equal OFF.
DISCARDED_RESTAGE = ["d2q06c", "ken_07", "ken_11", "ken_13", "pds_10"]


def _load(path: Path):
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsr().astype(float)
    return (
        A,
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def _shape(path: Path):
    A, b, c, lo, hi = _load(path)
    r = presolve_matrix(from_scipy_sparse(A), b.tolist(), c.tolist(), lo.tolist(), hi.tolist())
    return None if r is None else (r._matrix.shape[0], r._matrix.shape[1], r._matrix.nnz)


def _bounds(lo, hi):
    return [
        (None if lo_v == -INF else lo_v, None if hi_v == INF else hi_v)
        for lo_v, hi_v in zip(lo, hi, strict=True)
    ]


def _reconstruct_and_check(path: Path, tol: float = 2e-5) -> None:
    A, b, c, lo, hi = _load(path)
    oracle = linprog(c, A_eq=A, b_eq=b, bounds=_bounds(lo, hi), method="highs")
    assert oracle.success
    r = presolve_matrix(from_scipy_sparse(A), b.tolist(), c.tolist(), lo.tolist(), hi.tolist())
    assert r is not None
    indptr, indices, data = r._matrix.to_components()
    red_A = scipy_csr((data, indices, indptr), shape=(r._matrix.shape[0], r._matrix.shape[1]))
    red = linprog(r.c, A_eq=red_A, b_eq=r.b, bounds=_bounds(r.lo, r.hi), method="highs")
    assert red.success
    x_full = np.array(postsolve_x(list(red.x), r), dtype=float)
    # original-space feasibility
    assert float(np.max(np.abs(A @ x_full - b))) < 1e-6
    assert float(np.max(np.maximum(lo - x_full, 0.0))) < 1e-7
    assert float(np.max(np.maximum(x_full - hi, 0.0))) < 1e-7
    # objective agreement with the external oracle
    obj_full = float(c @ x_full)
    assert abs(obj_full - float(oracle.fun)) / max(1.0, abs(float(oracle.fun))) < tol
    # reduced objective + offset reconstructs the original objective
    obj_red = float(red.fun) + r.objective_offset
    assert abs(obj_red - obj_full) / max(1.0, abs(obj_full)) < 1e-9


# --------------------------------------------------------------------------- #
# Always-run: composition / remap algebra (no fixtures needed)
# --------------------------------------------------------------------------- #


def test_fixpoint_worth_restage_threshold() -> None:
    # 2% of rows or cols removed by classic opens the re-stage cost guard.
    assert _fixpoint_worth_restage(2, 0, 100, 100) is True  # exactly 2% rows
    assert _fixpoint_worth_restage(0, 2, 100, 100) is True  # exactly 2% cols
    assert _fixpoint_worth_restage(1, 1, 100, 100) is False  # 1% each, below guard
    assert _fixpoint_worth_restage(0, 0, 100, 100) is False  # OSA: classic no-op


def test_fixpoint_reduction_is_substantial_threshold() -> None:
    # The composed second reduction is kept only when it removes >= 2% of the
    # reduced problem's rows or cols; smaller reductions are discarded.
    assert P._fixpoint_reduction_is_substantial(2, 0, 100, 100) is True
    assert P._fixpoint_reduction_is_substantial(0, 2, 100, 100) is True
    assert P._fixpoint_reduction_is_substantial(1, 1, 100, 100) is False


def test_fixpoint_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINPROGX_PRESOLVE_FIXPOINT", raising=False)
    assert P._fixpoint_enabled() is True  # default on
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "0")
    assert P._fixpoint_enabled() is False
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    assert P._fixpoint_enabled() is True


def test_remap_record_all_types() -> None:
    a1 = [10, 20, 30]  # intermediate column -> original column
    assert _remap_record(_FixedVar(1, 4.0), a1) == _FixedVar(20, 4.0)
    assert _remap_record(_Doubleton(0, 2, 1.5, -2.0, 7.0), a1) == _Doubleton(10, 30, 1.5, -2.0, 7.0)
    cs = _remap_record(_ColumnSingleton(2, 3.0, 9.0, ((0, 1.0), (1, -1.0))), a1)
    assert cs == _ColumnSingleton(30, 3.0, 9.0, ((10, 1.0), (20, -1.0)))
    assert _remap_record(_DuplicateColumn(1, 0, 0.0, 5.0, 0.0, 8.0), a1) == _DuplicateColumn(
        20, 10, 0.0, 5.0, 0.0, 8.0
    )


def _mk_result(active_cols, records, orig_cols, cols):
    # Minimal PresolveResult; only the fields compose/postsolve touch matter.
    m = csr_matrix(1, cols, [0, 0], [], [])
    return PresolveResult(
        rows=1,
        cols=cols,
        indptr=[],
        indices=[],
        data=[],
        b=[0.0],
        c=[0.0] * cols,
        lo=[0.0] * cols,
        hi=[INF] * cols,
        objective_offset=0.0,
        removed_rows=0,
        removed_cols=orig_cols - cols,
        _records=records,
        _active_cols=active_cols,
        _original_cols=orig_cols,
        _reduction_counts=_empty_reduction_counts(),
        _matrix=m,
    )


def test_compose_reductions_postsolve() -> None:
    # original cols x0,x1,x2. first fixes x1=5, survivors [x0,x2] (M1 cols 0,1).
    first = _mk_result([0, 2], [_FixedVar(1, 5.0)], orig_cols=3, cols=2)
    # second (on M1) fixes M1-col1 (=x2) to 7, survivor M1-col0 (=x0).
    second = _mk_result([0], [_FixedVar(1, 7.0)], orig_cols=2, cols=1)
    composed = _compose_reductions(first, second)
    assert composed._active_cols == [0]
    assert composed._original_cols == 3
    # x0 = 9 in the doubly-reduced space -> [9, 5, 7] in the original space.
    assert postsolve_x([9.0], composed) == [9.0, 5.0, 7.0]
    assert composed.removed_cols == 2


def test_compose_offset_and_counts() -> None:
    first = _mk_result([0, 2], [_FixedVar(1, 5.0)], orig_cols=3, cols=2)
    first.objective_offset = 3.0
    first._reduction_counts["fixed_columns"] = 1
    second = _mk_result([0], [_FixedVar(1, 7.0)], orig_cols=2, cols=1)
    second.objective_offset = -2.0
    second._reduction_counts["fixed_columns"] = 1
    composed = _compose_reductions(first, second)
    assert composed.objective_offset == 1.0
    assert composed._reduction_counts["fixed_columns"] == 2


# --------------------------------------------------------------------------- #
# In-repo cre_a fixture: exercises the staged path end-to-end (always available)
# --------------------------------------------------------------------------- #


# These characterize the classic+fixpoint reduction, which composes strictly
# before the (now default-on) aggregation re-stage. Pin the aggregation knob off
# so the goldens isolate the fixpoint shapes; the aggregation port has its own
# bit-equivalence suite in test_presolve_equivalence.py.
@pytest.mark.skipif(not REPO_CRE_A.exists(), reason="in-repo cre_a fixture missing")
def test_cre_a_off_path_is_pre_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_AGG", "0")
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "0")
    assert _shape(REPO_CRE_A) == GOLDEN_OFF["cre_a"]


@pytest.mark.skipif(not REPO_CRE_A.exists(), reason="in-repo cre_a fixture missing")
def test_cre_a_on_path_reaches_second_fixpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_AGG", "0")
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    assert _shape(REPO_CRE_A) == EXPECT_ON["cre_a"]


@pytest.mark.skipif(not REPO_CRE_A.exists(), reason="in-repo cre_a fixture missing")
def test_cre_a_reconstruction_matches_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    _reconstruct_and_check(REPO_CRE_A)


# --------------------------------------------------------------------------- #
# Full local suite (gated on /tmp/lpsuite)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not SUITE.exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize("name", sorted(GOLDEN_OFF))
def test_off_path_byte_identical(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_AGG", "0")
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "0")
    assert _shape(SUITE / f"lp_{name}.mat") == GOLDEN_OFF[name]


@pytest.mark.skipif(not SUITE.exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize("name", sorted(EXPECT_ON))
def test_on_path_second_fixpoint(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_AGG", "0")
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    assert _shape(SUITE / f"lp_{name}.mat") == EXPECT_ON[name]


@pytest.mark.skipif(not SUITE.exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize("name", ["osa_14"])
def test_osa_gate_stays_closed(name: str) -> None:
    # Classic reduces nothing on OSA, so the re-stage never triggers: ON and OFF
    # both return None (raw matrix passed through unchanged).
    path = SUITE / f"lp_{name}.mat"
    A, b, c, lo, hi = _load(path)
    import os

    prev = os.environ.get("LINPROGX_PRESOLVE_FIXPOINT")
    try:
        os.environ["LINPROGX_PRESOLVE_FIXPOINT"] = "1"
        on = presolve_matrix(from_scipy_sparse(A), b.tolist(), c.tolist(), lo.tolist(), hi.tolist())
        os.environ["LINPROGX_PRESOLVE_FIXPOINT"] = "0"
        off = presolve_matrix(
            from_scipy_sparse(A), b.tolist(), c.tolist(), lo.tolist(), hi.tolist()
        )
    finally:
        if prev is None:
            os.environ.pop("LINPROGX_PRESOLVE_FIXPOINT", None)
        else:
            os.environ["LINPROGX_PRESOLVE_FIXPOINT"] = prev
    assert on is None
    assert off is None


@pytest.mark.skipif(not SUITE.exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize("name", ["cre_a", "80bau3b", "stocfor3", "d2q06c", "ken_07", "pds_10"])
def test_second_fixpoint_reconstruction(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    _reconstruct_and_check(SUITE / f"lp_{name}.mat")


@pytest.mark.skipif(not SUITE.exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize("name", DISCARDED_RESTAGE)
def test_tiny_second_reduction_is_discarded(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # These trigger the second pass (classic >= 2%) but the second reduction is
    # too small to keep; the acceptance gate must fall back to the classic shape
    # (keeping it regressed the solve by up to 41% in measurement).
    monkeypatch.setenv("LINPROGX_PRESOLVE_AGG", "0")
    monkeypatch.setenv("LINPROGX_PRESOLVE_FIXPOINT", "1")
    assert _shape(SUITE / f"lp_{name}.mat") == GOLDEN_OFF[name]
