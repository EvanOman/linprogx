"""Validate and cost the DS2 CHUZR component on real LP trajectories.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

What this does
--------------
The shipped dual simplex is not modified and not linked. Instead this module
runs a self-contained bounded-variable dual simplex (scipy ``splu`` per pivot,
HiGHS-style logical start plus the Phase-1 bound swap) over the presolved
LPnetlib instances, and at every pivot hands the *real* state -- basis, x_B,
lo_ext, hi_ext, edge weights -- to ``experiments/ds2_chuzr_bench.c``, which:

  * runs the DS2 component,
  * runs the shipped dense scan (scalar and AVX2 transcriptions),
  * brackets all three with rdtsc, rotating which one runs first,
  * records whether they selected the same row.

The DS2 component's answer is what drives the trajectory, so the iteration
count reported here is the iteration count *with the component substituted in*.

Cycles, not wall time: this box drifts 4-19% on cross-process wall
(docs/DS2-REWRITE.md, "Measurement discipline").

Usage
-----
    UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python \
        experiments/ds2_chuzr_validate.py --instances degen2 25fv47 greenbea
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from ctypes import POINTER, c_double, c_int, c_int32, c_int64, c_uint64, c_void_p
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "build" / "ds2" / "libds2chuzr.so"
SOURCES = [
    REPO / "src" / "linprogx" / "_ds2_chuzr.c",
    REPO / "experiments" / "ds2_chuzr_bench.c",
]

RULE_DANTZIG = 0
RULE_DSE = 1
LIST_OFF, LIST_ON, LIST_ALWAYS = 0, 1, 2

FEAS_TOL = 1e-7
DUAL_TOL = 1e-9
PIVOT_TOL = 1e-9
HARRIS_BAND = 1e-7
WEIGHT_FLOOR = 1e-4
REFRESH_EVERY = 100  # full x_B recompute + CHUZR invalidate cadence
REPORT_LEN = 23


# --------------------------------------------------------------------------
# C library
# --------------------------------------------------------------------------
def build_library() -> None:
    LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cc",
        "-O2",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-fPIC",
        "-shared",
        "-o",
        str(LIB),
        *[str(s) for s in SOURCES],
        "-lm",
    ]
    subprocess.run(cmd, check=True)


class Bench:
    """ctypes wrapper over experiments/ds2_chuzr_bench.c."""

    def __init__(self, m: int, rule: int, seed: int = 12345, random_start: bool = False) -> None:
        self.lib = ctypes.CDLL(str(LIB))
        f = self.lib
        f.ds2_bench_new.restype = c_void_p
        f.ds2_bench_new.argtypes = [c_int32, c_int, c_double, c_uint64, c_int]
        f.ds2_bench_free.argtypes = [c_void_p]
        f.ds2_bench_set_paranoid.argtypes = [c_void_p, c_int]
        f.ds2_bench_set_list_mode.argtypes = [c_void_p, c_int]
        f.ds2_bench_set_cutoff_enabled.argtypes = [c_void_p, c_int]
        f.ds2_bench_set_column_density.argtypes = [c_void_p, c_double]
        f.ds2_bench_invalidate.argtypes = [c_void_p]
        f.ds2_bench_rows_changed.argtypes = [
            c_void_p,
            POINTER(c_int32),
            c_int32,
            POINTER(c_int32),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            c_double,
        ]
        f.ds2_bench_call.argtypes = [
            c_void_p,
            c_int64,
            POINTER(c_int32),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            c_double,
            POINTER(c_double),
        ]
        f.ds2_bench_report.argtypes = [c_void_p, POINTER(c_double)]
        f.ds2_bench_samples.argtypes = [c_void_p, c_int, POINTER(c_double), c_int32]
        f.ds2_bench_samples.restype = c_int32
        f.ds2_bench_audit.argtypes = [
            c_void_p,
            POINTER(c_int32),
            POINTER(c_double),
            POINTER(c_double),
            POINTER(c_double),
            c_double,
            POINTER(c_int32),
        ]
        f.ds2_bench_audit.restype = c_int32
        f.ds2_bench_samples.argtypes = [c_void_p, c_int, POINTER(c_double), c_int32]
        f.ds2_bench_samples.restype = c_int32
        f.ds2_bench_has_avx2.restype = c_int
        f.ds2_bench_timer_overhead.restype = c_double
        f.ds2_bench_timer_overhead.argtypes = [c_int64]
        f.ds2_bench_tsc_now.restype = c_double

        self.handle = f.ds2_bench_new(m, rule, WEIGHT_FLOOR, seed, 1 if random_start else 0)
        if not self.handle:
            raise MemoryError("ds2_bench_new failed")
        self.m = m
        self.out = np.zeros(9, dtype=np.float64)
        self._out_p = self.out.ctypes.data_as(POINTER(c_double))

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.lib.ds2_bench_free(self.handle)
            self.handle = None

    def set_list_mode(self, mode: int) -> None:
        self.lib.ds2_bench_set_list_mode(self.handle, mode)

    def set_cutoff(self, on: bool) -> None:
        self.lib.ds2_bench_set_cutoff_enabled(self.handle, 1 if on else 0)

    def set_paranoid(self, on: bool) -> None:
        self.lib.ds2_bench_set_paranoid(self.handle, 1 if on else 0)

    def set_column_density(self, d: float) -> None:
        self.lib.ds2_bench_set_column_density(self.handle, c_double(d))

    def invalidate(self) -> None:
        self.lib.ds2_bench_invalidate(self.handle)

    def rows_changed(self, rows, basis, x_B, lo, hi) -> None:
        self.lib.ds2_bench_rows_changed(
            self.handle,
            rows.ctypes.data_as(POINTER(c_int32)),
            c_int32(rows.size),
            basis.ctypes.data_as(POINTER(c_int32)),
            x_B.ctypes.data_as(POINTER(c_double)),
            lo.ctypes.data_as(POINTER(c_double)),
            hi.ctypes.data_as(POINTER(c_double)),
            c_double(FEAS_TOL),
        )

    def call(self, pivot, basis, x_B, lo, hi, weights):
        self.lib.ds2_bench_call(
            self.handle,
            c_int64(pivot),
            basis.ctypes.data_as(POINTER(c_int32)),
            x_B.ctypes.data_as(POINTER(c_double)),
            lo.ctypes.data_as(POINTER(c_double)),
            hi.ctypes.data_as(POINTER(c_double)),
            weights.ctypes.data_as(POINTER(c_double)),
            c_double(FEAS_TOL),
            self._out_p,
        )
        return self.out

    def audit(self, basis, x_B, lo, hi) -> tuple[int, int]:
        first = c_int32(-1)
        bad = self.lib.ds2_bench_audit(
            self.handle,
            basis.ctypes.data_as(POINTER(c_int32)),
            x_B.ctypes.data_as(POINTER(c_double)),
            lo.ctypes.data_as(POINTER(c_double)),
            hi.ctypes.data_as(POINTER(c_double)),
            c_double(FEAS_TOL),
            ctypes.byref(first),
        )
        return int(bad), int(first.value)

    def report(self) -> dict[str, float]:
        buf = np.zeros(REPORT_LEN, dtype=np.float64)
        self.lib.ds2_bench_report(self.handle, buf.ctypes.data_as(POINTER(c_double)))
        keys = [
            "cyc_ds2",
            "cyc_scalar",
            "cyc_avx2",
            "calls",
            "agree_pos",
            "agree_merit",
            "disagree_merit",
            "both_none",
            "one_none",
            "rebuilds",
            "scanned",
            "dense_scanned",
            "dense_calls",
            "cutoff_misses",
            "list_len_sum",
            "paranoid_mismatch",
            "cyc_update",
            "n_updates",
            "changed_rows",
            "m",
            "recomputes",
            "cutoff_installed",
            "infeas_sum",
        ]
        rep = dict(zip(keys, (float(v) for v in buf), strict=True))
        # Medians, not the sums above: on a box at load 45 any single rdtsc
        # bracket that spans a context switch adds tens of thousands of ticks
        # to a call that costs hundreds, so a mean measures the machine's
        # load and not the code. Quantiles are what survive.
        for which, tag in ((0, "ds2"), (1, "update"), (2, "scalar"), (3, "avx2"), (4, "noop")):
            s = self.samples(which)
            rep[f"med_{tag}"] = float(np.median(s)) if s.size else 0.0
            rep[f"p10_{tag}"] = float(np.percentile(s, 10)) if s.size else 0.0
            rep[f"p90_{tag}"] = float(np.percentile(s, 90)) if s.size else 0.0
            rep[f"n_samples_{tag}"] = int(s.size)
        # PAIRED ratios. The p10-p90 spread below is an order of magnitude for
        # a deterministic scan: that spread is the machine, not the code. Both
        # halves of a pair are measured microseconds apart on the same state,
        # so a per-call RATIO cancels the common-mode noise, and the median of
        # those ratios is the estimator to trust. The raw quantiles stay in the
        # record so the noise remains visible rather than hidden.
        s_ds2, s_upd = self.samples(0), self.samples(1)
        s_sca, s_avx = self.samples(2), self.samples(3)
        n = min(len(s_ds2), len(s_upd), len(s_sca), len(s_avx))
        if n:
            tot = s_ds2[:n] + s_upd[:n]
            ok = (s_sca[:n] > 0) & (s_avx[:n] > 0) & (tot > 0)
            if ok.any():
                rep["ratio_scalar_over_ds2"] = float(np.median(s_sca[:n][ok] / tot[ok]))
                rep["ratio_avx2_over_ds2"] = float(np.median(s_avx[:n][ok] / tot[ok]))
                rep["ratio_avx2_over_ds2_scan"] = float(np.median(s_avx[:n][ok] / s_ds2[:n][ok]))
                rep["n_paired"] = int(ok.sum())
        return rep

    def samples(self, which: int, cap: int = 200_000) -> np.ndarray:
        buf = np.zeros(cap, dtype=np.float64)
        n = self.lib.ds2_bench_samples(
            self.handle, c_int(which), buf.ctypes.data_as(POINTER(c_double)), c_int32(cap)
        )
        return buf[:n]


def calibrate_tsc(lib: ctypes.CDLL, seconds: float = 0.2) -> float:
    """TSC ticks per second, so cycle counts can be quoted in microseconds."""
    t0 = lib.ds2_bench_tsc_now()
    w0 = time.perf_counter()
    while time.perf_counter() - w0 < seconds:
        pass
    t1 = lib.ds2_bench_tsc_now()
    w1 = time.perf_counter()
    return (t1 - t0) / (w1 - w0)


# --------------------------------------------------------------------------
# instance preparation
# --------------------------------------------------------------------------
def load_reduced(name: str, lpsuite: Path) -> dict[str, Any]:
    """Load an LPnetlib instance and apply linprogx's own presolve, so the
    trajectory is over the same reduced problem the shipped solver sees."""
    from scipy.io import loadmat

    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import from_scipy_sparse

    raw = loadmat(lpsuite / f"lp_{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsc().astype(np.float64)
    b = raw["b"].ravel().astype(np.float64)
    c = aux["c"].ravel().astype(np.float64)
    lo = aux["lo"].ravel().astype(np.float64)
    hi = aux["hi"].ravel().astype(np.float64)

    matrix = from_scipy_sparse(sp.csr_matrix(A))
    red = presolve_matrix(matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist())
    if red is None:
        return {"A": sp.csc_matrix(A), "b": b, "c": c, "lo": lo, "hi": hi}
    # The reduction keeps its matrix either as Python CSR lists or as a C
    # CSRMatrix, depending on which presolve path ran.
    if red.indptr:
        indptr, indices, values = red.indptr, red.indices, red.data
    else:
        indptr, indices, values = red._matrix.to_components()
    A_red = sp.csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(red.rows, red.cols),
    ).tocsc()
    return {
        "A": A_red,
        "b": np.asarray(red.b, dtype=np.float64),
        "c": np.asarray(red.c, dtype=np.float64),
        "lo": np.asarray(red.lo, dtype=np.float64),
        "hi": np.asarray(red.hi, dtype=np.float64),
    }


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------
AT_LO, AT_HI, BASIC = 0, 1, 4


def phase1_bounds(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HiGHS's dual Phase-1 bound map (HEkk.cpp:2653-2695), reimplemented.
    free -> [-1000, 1000]; upper-only -> [-1, 0]; lower-only -> [0, 1];
    boxed or fixed -> [0, 0]. Same basis, same factorisation, same weights."""
    lo_fin = np.isfinite(lo)
    hi_fin = np.isfinite(hi)
    p_lo = np.zeros_like(lo)
    p_hi = np.zeros_like(hi)
    free = ~lo_fin & ~hi_fin
    upper_only = ~lo_fin & hi_fin
    lower_only = lo_fin & ~hi_fin
    p_lo[free], p_hi[free] = -1000.0, 1000.0
    p_lo[upper_only], p_hi[upper_only] = -1.0, 0.0
    p_lo[lower_only], p_hi[lower_only] = 0.0, 1.0
    return p_lo, p_hi


def run_instance(
    name: str,
    data: dict[str, Any],
    rule: int,
    max_pivots: int,
    *,
    list_mode: int = LIST_ON,
    cutoff: bool = True,
    random_start: bool = False,
    paranoid_stride: int = 0,
    audit_stride: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    A = data["A"].tocsc()
    b = data["b"].astype(np.float64)
    c = data["c"].astype(np.float64)
    lo = data["lo"].astype(np.float64)
    hi = data["hi"].astype(np.float64)
    m, n = A.shape
    n_total = n + m

    # Logical form: A x - s = 0 with the row variables s fixed at b. Moving b
    # out of the RHS and into the bounds is what makes the Phase-1 bound swap
    # legal: with the RHS homogeneous, z = 0 satisfies the Phase-1 subproblem,
    # so Phase 1 is always feasible. (This is HiGHS's internal form,
    # HEkk.cpp:2653-2695; linprogx has the same device behind
    # LINPROGX_DS_LOGICAL_FORM, _csparse.c:10274.) Starting with every s basic
    # gives B = -I, a logical basis, so the exact DSE weights are 1 for free
    # (HEkkDual.cpp:148-155).
    A_ext = sp.hstack([A, -sp.identity(m, format="csc")], format="csc").tocsc()
    b_solve = np.zeros(m, dtype=np.float64)
    c_ext = np.concatenate([c, np.zeros(m)])
    lo_true = np.concatenate([lo, b])
    hi_true = np.concatenate([hi, b])

    p_lo, p_hi = phase1_bounds(lo_true, hi_true)
    lo_ext = p_lo.copy()
    hi_ext = p_hi.copy()

    basis = np.arange(n, n_total, dtype=np.int32)
    basis_pos = np.full(n_total, -1, dtype=np.int64)
    basis_pos[basis] = np.arange(m)

    status = np.full(n_total, AT_LO, dtype=np.int8)
    status[basis] = BASIC
    weights = np.ones(m, dtype=np.float64)

    bench = Bench(m, rule, random_start=random_start)
    bench.set_list_mode(list_mode)
    bench.set_cutoff(cutoff)

    phase = 1
    x_B = np.zeros(m, dtype=np.float64)
    x_N = np.zeros(n_total, dtype=np.float64)
    need_recompute = True

    def place_nonbasics(r: np.ndarray) -> int:
        """Put every nonbasic on the bound its reduced cost wants. Returns the
        number with no valid placement (dual infeasible on that column)."""
        bad = 0
        nb = basis_pos < 0
        want_lo = nb & (r >= 0.0)
        want_hi = nb & (r < 0.0)
        ok_lo = want_lo & np.isfinite(lo_ext)
        ok_hi = want_hi & np.isfinite(hi_ext)
        status[ok_lo] = AT_LO
        x_N[ok_lo] = lo_ext[ok_lo]
        status[ok_hi] = AT_HI
        x_N[ok_hi] = hi_ext[ok_hi]
        # fall back to whichever side is finite
        rest = nb & ~ok_lo & ~ok_hi
        if np.any(rest):
            idx = np.nonzero(rest)[0]
            for j in idx:
                if np.isfinite(lo_ext[j]):
                    status[j], x_N[j] = AT_LO, lo_ext[j]
                elif np.isfinite(hi_ext[j]):
                    status[j], x_N[j] = AT_HI, hi_ext[j]
                else:
                    status[j], x_N[j] = AT_LO, 0.0
                bad += 1
        x_N[basis] = 0.0
        return bad

    pivots = 0
    drift: list[tuple[int, float]] = []
    phase1_pivots = 0
    degenerate_streak = 0
    stall = 0
    t_start = time.perf_counter()
    outcome = "max_pivots"
    n_free_phase2 = 0

    # initial duals: c_B = 0 for the logical basis, so y = 0 and r = c_ext
    r = c_ext.copy()
    place_nonbasics(r)

    lu = None
    while pivots < max_pivots:
        Bmat = A_ext[:, basis].tocsc()
        try:
            lu = spla.splu(Bmat, permc_spec="COLAMD", diag_pivot_thresh=0.1)
        except RuntimeError:
            outcome = "singular_basis"
            break

        y = lu.solve(c_ext[basis], trans="T")
        r = c_ext - A_ext.T @ y
        r[basis] = 0.0

        if need_recompute:
            rhs = b_solve - A_ext @ x_N
            x_B = lu.solve(rhs)
            bench.invalidate()
            need_recompute = False

        if paranoid_stride:
            bench.set_paranoid(pivots % paranoid_stride == 0)

        res = bench.call(pivots, basis, x_B, lo_ext, hi_ext, weights)
        # Audit AFTER the call: an invalidate leaves the maintained array
        # deliberately stale until the next ds2_chuzr rebuilds it.
        if audit_stride and pivots % audit_stride == 0 and pivots > 0:
            bad, first = bench.audit(basis, x_B, lo_ext, hi_ext)
            if bad:
                raise AssertionError(
                    f"{name}: maintained violations disagree at pivot "
                    f"{pivots}: {bad} rows, first {first}"
                )
        leaving = int(res[0])
        sigma = int(res[1])

        if leaving < 0:
            if phase == 1:
                phase1_pivots = pivots
                lo_ext = lo_true.copy()
                hi_ext = hi_true.copy()
                n_free_phase2 = int(place_nonbasics(r))
                phase = 2
                need_recompute = True
                bench.invalidate()
                if verbose:
                    print(
                        f"    phase 1 done at pivot {pivots} "
                        f"(no-placement columns: {n_free_phase2})"
                    )
                continue
            outcome = "optimal"
            break

        # rho = B^-T e_leaving ; the pivot row over all columns
        e = np.zeros(m)
        e[leaving] = 1.0
        rho = lu.solve(e, trans="T")
        alpha_row = A_ext.T @ rho

        nb = basis_pos < 0
        if sigma > 0:  # x_B[leaving] must increase
            cand = nb & (
                ((status == AT_LO) & (alpha_row < -PIVOT_TOL))
                | ((status == AT_HI) & (alpha_row > PIVOT_TOL))
            )
        else:  # x_B[leaving] must decrease
            cand = nb & (
                ((status == AT_LO) & (alpha_row > PIVOT_TOL))
                | ((status == AT_HI) & (alpha_row < -PIVOT_TOL))
            )
        cols = np.nonzero(cand)[0]
        if cols.size == 0:
            outcome = f"dual_unbounded_phase{phase}"
            break

        ratios = np.abs(r[cols]) / np.abs(alpha_row[cols])
        if degenerate_streak > 200:
            entering = int(cols[0])  # Bland
        else:
            theta_min = float(ratios.min())
            band = cols[ratios <= theta_min + HARRIS_BAND]
            entering = int(band[np.argmax(np.abs(alpha_row[band]))])

        pivot_val = float(alpha_row[entering])
        if abs(pivot_val) < PIVOT_TOL:
            stall += 1
            if stall > 50:
                outcome = "tiny_pivot_stall"
                break
            need_recompute = True
            pivots += 1
            continue

        if abs(float(r[entering])) <= 1e-12:
            degenerate_streak += 1
        else:
            degenerate_streak = 0

        u = lu.solve(A_ext[:, entering].toarray().ravel())
        leaving_var = int(basis[leaving])
        target = lo_ext[leaving_var] if sigma > 0 else hi_ext[leaving_var]
        t_prim = (x_B[leaving] - target) / pivot_val

        # exact Forrest-Goldfarb DSE update (same recurrence as the shipped
        # solver, _csparse.c:15660-15700); gamma_r is anchored to ||rho||^2,
        # which is exact and free because rho is already in hand
        if rule == RULE_DSE:
            tau = lu.solve(rho)
            gamma_r = float(rho @ rho)
            # PRICING DECAY PROBE. The brief's prime suspect is that pricing
            # quality decays as the trajectory lengthens. For the selected row
            # the exact weight is ||rho||^2 and rho is already in hand, so the
            # ratio stored/exact is free to record every pivot. A recurrence
            # that is drifting shows up here as the ratio walking away from 1.
            if gamma_r > 0.0:
                drift.append((pivots, float(weights[leaving] / gamma_r)))
            if gamma_r < WEIGHT_FLOOR:
                gamma_r = WEIGHT_FLOOR
            inv = 1.0 / pivot_val
            ratio = u * inv
            new_w = weights - 2.0 * ratio * tau + ratio * ratio * gamma_r
            np.maximum(new_w, WEIGHT_FLOOR, out=new_w)
            weights[:] = new_w
            weights[leaving] = max(gamma_r * inv * inv, WEIGHT_FLOOR)

        # Primal update: only rows in the FTRAN pattern move. Drop numerical
        # fill below the shipped hyper-sparse FTRAN's own nonzero threshold
        # (_csparse.c uses 1e-15 on the BTRAN result) and zero it in u too, so
        # the pattern handed to the component really is the set of rows whose
        # x_B changed. Without this, scipy's dense solve reports every
        # denormal as a nonzero and the pattern is spuriously dense.
        u[np.abs(u) < 1e-15] = 0.0
        changed = np.nonzero(u)[0].astype(np.int32)
        x_B -= t_prim * u
        entering_val = x_N[entering] + t_prim
        x_B[leaving] = entering_val

        status[leaving_var] = AT_LO if sigma > 0 else AT_HI
        x_N[leaving_var] = target
        status[entering] = BASIC
        x_N[entering] = 0.0
        basis_pos[leaving_var] = -1
        basis[leaving] = np.int32(entering)
        basis_pos[entering] = leaving

        if changed.size == 0 or changed[-1] != leaving:
            changed = np.union1d(changed, np.int32(leaving)).astype(np.int32)
        bench.set_column_density(changed.size / float(m))
        bench.rows_changed(changed, basis, x_B, lo_ext, hi_ext)

        pivots += 1
        if pivots % REFRESH_EVERY == 0:
            need_recompute = True
        if verbose and pivots % 500 == 0:
            infeas = int(
                np.count_nonzero(
                    (x_B < lo_ext[basis] - FEAS_TOL) | (x_B > hi_ext[basis] + FEAS_TOL)
                )
            )
            print(
                f"    pivot {pivots:6d} phase {phase} infeasible {infeas:5d} "
                f"({time.perf_counter() - t_start:.0f}s)"
            )

    rep: dict[str, Any] = dict(bench.report())
    rep.update(
        instance=name,
        rule="dse" if rule == RULE_DSE else "dantzig",
        list_mode=list_mode,
        cutoff=cutoff,
        random_start=random_start,
        m=m,
        n=n,
        n_total=n_total,
        pivots=pivots,
        phase1_pivots=phase1_pivots,
        phase=phase,
        outcome=outcome,
        wall_s=time.perf_counter() - t_start,
        no_placement_cols=n_free_phase2,
        weight_drift=drift,
    )
    if outcome == "optimal":
        x_full = x_N.copy()
        x_full[basis] = x_B
        rep["objective"] = float(c @ x_full[:n])
        rep["primal_residual"] = float(np.max(np.abs(A @ x_full[:n] - b)))
    return rep


# --------------------------------------------------------------------------
def summarise(rep: dict[str, Any], tsc_hz: float) -> str:
    calls = max(rep["calls"], 1.0)
    ds2 = rep["med_ds2"]
    upd = rep["med_update"]
    sca = rep["med_scalar"]
    avx = rep["med_avx2"]
    tot = ds2 + upd
    decided = rep["calls"] - rep["both_none"]
    agree = rep["agree_merit"] + rep["both_none"]
    return (
        f"{rep['instance']:>9} {rep['rule']:>7} m={int(rep['m']):5d} "
        f"pivots={int(rep['pivots']):6d} {rep['outcome']:<12} | "
        f"ds2 {ds2:8.0f} + upd {upd:7.0f} = {tot:8.0f} cyc  "
        f"scalar {sca:8.0f}  avx2 {avx:8.0f}  "
        f"speedup {sca / max(tot, 1e-9):5.2f}x/{avx / max(tot, 1e-9):5.2f}x | "
        f"agree {agree:.0f}/{rep['calls']:.0f} "
        f"(pos {rep['agree_pos']:.0f}/{decided:.0f}) "
        f"mismatch {rep['paranoid_mismatch']:.0f} | "
        f"list {rep['list_len_sum'] / calls:6.1f} "
        f"dense_calls {rep['dense_calls']:.0f} "
        f"rebuilds {rep['rebuilds']:.0f} "
        f"PAIRED avx2/ds2 {rep.get('ratio_avx2_over_ds2', float('nan')):5.2f}x "
        f"(scan only {rep.get('ratio_avx2_over_ds2_scan', float('nan')):5.2f}x) "
        f"changed/pivot {rep['changed_rows'] / max(rep['n_updates'], 1.0):6.1f} "
        f"({100 * rep['changed_rows'] / max(rep['n_updates'], 1.0) / max(rep['m'], 1.0):4.1f}% of m) "
        f"noop {rep.get('med_noop', 0.0):6.0f} "
        f"| {tot / tsc_hz * 1e6:.2f} us/pivot"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["degen2", "25fv47", "greenbea"])
    ap.add_argument("--lpsuite", type=Path, default=Path("/tmp/lpsuite"))
    ap.add_argument("--max-pivots", type=int, default=20000)
    ap.add_argument("--rules", nargs="+", default=["dantzig", "dse"])
    ap.add_argument("--audit-stride", type=int, default=250)
    ap.add_argument("--paranoid-stride", type=int, default=0)
    ap.add_argument("--list-modes", nargs="+", default=["on"])
    ap.add_argument("--out", type=Path, default=REPO / "probe_out" / "ds2-chuzr.json")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        build_library()
    probe = ctypes.CDLL(str(LIB))
    probe.ds2_bench_tsc_now.restype = c_double
    probe.ds2_bench_timer_overhead.restype = c_double
    probe.ds2_bench_timer_overhead.argtypes = [c_int64]
    probe.ds2_bench_has_avx2.restype = c_int
    tsc_hz = calibrate_tsc(probe)
    overhead = probe.ds2_bench_timer_overhead(200000)
    print(
        f"TSC {tsc_hz / 1e9:.3f} GHz; rdtscp bracket overhead "
        f"{overhead:.1f} cycles; AVX2 {bool(probe.ds2_bench_has_avx2())}"
    )

    rule_map = {"dantzig": RULE_DANTZIG, "dse": RULE_DSE}
    mode_map = {"off": LIST_OFF, "on": LIST_ON, "always": LIST_ALWAYS}

    results = []
    for name in args.instances:
        data = load_reduced(name, args.lpsuite)
        print(f"{name}: m={data['A'].shape[0]} n={data['A'].shape[1]} nnz={data['A'].nnz}")
        for rule_name in args.rules:
            for mode_name in args.list_modes:
                rep = run_instance(
                    name,
                    data,
                    rule_map[rule_name],
                    args.max_pivots,
                    list_mode=mode_map[mode_name],
                    paranoid_stride=args.paranoid_stride,
                    audit_stride=args.audit_stride,
                )
                rep["timer_overhead_cycles"] = overhead
                rep["tsc_hz"] = tsc_hz
                results.append(rep)
                print("  " + summarise(rep, tsc_hz))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
