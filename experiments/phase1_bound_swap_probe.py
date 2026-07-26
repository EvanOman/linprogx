"""PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

Validate the dual-Phase-1-by-bound-swap hypothesis BEFORE touching the C solver.

HiGHS runs dual Phase 1 as the SAME dual simplex on the SAME matrix, costs,
basis and factorization, with only the primal BOUNDS replaced by a synthetic
map (HEkk::initialiseBound):

    free (-inf,+inf) -> [-1000, 1000]
    upper-only       -> [-1,  0]
    lower-only       -> [ 0,  1]
    boxed or fixed   -> [ 0,  0]

Under that map every variable is boxed with finite bounds, so every nonbasic can
be placed on the side matching its reduced-cost sign: dual feasibility is
achievable BY CONSTRUCTION and no big-M is ever required.

linprogx instead invents big-M artificial bounds (M = 1e5 x scale) for every
column whose reduced cost points at an infinite bound.  On greenbea that is
3,611/3,868 columns (93.4%).

This probe simulates the two-phase structure using the existing diagnostic
hooks (LINPROGX_DS_EXPORT_BASIS + LINPROGX_DS_WARM_START), so the hypothesis can
be tested without a 2,700-line refactor:

  phase 1 : solve with the SAME b and costs but Phase-1 bounds, export the basis
  phase 2 : warm-start the TRUE problem from that basis, count pivots

DECISION RULE (predeclared): the hypothesis is supported only if
    phase1_pivots + phase2_pivots  <  4,399
by a margin large enough to matter after per-pivot costs.  Phase-1 pivots are
NOT free -- they are real dual simplex iterations on the same kernels -- so the
comparison is on the TOTAL.

Usage:
    PYTHONPATH=. uv run python experiments/phase1_bound_swap_probe.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")


def load_instance(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A_scipy": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def phase1_bounds(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """HiGHS's dual Phase-1 bound map.  Independent reimplementation."""
    p_lo: list[float] = []
    p_hi: list[float] = []
    for low, up in zip(lo, hi, strict=True):
        lo_fin = low != -INF
        hi_fin = up != INF
        if not lo_fin and not hi_fin:
            p_lo.append(-1000.0)
            p_hi.append(1000.0)
        elif not lo_fin:
            p_lo.append(-1.0)
            p_hi.append(0.0)
        elif not hi_fin:
            p_lo.append(0.0)
            p_hi.append(1.0)
        else:
            p_lo.append(0.0)
            p_hi.append(0.0)
    return p_lo, p_hi


def presolved(data: dict[str, Any]) -> Any:
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    original = from_scipy_sparse(data["A_scipy"])
    reduction = presolve_matrix(
        original,
        data["b"].tolist(),
        data["c"].tolist(),
        data["lo"].tolist(),
        data["hi"].tolist(),
        algorithm="auto",
    )
    if reduction is None:
        raise SystemExit("presolve returned no reduction")
    matrix = reduction._matrix
    if matrix is None:
        matrix = csr_matrix(
            reduction.rows,
            reduction.cols,
            reduction.indptr,
            reduction.indices,
            reduction.data,
        )
    return matrix, reduction


def main() -> None:
    data = load_instance(SUITE / "lp_greenbea.mat")
    matrix, red = presolved(data)
    rows, cols = matrix.shape
    print(f"presolved {rows} x {cols}, nnz={matrix.nnz}")

    one_sided = sum(
        1
        for low, up in zip(red.lo, red.hi, strict=True)
        if (low == -INF) != (up == INF) or (low == -INF and up == INF)
    )
    print(f"one-sided or free columns: {one_sided}/{cols} = {100.0 * one_sided / cols:.1f}%")

    # ---- baseline: the shipped big-M path -------------------------------
    base = matrix.solve_eq_box_dual_simplex(
        red.c, red.b, red.lo, red.hi, max_iter=50_000, leaving_rule=1, expand=1
    )
    print(f"\nBASELINE (big-M)   status={base['status']:10s} pivots={int(base['iterations'])}")

    # ---- phase 1: same b, same costs, Phase-1 bounds --------------------
    # CRITICAL: in HiGHS the RHS lives in the ROW LOGICALS (Ax - s = 0 with
    # l_r <= s <= u_r).  Under the Phase-1 map an equality row's logical is
    # boxed -> [0,0], so Phase 1 actually solves Ax = 0.  That IS the
    # b-invariance we proved clean-room, and it is exactly the homogeneous
    # auxiliary min c'x s.t. Ax=0, x in [0,1] from our own Fenchel derivation.
    p_lo, p_hi = phase1_bounds(list(red.lo), list(red.hi))
    zero_b = [0.0] * rows
    os.environ["LINPROGX_DS_EXPORT_BASIS"] = "1"
    ph1 = matrix.solve_eq_box_dual_simplex(
        red.c, zero_b, p_lo, p_hi, max_iter=50_000, leaving_rule=1, expand=1
    )
    p1_iters = int(ph1["iterations"])
    print(f"PHASE 1 (bound swap) status={ph1['status']:10s} pivots={p1_iters}")
    if "basis" not in ph1:
        raise SystemExit("basis export hook did not fire; cannot continue")

    # ---- phase 2: warm-start the TRUE problem from phase 1's basis ------
    os.environ["LINPROGX_DS_WARM_START"] = "1"
    ph2 = matrix.solve_eq_box_dual_simplex(
        red.c,
        red.b,
        red.lo,
        red.hi,
        max_iter=50_000,
        leaving_rule=1,
        expand=1,
        initial_basis=ph1["basis"],
        initial_bound_status=ph1["bound_status"],
    )
    p2_iters = int(ph2["iterations"])
    print(f"PHASE 2 (warm)      status={ph2['status']:10s} pivots={p2_iters}")

    total = p1_iters + p2_iters
    baseline = int(base["iterations"])
    print(f"\nTOTAL two-phase = {p1_iters} + {p2_iters} = {total}")
    print(f"baseline big-M  = {baseline}")
    delta = 100.0 * (baseline - total) / baseline
    print(f"pivot change    = {delta:+.2f}%")
    print(f"objective ph2   = {ph2.get('objective')!r}")
    print(f"objective base  = {base.get('objective')!r}")
    print(
        "\nVERDICT: "
        + (
            "SUPPORTED - two-phase uses fewer total pivots"
            if total < baseline
            else "NOT SUPPORTED - two-phase is not cheaper in pivots"
        )
    )


if __name__ == "__main__":
    main()
