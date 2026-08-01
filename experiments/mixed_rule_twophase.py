"""PROVENANCE: SOURCE-INFORMED (HiGHS).

Do the two phases of the dual simplex want DIFFERENT pricing rules?

The recorded bound-swap sweep says yes, and nobody ran the combination:

    arm            phase1   phase2   total
    Dantzig         2,418    2,399    4,817
    Devex           5,603    2,239    7,842
    exact DSE       5,198    1,883    7,081     <- best phase 2
                    ^^^^^ Dantzig wins phase 1 by 2.1x

Dantzig is 2.1x better in phase 1; DSE is 1.27x better in phase 2. A rule keyed
on the PHASE is a global mechanism -- it is not per-problem tuning, since the
same rule applies to every instance.

Mechanistically this is the DSE weight-INITIALISATION problem. Exact DSE weights
are meaningful only once they reflect the current basis; from a cold crash basis
they are approximations, and a bad approximation makes DSE behave worse than
Dantzig. By phase 2 the basis has settled, and DSE's weights earn their cost.
That is exactly the shape of the table above.

Uses the existing EXPORT_BASIS / WARM_START hooks so no solver change is needed.

Usage:
    PYTHONPATH=. uv run python experiments/mixed_rule_twophase.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
RULES = {1: "dantzig", 0: "devex", 5: "exactDSE"}
CELLS = ["lp_greenbea", "lp_25fv47", "lp_degen2", "lp_greenbeb"]
HIGHS = {"lp_greenbea": 2836, "lp_25fv47": 3033, "lp_degen2": 537, "lp_greenbeb": 4902}


def load(name: str) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(SUITE / f"{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def presolved(d: dict[str, Any]) -> Any:
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    red = presolve_matrix(
        from_scipy_sparse(d["A"]),
        d["b"].tolist(),
        d["c"].tolist(),
        d["lo"].tolist(),
        d["hi"].tolist(),
        algorithm="auto",
    )
    if red is None:
        return None
    m = red._matrix
    if m is None:
        m = csr_matrix(red.rows, red.cols, red.indptr, red.indices, red.data)
    return m, red


def phase1_bounds(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """HiGHS's dual Phase-1 bound map. Independent reimplementation."""
    p_lo: list[float] = []
    p_hi: list[float] = []
    for low, up in zip(lo, hi, strict=True):
        lf, hf = low != -INF, up != INF
        if not lf and not hf:
            p_lo.append(-1000.0)
            p_hi.append(1000.0)
        elif not lf:
            p_lo.append(-1.0)
            p_hi.append(0.0)
        elif not hf:
            p_lo.append(0.0)
            p_hi.append(1.0)
        else:
            p_lo.append(0.0)
            p_hi.append(0.0)
    return p_lo, p_hi


def main() -> None:
    for name in CELLS:
        p = SUITE / f"{name}.mat"
        if not p.exists():
            continue
        pr = presolved(load(name))
        if pr is None:
            continue
        m, red = pr
        rows = m.shape[0]
        p_lo, p_hi = phase1_bounds(list(red.lo), list(red.hi))
        zero_b = [0.0] * rows

        # single-phase baselines (the shipped big-M path)
        base = {}
        for r in (1, 5):
            os.environ.pop("LINPROGX_DS_EXPORT_BASIS", None)
            os.environ.pop("LINPROGX_DS_WARM_START", None)
            res = m.solve_eq_box_dual_simplex(
                red.c, red.b, red.lo, red.hi, max_iter=100_000, leaving_rule=r, expand=1
            )
            base[r] = (int(res["iterations"]), res["status"])

        print(f"\n=== {name}   (HiGHS {HIGHS.get(name, '?')}) ===")
        for r, (it, st) in base.items():
            print(f"  single-phase big-M {RULES[r]:9s} {it:>7} {st}")

        print(
            f"  {'ph1 rule':>10} {'ph2 rule':>10} {'ph1':>7} {'ph2':>7} {'total':>7} {'vs HiGHS':>9}  status"
        )
        best = None
        for r1 in (1, 0, 5):
            os.environ["LINPROGX_DS_EXPORT_BASIS"] = "1"
            os.environ.pop("LINPROGX_DS_WARM_START", None)
            ph1 = m.solve_eq_box_dual_simplex(
                red.c, zero_b, p_lo, p_hi, max_iter=100_000, leaving_rule=r1, expand=1
            )
            if "basis" not in ph1:
                print("    (basis export hook did not fire)")
                break
            i1 = int(ph1["iterations"])
            for r2 in (1, 0, 5):
                os.environ["LINPROGX_DS_WARM_START"] = "1"
                ph2 = m.solve_eq_box_dual_simplex(
                    red.c,
                    red.b,
                    red.lo,
                    red.hi,
                    max_iter=100_000,
                    leaving_rule=r2,
                    expand=1,
                    initial_basis=ph1["basis"],
                    initial_bound_status=ph1["bound_status"],
                )
                i2 = int(ph2["iterations"])
                tot = i1 + i2
                ok = ph2["status"] == "optimal"
                ratio = tot / HIGHS[name] if name in HIGHS else float("nan")
                mark = ""
                if ok and (best is None or tot < best[0]):
                    best = (tot, r1, r2)
                    mark = "  <-- best so far"
                print(
                    f"  {RULES[r1]:>10} {RULES[r2]:>10} {i1:>7} {i2:>7} {tot:>7} "
                    f"{ratio:>8.3f}x  {ph2['status']}{mark}",
                    flush=True,
                )
        if best:
            tot, r1, r2 = best
            b1 = base[1][0]
            print(
                f"  BEST: ph1={RULES[r1]} ph2={RULES[r2]} total={tot} "
                f"({100.0 * (tot - b1) / b1:+.1f}% vs shipped big-M dantzig {b1})"
            )


if __name__ == "__main__":
    main()
