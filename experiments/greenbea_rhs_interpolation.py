"""PROVENANCE: SOURCE-INFORMED (HiGHS) era, but the experiment is independent.

WHY DOES EXACT DSE INVERT ON GREENBEA ALONE?

Exact DSE improves the trajectory on eight of the nine simplex-routed cells and
degrades exactly one: greenbea. Not size, not the big-M basis, not a phase.

Then this turned up. greenbea and greenbeb are LITERALLY THE SAME LP:

    A identical    (2392 x 5598, nnz 31070, elementwise difference = 0)
    b identical    (all zeros)
    c identical    (622 nonzeros)

They differ in 333 BOUND values, of which 292 are FIXED variables (lo == hi)
carrying different fixed values. Since b = 0, those fixed values ARE the
right-hand side. greenbea and greenbeb are one model under two demand scenarios.

And they respond OPPOSITELY to DSE:
    greenbea  Dantzig 4,399 -> DSE 4,675   (DSE 1.06x WORSE)
    greenbeb  Dantzig 8,919 -> DSE 5,633   (DSE 1.58x BETTER)

So DSE's advantage here is a function of the RIGHT-HAND SIDE ALONE, with the
matrix and costs held exactly fixed. This interpolates between the two demand
vectors and measures where the crossover is -- which decides whether greenbea is
an unlucky draw on a continuum, or something categorically different.

Only the 292 fixed columns are interpolated: they are all finite, so no bound
KIND changes and the comparison stays clean.

Usage:
    PYTHONPATH=. uv run python experiments/greenbea_rhs_interpolation.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def get(name: str) -> dict[str, Any]:
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


def solve(d: dict[str, Any], rule: int) -> tuple[int, str]:
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
        return -1, "no-presolve"
    m = red._matrix
    if m is None:
        m = csr_matrix(red.rows, red.cols, red.indptr, red.indices, red.data)
    # churn OFF so this measures the RULE, not the shipped penalty
    os.environ["LINPROGX_DS_CHURN_DANTZIG"] = "0"
    r = m.solve_eq_box_dual_simplex(
        red.c, red.b, red.lo, red.hi, max_iter=100_000, leaving_rule=rule, expand=1
    )
    return int(r["iterations"]), r["status"]


def main() -> None:
    import numpy as np

    a, b = get("lp_greenbea"), get("lp_greenbeb")
    fixed_a = np.isfinite(a["lo"]) & np.isfinite(a["hi"]) & (a["lo"] == a["hi"])
    fixed_b = np.isfinite(b["lo"]) & np.isfinite(b["hi"]) & (b["lo"] == b["hi"])
    both_fixed = fixed_a & fixed_b
    differ = both_fixed & (a["lo"] != b["lo"])
    idx = np.where(differ)[0]
    print(f"interpolating {len(idx)} columns that are FIXED in both and differ in value")
    print(f"  greenbea sum of fixed values {a['lo'][idx].sum():.6g}")
    print(f"  greenbeb sum of fixed values {b['lo'][idx].sum():.6g}\n")

    print(f"{'t':>6} {'dantzig':>9} {'DSE':>9} {'DSE/dantzig':>12}   verdict")
    print("-" * 56)
    for t in FRACTIONS:
        d = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in a.items()}
        d["lo"] = a["lo"].copy()
        d["hi"] = a["hi"].copy()
        d["lo"][idx] = (1.0 - t) * a["lo"][idx] + t * b["lo"][idx]
        d["hi"][idx] = d["lo"][idx]
        p1, s1 = solve(d, 1)
        p5, s5 = solve(d, 5)
        if p1 <= 0 or p5 <= 0 or s1 != "optimal" or s5 != "optimal":
            print(f"{t:>6.2f} {p1:>9} {p5:>9} {'--':>12}   {s1}/{s5}")
            continue
        ratio = p5 / p1
        verdict = "DSE WINS" if ratio < 1.0 else "DSE loses"
        print(f"{t:>6.2f} {p1:>9} {p5:>9} {ratio:>12.3f}   {verdict}", flush=True)

    print("\nt=0 is greenbea's demand vector, t=1 is greenbeb's, on an IDENTICAL")
    print("matrix and cost vector. A smooth crossover means greenbea is an unlucky")
    print("draw on a continuum; a cliff at t=0 means it is categorically different.")


if __name__ == "__main__":
    main()
