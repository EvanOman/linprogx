"""PROVENANCE: independent experiment (no HiGHS knowledge used).

WHICH BOUNDS CARRY GREENBEA'S DSE INVERSION?

greenbea and greenbeb are the SAME LP -- identical A (2392x5598, nnz 31070,
elementwise difference 0), identical b (all zeros), identical c -- differing only
in 333 bound values. Yet:

    greenbea  Dantzig 4,399 -> DSE 4,675   (DSE 1.06x WORSE)
    greenbeb  Dantzig 8,919 -> DSE 5,633   (DSE 1.58x BETTER)

A first attempt interpolated only the 82 columns FIXED in both, and did not flip
the verdict -- but it also never reached greenbeb, so it only rules those 82 out.

This patches ALL 333 differing columns, a graded fraction at a time, so that
f=1.0 reproduces greenbeb EXACTLY. That endpoint is the validity check: if f=1.0
does not give greenbeb's 8,919/5,633, the patch is not doing what it claims and
nothing else in the table can be believed.

The 333 differ by KIND as well as value (292 boxed->boxed, 18 lower->boxed,
14 lower->lower, 5 boxed->lower, 4 lower->free), so columns are swapped wholesale
rather than interpolated. Columns are patched in index order, which is arbitrary
with respect to the algorithm and therefore a fair ramp.

Usage:
    PYTHONPATH=. uv run python experiments/greenbea_bound_patch.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
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
    os.environ["LINPROGX_DS_CHURN_DANTZIG"] = "0"  # measure the RULE, not the ship
    r = m.solve_eq_box_dual_simplex(
        red.c, red.b, red.lo, red.hi, max_iter=100_000, leaving_rule=rule, expand=1
    )
    return int(r["iterations"]), r["status"]


def main() -> None:
    import numpy as np

    a, b = get("lp_greenbea"), get("lp_greenbeb")
    same_lo = (a["lo"] == b["lo"]) | (np.isnan(a["lo"]) & np.isnan(b["lo"]))
    same_hi = (a["hi"] == b["hi"]) | (np.isnan(a["hi"]) & np.isnan(b["hi"]))
    idx = np.where(~(same_lo & same_hi))[0]
    print(f"patching {len(idx)} differing bound columns, in index order\n")
    print(f"{'f':>6} {'cols':>6} {'dantzig':>9} {'DSE':>9} {'DSE/dtz':>9}   verdict")
    print("-" * 58)
    for f in FRACTIONS:
        k = int(round(f * len(idx)))
        sel = idx[:k]
        d = dict(a)
        d["lo"] = a["lo"].copy()
        d["hi"] = a["hi"].copy()
        d["lo"][sel] = b["lo"][sel]
        d["hi"][sel] = b["hi"][sel]
        p1, s1 = solve(d, 1)
        p5, s5 = solve(d, 5)
        if p1 <= 0 or p5 <= 0 or s1 != "optimal" or s5 != "optimal":
            print(f"{f:>6.2f} {k:>6} {p1:>9} {p5:>9} {'--':>9}   {s1}/{s5}")
            continue
        ratio = p5 / p1
        note = "DSE WINS" if ratio < 1.0 else "DSE loses"
        if f == 1.0:
            note += "   <-- must equal greenbeb 8919/5633"
        print(f"{f:>6.2f} {k:>6} {p1:>9} {p5:>9} {ratio:>9.3f}   {note}", flush=True)

    print("\nVALIDITY: the f=1.00 row must reproduce greenbeb (8,919 / 5,633).")
    print("If it does not, the patch is incomplete and the ramp means nothing.")


if __name__ == "__main__":
    main()
