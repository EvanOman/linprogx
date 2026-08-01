"""PROVENANCE: SOURCE-INFORMED (HiGHS).

Does exact DSE + churn win the WHOLE simplex class?

Two independent findings collided:
  * DS2-CHUZR measured that exact DSE (leaving_rule=5) is a huge win on the
    class -- 25fv47 8,300 -> 2,613 (BELOW HiGHS's 3,033), degen2 1,447 -> 653 --
    but LOSES greenbea (4,399 -> 4,675). The campaign's recorded "exact DSE is
    worse" verdict was a GREENBEA-ONLY fact, taken on the cold big-M path.
  * The churn penalty is the one mechanism that helps greenbea specifically.

So DSE+churn is the untested cell. If it holds DSE's class win AND repairs
greenbea, it is a single global rule that beats HiGHS on trajectory across the
class -- with no per-problem tuning.

Certifies every result: status optimal AND objective within eps=2e-5 relative of
the shipped Dantzig baseline. A pivot win that changes the answer is a kill.

Usage:
    PYTHONPATH=. uv run python experiments/dse_churn_class_sweep.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
EPS = 2e-5
CELLS = [
    "lp_25fv47",
    "lp_agg2",
    "lp_agg3",
    "lp_degen2",
    "lp_fffff800",
    "lp_greenbea",
    "lp_greenbeb",
    "lp_israel",
    "lp_tuff",
]

# (label, leaving_rule, churn_alpha or None, deadband)
ARMS = [
    ("dantzig (ships)", 1, None, 0),
    ("DSE", 5, None, 0),
    ("DSE+churn a=0.5 d=5", 5, 0.5, 5),
    ("DSE+churn a=1.0 d=5", 5, 1.0, 5),
    ("DSE+churn a=2.0 d=5", 5, 2.0, 5),
    ("DSE+churn a=1.0 d=2", 5, 1.0, 2),
    ("DSE+churn a=0.5 d=10", 5, 0.5, 10),
]

HIGHS = {
    "lp_25fv47": 3033,
    "lp_agg2": 534,
    "lp_agg3": 563,
    "lp_degen2": 537,
    "lp_fffff800": 424,
    "lp_greenbea": 2836,
    "lp_greenbeb": 4902,
    "lp_israel": 240,
    "lp_tuff": 174,
}


def load(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
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


def run(m: Any, red: Any, rule: int, alpha: float | None, dead: int) -> dict[str, Any]:
    for k in (
        "LINPROGX_DS_CHURN_DANTZIG",
        "LINPROGX_DS_CHURN_DSE",
        "LINPROGX_DS_CHURN_ALPHA",
        "LINPROGX_DS_CHURN_DEADBAND",
        "LINPROGX_DS_CHURN_CAP",
    ):
        os.environ.pop(k, None)
    if alpha is not None:
        os.environ["LINPROGX_DS_CHURN_DSE" if rule == 5 else "LINPROGX_DS_CHURN_DANTZIG"] = "1"
        os.environ["LINPROGX_DS_CHURN_ALPHA"] = repr(alpha)
        os.environ["LINPROGX_DS_CHURN_DEADBAND"] = str(dead)
        os.environ["LINPROGX_DS_CHURN_CAP"] = "1000"
    return m.solve_eq_box_dual_simplex(
        red.c, red.b, red.lo, red.hi, max_iter=100_000, leaving_rule=rule, expand=1
    )


def main() -> None:
    loaded = {}
    for name in CELLS:
        p = SUITE / f"{name}.mat"
        if p.exists():
            pr = presolved(load(p))
            if pr is not None:
                loaded[name] = pr

    base = {n: run(m, r, 1, None, 0) for n, (m, r) in loaded.items()}
    hdr = " ".join(f"{n.replace('lp_', ''):>9}" for n in loaded)
    print(f"{'arm':>21} | {hdr}     CLASS   vs HiGHS")
    print("-" * (24 + 10 * len(loaded) + 20))

    for label, rule, alpha, dead in ARMS:
        cells, tot, hx_tot, bad = [], 0, 0, []
        for name, (m, red) in loaded.items():
            r = run(m, red, rule, alpha, dead)
            piv = int(r["iterations"])
            ok = r["status"] == "optimal"
            if ok and base[name].get("objective") is not None and r.get("objective") is not None:
                rel = abs(r["objective"] - base[name]["objective"]) / max(
                    1.0, abs(base[name]["objective"])
                )
                ok = rel <= EPS
            if not ok:
                bad.append(f"{name}:{r['status']}")
                cells.append(f"{'FAIL':>9}")
            else:
                cells.append(f"{piv:>9}")
                tot += piv
                hx_tot += HIGHS.get(name, 0)
        ratio = tot / hx_tot if hx_tot else float("nan")
        print(
            f"{label:>21} | "
            + " ".join(cells)
            + f"  {tot:>8}   {ratio:>6.3f}x"
            + (f"  BAD={len(bad)}" if bad else ""),
            flush=True,
        )
        if bad:
            print(f"{'':>21}   ! {', '.join(bad)}")
    print(
        f"\n{'HiGHS':>21} | "
        + " ".join(f"{HIGHS.get(n, 0):>9}" for n in loaded)
        + f"  {sum(HIGHS.get(n, 0) for n in loaded):>8}   1.000x"
    )
    print("\nCLASS ratio < 1.0 means linprogx beats HiGHS on total class trajectory.")


if __name__ == "__main__":
    main()
