"""Does ONE global churn setting help the WHOLE simplex class?

The no-per-problem-tuning rule means a mechanism is only shippable if a single
(alpha, deadband) improves the class. Three good instances prove nothing.

Measures PIVOTS (load-invariant on this shared box) and certifies every result:
status must be optimal AND the objective must match the baseline within eps=2e-5
relative. A setting that wins pivots by changing the answer is a kill, not a win.

Usage:
    PYTHONPATH=. uv run python experiments/churn_class_sweep.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
EPS = 2e-5

CELLS = [
    "lp_25fv47",
    "lp_agg2",
    "lp_agg3",
    "lp_cycle",
    "lp_degen2",
    "lp_fffff800",
    "lp_greenbea",
    "lp_greenbeb",
    "lp_israel",
    "lp_pilotnov",
    "lp_tuff",
]

# (alpha, deadband) -- global settings only. alpha=0 is the baseline.
GRID = [(0.0, 0), (0.5, 5), (1.0, 5), (2.0, 5), (4.0, 5), (1.0, 10), (2.0, 10), (1.0, 2)]


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


def run(matrix: Any, red: Any, alpha: float, deadband: int) -> dict[str, Any]:
    if alpha > 0.0:
        os.environ["LINPROGX_DS_CHURN_DANTZIG"] = "1"
        os.environ["LINPROGX_DS_CHURN_ALPHA"] = repr(alpha)
        os.environ["LINPROGX_DS_CHURN_DEADBAND"] = str(deadband)
        os.environ["LINPROGX_DS_CHURN_CAP"] = "1000"
        rule = 1
    else:
        for k in (
            "LINPROGX_DS_CHURN_DANTZIG",
            "LINPROGX_DS_CHURN_ALPHA",
            "LINPROGX_DS_CHURN_DEADBAND",
            "LINPROGX_DS_CHURN_CAP",
        ):
            os.environ.pop(k, None)
        rule = 1
    return matrix.solve_eq_box_dual_simplex(
        red.c, red.b, red.lo, red.hi, max_iter=100_000, leaving_rule=rule, expand=1
    )


def main() -> None:
    loaded = {}
    for name in CELLS:
        p = SUITE / f"{name}.mat"
        if not p.exists():
            print(f"  (missing {name})")
            continue
        pr = presolved(load(p))
        if pr is None:
            print(f"  (no presolve for {name})")
            continue
        loaded[name] = pr

    # baseline first, for objective certification
    base: dict[str, dict[str, Any]] = {}
    print(f"{'instance':14s} {'base piv':>9} {'status':>10}")
    for name, (m, red) in loaded.items():
        r = run(m, red, 0.0, 0)
        base[name] = r
        print(f"{name:14s} {int(r['iterations']):>9} {r['status']:>10}", flush=True)

    print(f"\n{'setting':>14} | " + " ".join(f"{n.replace('lp_', ''):>9}" for n in loaded))
    print("-" * (17 + 10 * len(loaded)))
    results = {}
    for alpha, dead in GRID:
        if alpha == 0.0:
            continue
        cells = []
        tot_b = tot_t = 0
        bad = []
        for name, (m, red) in loaded.items():
            r = run(m, red, alpha, dead)
            b = base[name]
            piv = int(r["iterations"])
            ok = r["status"] == "optimal" == b["status"]
            if ok and b.get("objective") is not None and r.get("objective") is not None:
                rel = abs(r["objective"] - b["objective"]) / max(1.0, abs(b["objective"]))
                ok = rel <= EPS
            if not ok:
                bad.append(f"{name}:{r['status']}")
                cells.append(f"{'FAIL':>9}")
            else:
                d = 100.0 * (piv - int(b["iterations"])) / max(1, int(b["iterations"]))
                cells.append(f"{d:>+8.1f}%")
                tot_b += int(b["iterations"])
                tot_t += piv
        cls = 100.0 * (tot_t - tot_b) / max(1, tot_b)
        results[(alpha, dead)] = (cls, len(bad))
        tag = f"a={alpha} d={dead}"
        print(
            f"{tag:>14} | "
            + " ".join(cells)
            + f"   CLASS {cls:+.2f}%"
            + (f"  BAD={len(bad)}" if bad else ""),
            flush=True,
        )
        if bad:
            print(f"{'':>14}   ! {', '.join(bad)}")

    print("\nBest global settings by class pivot change (certified only):")
    for (a, d), (cls, nbad) in sorted(results.items(), key=lambda kv: kv[1][0]):
        flag = "" if nbad == 0 else f"  <-- {nbad} UNCERTIFIED, disqualified"
        print(f"  alpha={a:<4} deadband={d:<3} class {cls:+.2f}%{flag}")


if __name__ == "__main__":
    main()
