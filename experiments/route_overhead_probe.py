"""Where does greenbea's PUBLIC-ROUTE time actually go?

Every attack of this campaign has targeted the dual simplex kernel. But the
board measures the PUBLIC route, and a first measurement shows the public route
costing ~538 ms CPU on greenbea where the bare presolved
`solve_eq_box_dual_simplex` call is ~377 ms in the ledger. If a large fraction
of the board cell is spent OUTSIDE the pivot loop, that is unattacked ground --
and greenbea needs only 13.46%.

Measures CPU time (load-invariant on this shared box, per the campaign's
measurement doctrine), not wall.

Usage:
    PYTHONPATH=. uv run python experiments/route_overhead_probe.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
EPS = 2e-5
REPS = 5


def cpu() -> float:
    return time.process_time()


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


def median(xs: list[float]) -> float:
    import statistics as st

    return st.median(xs)


def main() -> None:
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix, from_scipy_sparse

    d = load("lp_greenbea")
    bounds = [
        (None if lo == -INF else float(lo), None if hi == INF else float(hi))
        for lo, hi in zip(d["lo"], d["hi"], strict=True)
    ]

    # ---- stage 1: building the SparseLPProblem (Python-side marshalling) ----
    t = []
    for _ in range(REPS):
        c0 = cpu()
        mat = from_scipy_sparse(d["A"])
        SparseLPProblem(
            c=d["c"].tolist(), A_eq=mat, b_eq=d["b"].tolist(), objective="min", bounds=bounds
        )
        t.append(cpu() - c0)
    t_build = median(t) * 1e3

    problem = SparseLPProblem(
        c=d["c"].tolist(),
        A_eq=from_scipy_sparse(d["A"]),
        b_eq=d["b"].tolist(),
        objective="min",
        bounds=bounds,
    )

    # ---- stage 2: presolve alone ----
    t = []
    for _ in range(REPS):
        c0 = cpu()
        red = presolve_matrix(
            from_scipy_sparse(d["A"]),
            d["b"].tolist(),
            d["c"].tolist(),
            d["lo"].tolist(),
            d["hi"].tolist(),
            algorithm="auto",
        )
        t.append(cpu() - c0)
    t_presolve = median(t) * 1e3
    assert red is not None
    m = red._matrix or csr_matrix(red.rows, red.cols, red.indptr, red.indices, red.data)

    # ---- stage 3: the bare dual simplex on the presolved system ----
    t = []
    for _ in range(REPS):
        c0 = cpu()
        r = m.solve_eq_box_dual_simplex(
            red.c, red.b, red.lo, red.hi, max_iter=50_000, leaving_rule=1, expand=1
        )
        t.append(cpu() - c0)
    t_simplex = median(t) * 1e3
    pivots = int(r["iterations"])

    # ---- stage 4: the whole public route ----
    t = []
    for _ in range(REPS):
        c0 = cpu()
        res = SparseSolver(
            algorithm="auto", max_iterations=50_000, eps=EPS, check_interval=50_000
        ).solve(problem)
        t.append(cpu() - c0)
    t_route = median(t) * 1e3

    other = t_route - t_presolve - t_simplex
    print(f"greenbea, median of {REPS}, CPU ms (load-invariant)\n")
    print(f"  problem construction (not in route) {t_build:8.1f}")
    print(f"  {'-' * 46}")
    print(
        f"  presolve                            {t_presolve:8.1f}   "
        f"{100 * t_presolve / t_route:5.1f}%"
    )
    print(
        f"  dual simplex ({pivots} pivots)        {t_simplex:8.1f}   "
        f"{100 * t_simplex / t_route:5.1f}%"
    )
    print(f"  everything else (route/cert/glue)   {other:8.1f}   {100 * other / t_route:5.1f}%")
    print(f"  {'-' * 46}")
    print(f"  PUBLIC ROUTE TOTAL                  {t_route:8.1f}   100.0%")
    print(f"\n  us/pivot inside the simplex: {1000 * t_simplex / pivots:.1f}")
    print(f"  status: {res.solution.status.value}  backend: {res.backend}")
    print("\n  greenbea needs 13.46% to flip its board cell.")
    nonsimplex = 100 * (t_route - t_simplex) / t_route
    print(f"  Non-simplex share of the board cell: {nonsimplex:.1f}%")


if __name__ == "__main__":
    main()
