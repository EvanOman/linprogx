"""Which LPnetlib cells actually execute the DS kernel the endgame units touch?

The Harris early-outs and the narrow CSR index cache live inside
CSRMatrix_solve_eq_box_dual_simplex.  Cells whose PUBLIC route is IPM or PDHG
never execute that code and are bit-identical for a trivial reason.  This
census records the public backend per instance so a v3 cert can be scoped to
the cells that are genuinely touched, instead of paying for 23 cells to prove
that 20 of them never ran the changed code.

Usage:
    PYTHONPATH=. uv run python experiments/route_census.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
# lp_qap15 is coverage-only: HiGHS times out at 300s, and the campaign protocol
# forbids it in paired mode.  Excluded here too so the census cannot suggest it.
SKIP = {"lp_qap15"}


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


def bounds_of(data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    return [
        (None if low == float("-inf") else float(low),
         None if up == float("inf") else float(up))
        for low, up in zip(data["lo"], data["hi"], strict=True)
    ]


def main() -> None:
    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    rows = []
    for path in sorted(SUITE.glob("lp_*.mat")):
        name = path.stem
        if name in SKIP:
            print(f"{name:16s} SKIPPED (coverage-only, forbidden in paired mode)")
            continue
        data = load_instance(path)
        problem = SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=from_scipy_sparse(data["A_scipy"]),
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )
        begin = time.perf_counter()
        try:
            result = SparseSolver(
                algorithm="auto", max_iterations=50_000, eps=2e-5,
                check_interval=50_000,
            ).solve(problem)
            backend = result.backend
            status = result.solution.status.value
            iters = result.solution.iterations
        except Exception as exc:  # noqa: BLE001 - census must not abort
            backend, status, iters = f"ERROR:{type(exc).__name__}", "error", 0
        wall = time.perf_counter() - begin
        touched = "dual-simplex" in backend
        rows.append({"instance": name, "backend": backend, "status": status,
                     "iterations": iters, "seconds": wall, "ds_touched": touched})
        print(f"{name:16s} {backend:34s} {status:10s} it={iters:<7d} "
              f"{wall:8.3f}s  {'DS-TOUCHED' if touched else ''}")

    touched = [r["instance"] for r in rows if r["ds_touched"]]
    print(f"\nDS-touched cells ({len(touched)}): {','.join(touched)}")
    Path("/tmp/route_census.json").write_text(json.dumps(rows, indent=2))
    print("artifact: /tmp/route_census.json")


if __name__ == "__main__":
    main()
