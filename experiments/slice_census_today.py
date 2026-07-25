"""Today's greenbea slice census: 13-phase profile + solve-slice anatomy.

The inherited slice shares (37% solves / 16.92% BTRAN) predate the shipped SIMD
pricing unit.  This re-measures them on the current checkout.  Phase SHARES are
far more load-robust than absolute walls, which matters because this box runs at
load ~11/12 cores; absolute microseconds here are indicative only.

Usage:
    PYTHONPATH=. LINPROGX_DS_SOLVE_SLICE=1 uv run python experiments/slice_census_today.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="lp_greenbea")
    parser.add_argument("--out", default="/tmp/slice_census_today.json")
    args = parser.parse_args()

    os.environ.setdefault("LINPROGX_DS_SOLVE_SLICE", "1")

    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    data = load_instance(SUITE / f"{args.instance}.mat")
    original = from_scipy_sparse(data["A_scipy"])
    c, b = data["c"].tolist(), data["b"].tolist()
    lo, hi = data["lo"].tolist(), data["hi"].tolist()

    reduction = presolve_matrix(original, b, c, lo, hi, algorithm="auto")
    if reduction._matrix is not None:
        matrix = reduction._matrix
    else:
        matrix = csr_matrix(
            reduction.rows, reduction.cols, reduction.indptr,
            reduction.indices, reduction.data,
        )
    print(f"presolved shape   {matrix.shape}  nnz={matrix.nnz}")

    out = matrix.solve_eq_box_dual_simplex(
        reduction.c, reduction.b, reduction.lo, reduction.hi,
        max_iter=50_000, leaving_rule=1, expand=1,
    )
    iters = int(out["iterations"])
    print(f"status            {out['status']}")
    print(f"iterations        {iters}")

    phases = out.get("phase_us") or {}
    total = sum(phases.values())
    print(f"\n--- 13-phase profile (sum {total / 1e3:.2f} ms; shares are the robust part) ---")
    for name, us in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"{name:18s} {us / 1e3:9.3f} ms  {100.0 * us / total:6.2f}%  "
              f"{us / max(1, iters):8.3f} us/pivot")

    slice_info = out.get("solve_slice") or {}
    if slice_info:
        print("\n--- solve-slice anatomy ---")
        for name, value in sorted(slice_info.items()):
            if name.endswith(("_us", "dense", "sparse", "total")) and isinstance(value, float):
                print(f"{name:22s} {value / 1e3:9.3f} ms")
            else:
                print(f"{name:22s} {value}")

    grouped = {
        "solves (btran_rho+pivot_row+ftran_col)":
            phases.get("btran_rho", 0) + phases.get("pivot_row", 0) + phases.get("ftran_col", 0),
        "btran_rho alone": phases.get("btran_rho", 0),
        "factor (lu_update+refactor)":
            phases.get("lu_update", 0) + phases.get("refactor", 0),
        "pricing (leaving_scan+pricing_update)":
            phases.get("leaving_scan", 0) + phases.get("pricing_update", 0),
        "ratio_test": phases.get("ratio_test", 0),
        "rcost_update": phases.get("rcost_update", 0),
    }
    print("\n--- grouped shares vs inherited framing ---")
    for name, us in grouped.items():
        print(f"{name:42s} {100.0 * us / total:6.2f}%")

    Path(args.out).write_text(json.dumps(
        {"iterations": iters, "status": out["status"], "phase_us": phases,
         "solve_slice": {k: v for k, v in slice_info.items()},
         "presolved_shape": list(matrix.shape), "presolved_nnz": matrix.nnz},
        indent=2, default=str))
    print(f"\nartifact: {args.out}")


if __name__ == "__main__":
    main()
