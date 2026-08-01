"""Opportunity ceiling for the W2-B targets.

Splits the shipped whole cell into the C simplex call and everything else
(problem construction, presolve, postsolve, objective and residual).  The
non-solve remainder is a floor no trajectory or per-pivot mechanism can go
below, so ``floor / whole_cell`` is a hard lower bound on any candidate ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.suite_bench import bounds_of, load_instance  # noqa: E402

INSTANCES = ("lp_25fv47", "lp_degen2", "lp_greenbeb", "lp_greenbea")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--fixtures", default="/tmp/lpsuite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np

    from linprogx.presolve import aggressive_aggregate_for_ds2, postsolve_x, presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    out = {}
    for name in INSTANCES:
        data = load_instance(Path(args.fixtures) / f"{name}.mat")
        totals, solves = [], []
        for _ in range(args.repeats):
            t0 = time.process_time()
            matrix = from_scipy_sparse(data["A_scipy"])
            b = data["b"].tolist()
            c = data["c"].tolist()
            bounds = bounds_of(data)
            lo = [float("-inf") if lo_ is None else float(lo_) for lo_, _ in bounds]
            hi = [float("inf") if hi_ is None else float(hi_) for _, hi_ in bounds]
            red = presolve_matrix(matrix, b, c, lo, hi, algorithm="auto")
            if red is None:
                raise SystemExit(f"{name}: presolve returned no reduction")
            agg = aggressive_aggregate_for_ds2(red)
            if agg is not None:
                red = agg
            work = (
                red._matrix
                if red._matrix is not None
                else csr_matrix(red.rows, red.cols, red.indptr, red.indices, red.data)
            )
            s0 = time.process_time()
            if agg is not None:
                raw = work.solve_eq_box_ds2(red.c, red.b, red.lo, red.hi, max_iter=50_000)
            else:
                raw = work.solve_eq_box_dual_simplex(
                    red.c, red.b, red.lo, red.hi, max_iter=50_000, leaving_rule=1, expand=1
                )
            solve = time.process_time() - s0
            x = postsolve_x([float(v) for v in raw["x"]], red)
            _ = sum(v * coef for v, coef in zip(x, c, strict=True))
            _ = float(np.max(np.abs(data["A_scipy"] @ np.array(x) - data["b"])))
            totals.append(time.process_time() - t0)
            solves.append(solve)
        total = statistics.median(totals)
        solve = statistics.median(solves)
        out[name] = {
            "repeats": args.repeats,
            "cell_cpu_median_s": total,
            "simplex_call_cpu_median_s": solve,
            "non_solve_floor_cpu_s": total - solve,
            "floor_fraction_of_cell": (total - solve) / total,
            "min_attainable_ratio_at_zero_solve_time": (total - solve) / total,
        }
        print(name, json.dumps(out[name]), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
