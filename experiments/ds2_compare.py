"""DS1 vs DS2 iteration comparison on the simplex-routed LPnetlib instances.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

Iteration counts are load-invariant and are the target; wall is reported only
as context and drifts 4-19% on this box.

    PYTHONPATH=. uv run python experiments/ds2_compare.py greenbea 25fv47 ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.ds2_run import DEFAULT_LPSUITE, prepare
from linprogx.presolve import postsolve_x


def solve(prepared, ds2: bool, max_iter: int):
    matrix = prepared["matrix"]
    if ds2:
        method = matrix.solve_eq_box_ds2
        kwargs = {}
    else:
        # The shipped auto route's dual-simplex configuration.
        method = matrix.solve_eq_box_dual_simplex
        kwargs = {"leaving_rule": 1, "expand": 1}
    start = time.perf_counter()
    result = method(
        prepared["solve_c"],
        prepared["solve_b"],
        prepared["solve_lo"],
        prepared["solve_hi"],
        max_iter=max_iter,
        **kwargs,
    )
    elapsed = time.perf_counter() - start
    x = [float(v) for v in result["x"]]
    if prepared["reduction"] is not None:
        x = postsolve_x(x, prepared["reduction"])
    objective = sum(v * coef for v, coef in zip(x, prepared["c"], strict=True))
    residual = 0.0
    for value, rhs in zip(prepared["orig_matrix"].matvec(x), prepared["b"], strict=True):
        residual = max(residual, abs(float(value) - float(rhs)))
    return {
        "status": result["status"],
        "iterations": int(result["iterations"]),
        "objective": objective,
        "residual": residual,
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--suite", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--max-iter", type=int, default=100_000)
    args = parser.parse_args()

    header = (
        f"{'instance':>10} {'DS1 iters':>10} {'DS2 iters':>10} {'ratio':>7}  "
        f"{'DS1 status':<16} {'DS2 status':<16} {'obj match':>10} "
        f"{'DS1 ms':>9} {'DS2 ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for name in args.instances:
        prepared = prepare(name, args.suite)
        one = solve(prepared, False, args.max_iter)
        two = solve(prepared, True, args.max_iter)
        ratio = two["iterations"] / one["iterations"] if one["iterations"] else float("nan")
        scale = max(1.0, abs(one["objective"]))
        match = abs(one["objective"] - two["objective"]) / scale
        print(
            f"{name:>10} {one['iterations']:>10} {two['iterations']:>10} "
            f"{ratio:>7.2f}  {one['status']:<16} {two['status']:<16} "
            f"{match:>10.2e} {one['seconds'] * 1e3:>9.1f} {two['seconds'] * 1e3:>9.1f}"
        )


if __name__ == "__main__":
    main()
