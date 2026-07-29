"""DS2 driver: run one LPnetlib instance through the dual simplex path.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

Loads an LPnetlib .mat instance, applies the shipped presolve, and calls the
native dual simplex directly -- either the shipped entry point
(`solve_eq_box_dual_simplex`) or the DS2 rewrite (`solve_eq_box_ds2`).

Usage:
    PYTHONPATH=. uv run python experiments/ds2_run.py greenbea --ds2
    PYTHONPATH=. uv run python experiments/ds2_run.py greenbea            # baseline
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from linprogx.presolve import postsolve_x, presolve_matrix
from linprogx.sparse import csr_matrix, from_scipy_sparse

DEFAULT_LPSUITE = Path("/tmp/lpsuite")


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


def prepare(name: str, suite: Path) -> dict[str, Any]:
    data = load_instance(suite / f"lp_{name}.mat")
    matrix = from_scipy_sparse(data["A_scipy"])
    b = [float(v) for v in data["b"]]
    c = [float(v) for v in data["c"]]
    lo = [float(v) for v in data["lo"]]
    hi = [float(v) for v in data["hi"]]

    reduction = presolve_matrix(matrix, b, c, lo, hi)
    if reduction is not None:
        if reduction._matrix is not None:
            solve_matrix = reduction._matrix
        else:
            solve_matrix = csr_matrix(
                reduction.rows,
                reduction.cols,
                reduction.indptr,
                reduction.indices,
                reduction.data,
            )
        solve_b, solve_c = reduction.b, reduction.c
        solve_lo, solve_hi = reduction.lo, reduction.hi
    else:
        solve_matrix = matrix
        solve_b, solve_c = b, c
        solve_lo, solve_hi = lo, hi

    return {
        "matrix": solve_matrix,
        "orig_matrix": matrix,
        "b": b,
        "c": c,
        "solve_b": solve_b,
        "solve_c": solve_c,
        "solve_lo": solve_lo,
        "solve_hi": solve_hi,
        "reduction": reduction,
    }


def run(name: str, suite: Path, ds2: bool, max_iter: int, **kwargs: Any) -> dict[str, Any]:
    prepared = prepare(name, suite)
    matrix = prepared["matrix"]
    method = matrix.solve_eq_box_ds2 if ds2 else matrix.solve_eq_box_dual_simplex
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

    # Max equality residual in ORIGINAL units.
    residual = 0.0
    row_values = prepared["orig_matrix"].matvec(x)
    for value, rhs in zip(row_values, prepared["b"], strict=True):
        residual = max(residual, abs(float(value) - float(rhs)))

    return {
        "instance": name,
        "status": result["status"],
        "iterations": int(result["iterations"]),
        "objective": objective,
        "residual": residual,
        "seconds": elapsed,
        "raw": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--suite", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--ds2", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100_000)
    parser.add_argument("--expand", type=int, default=None)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.expand is not None and not args.ds2:
        kwargs["expand"] = args.expand

    for name in args.instances:
        out = run(name, args.suite, args.ds2, args.max_iter, **kwargs)
        print(
            f"{out['instance']:>10}  {'DS2' if args.ds2 else 'DS1'}  "
            f"status={out['status']:<16} iters={out['iterations']:>7}  "
            f"obj={out['objective']:.8f}  resid={out['residual']:.3e}  "
            f"{out['seconds'] * 1e3:.1f} ms"
        )


if __name__ == "__main__":
    main()
