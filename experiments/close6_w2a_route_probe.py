"""Reproduce the shipped baseline and record the realised route per case.

Prints one JSON object per instance: production-auto pivots / objective /
residual / backend / Solution.message (which discriminates the stall-predictor
shortcut from the post-IPM rescue), plus the presolve shapes and whether the
DS2 aggregation gate accepts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from close6_w2a_lib import CASES, EPS, FIXTURE_DIR, Prepared, load_instance, problem_of


def probe(name: str) -> dict[str, object]:
    from linprogx.sparse import SparseSolver

    data = load_instance(FIXTURE_DIR / f"{name}.mat")
    problem = problem_of(data)
    t0 = time.process_time()
    w0 = time.perf_counter()
    result = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=EPS, check_interval=50_000
    ).solve(problem)
    cpu = time.process_time() - t0
    wall = time.perf_counter() - w0

    import numpy as np

    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))

    prep = Prepared(data)
    message = result.solution.message or ""
    if "stall predictor" in message:
        route = "shortcut"
    elif "after the IPM stalled" in message:
        route = "ipm-rescue"
    elif "IPM converged" in message:
        route = "ipm"
    else:
        route = "other"

    return {
        "instance": name,
        "backend": result.backend,
        "route": route,
        "message": message,
        "pivots": result.solution.iterations,
        "objective": result.solution.objective_value,
        "residual": residual,
        "status": result.solution.status.value,
        "cpu": cpu,
        "wall": wall,
        "raw_shape": [*prep.raw_matrix.shape, prep.raw_matrix.nnz],
        "base_shape": list(prep.shape("base")),
        "gate_accepts": prep.gate_accepts,
        "agg_shape": list(prep.shape("agg")),
        "forced_agg_shape": list(prep.shape("forced_agg")),
        "presolve_cpu": prep.presolve_cpu,
    }


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for name in names:
        print(json.dumps(probe(name)), flush=True)


if __name__ == "__main__":
    main()
