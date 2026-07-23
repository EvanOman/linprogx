"""Exact dual steepest-edge versus Dantzig on the DS anatomy fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.suite_bench import load_instance
from linprogx.presolve import postsolve_x, presolve_matrix
from linprogx.sparse import csr_matrix, from_scipy_sparse

INSTANCES = ("greenbea", "woodw", "stocfor3", "cre_d", "80bau3b")
CONFIGS = {"dantzig": 1, "exact-dse": 5}
DEFAULT_LPSUITE = Path("/tmp/lpsuite")
DEFAULT_OUT = Path("probe_out/dse-probe.json")
TOL = 1e-8
OBJECTIVE_REL_LIMIT = 2e-5
GREENBEA_LIVE_PIVOTS = 3_500


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _prepare(data: dict[str, Any]) -> dict[str, Any]:
    matrix = from_scipy_sparse(data["A_scipy"])
    b = _float_list(data["b"])
    c = _float_list(data["c"])
    lo = _float_list(data["lo"])
    hi = _float_list(data["hi"])
    reduction = presolve_matrix(matrix, b, c, lo, hi)
    if reduction is None:
        return {
            "matrix": matrix,
            "b": b,
            "c": c,
            "solve_b": b,
            "solve_c": c,
            "solve_lo": lo,
            "solve_hi": hi,
            "reduction": None,
        }
    reduced_matrix = reduction._matrix
    if reduced_matrix is None:
        reduced_matrix = csr_matrix(
            reduction.rows,
            reduction.cols,
            reduction.indptr,
            reduction.indices,
            reduction.data,
        )
    return {
        "matrix": reduced_matrix,
        "b": b,
        "c": c,
        "solve_b": reduction.b,
        "solve_c": reduction.c,
        "solve_lo": reduction.lo,
        "solve_hi": reduction.hi,
        "reduction": reduction,
    }


def _solve(data: dict[str, Any], prepared: dict[str, Any], leaving_rule: int) -> dict[str, Any]:
    max_iter = 300_000 if data["name"] == "cre_d" else 50_000
    start = time.perf_counter()
    result = prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["solve_c"],
        prepared["solve_b"],
        prepared["solve_lo"],
        prepared["solve_hi"],
        max_iter=max_iter,
        tol=TOL,
        expand=1,
        leaving_rule=leaving_rule,
    )
    wall = time.perf_counter() - start
    x = _float_list(result.get("x", []))
    if prepared["reduction"] is not None and x:
        x = postsolve_x(x, prepared["reduction"])
    objective = None
    residual = None
    if x:
        x_array = np.asarray(x)
        objective = float(np.dot(np.asarray(prepared["c"]), x_array))
        residual = float(np.max(np.abs(data["A_scipy"] @ x_array - data["b"])))
    return {
        "status": result.get("status"),
        "pivots": int(result.get("iterations", 0)),
        "wall_seconds": wall,
        "objective": objective,
        "max_primal_residual": residual,
    }


def _run(name: str, directory: Path) -> dict[str, Any]:
    data = load_instance(directory / f"lp_{name}.mat")
    data["name"] = name
    prepared = _prepare(data)
    runs = {label: _solve(data, prepared, leaving_rule) for label, leaving_rule in CONFIGS.items()}
    dantzig_obj = runs["dantzig"]["objective"]
    dse_obj = runs["exact-dse"]["objective"]
    objective_delta = None
    objective_relative_delta = None
    if dantzig_obj is not None and dse_obj is not None:
        objective_delta = dse_obj - dantzig_obj
        objective_relative_delta = abs(objective_delta) / max(1.0, abs(dantzig_obj))
    return {
        "instance": name,
        "runs": runs,
        "objective_delta": objective_delta,
        "objective_relative_delta": objective_relative_delta,
        "objective_equal": (
            objective_relative_delta is not None and objective_relative_delta <= OBJECTIVE_REL_LIMIT
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for name in INSTANCES:
        row = _run(name, args.directory)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    greenbea_dse = rows[0]["runs"]["exact-dse"]["pivots"]
    verdict = "LIVE" if greenbea_dse < GREENBEA_LIVE_PIVOTS else "KILLED"
    payload = {
        "probe": "exact-forrest-goldfarb-dse",
        "settings": {
            "configs": CONFIGS,
            "expand": 1,
            "tol": TOL,
            "objective_relative_limit": OBJECTIVE_REL_LIMIT,
            "greenbea_live_pivots": GREENBEA_LIVE_PIVOTS,
        },
        "verdict": verdict,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "out": str(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
