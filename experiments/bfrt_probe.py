"""DS-BFRT re-probe on selected LP suite instances.

Runs the native CSR dual simplex path directly on the same presolved
equality-plus-bounds problem shape used by SparseSolver, comparing Harris
baseline (bfrt=0) against BFRT (bfrt=1).
"""

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

INSTANCES = ("greenbea", "woodw", "80bau3b")
DEFAULT_LPSUITE = Path("/tmp/lpsuite")
DEFAULT_OUT = Path("probe_out/bfrt-probe.json")
MAX_ITER = 50_000
TOL = 1e-8
RESIDUAL_LIMIT = 2e-5
OBJECTIVE_REL_LIMIT = 2e-5


def _as_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _prepare(data: dict[str, Any]) -> dict[str, Any]:
    matrix = from_scipy_sparse(data["A_scipy"])
    b = _as_float_list(data["b"])
    c = _as_float_list(data["c"])
    lo = _as_float_list(data["lo"])
    hi = _as_float_list(data["hi"])

    reduction = presolve_matrix(matrix, b, c, lo, hi)
    if reduction is not None:
        if reduction._matrix is not None:
            matrix = reduction._matrix
        else:
            matrix = csr_matrix(
                reduction.rows,
                reduction.cols,
                reduction.indptr,
                reduction.indices,
                reduction.data,
            )
        solve_b = reduction.b
        solve_c = reduction.c
        solve_lo = reduction.lo
        solve_hi = reduction.hi
    else:
        solve_b = b
        solve_c = c
        solve_lo = lo
        solve_hi = hi

    return {
        "matrix": matrix,
        "b": b,
        "c": c,
        "solve_b": solve_b,
        "solve_c": solve_c,
        "solve_lo": solve_lo,
        "solve_hi": solve_hi,
        "reduction": reduction,
        "presolve": {
            "removed_rows": 0 if reduction is None else reduction.removed_rows,
            "removed_cols": 0 if reduction is None else reduction.removed_cols,
        },
    }


def _solve_one(data: dict[str, Any], prepared: dict[str, Any], bfrt: int) -> dict[str, Any]:
    start = time.perf_counter()
    result = prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["solve_c"],
        prepared["solve_b"],
        prepared["solve_lo"],
        prepared["solve_hi"],
        max_iter=MAX_ITER,
        tol=TOL,
        expand=1,
        bfrt=bfrt,
    )
    wall = time.perf_counter() - start

    x = [float(value) for value in result.get("x", [])]
    if prepared["reduction"] is not None and x:
        x = postsolve_x(x, prepared["reduction"])

    if x:
        x_array = np.array(x, dtype=float)
        objective = float(np.dot(np.array(prepared["c"], dtype=float), x_array))
        residual = float(np.max(np.abs(data["A_scipy"] @ x_array - data["b"])))
    else:
        objective = None
        residual = None

    row = {
        "status": result.get("status"),
        "iterations": int(result.get("iterations", 0)),
        "wall": wall,
        "bound_flips": int(result.get("bound_flips", 0)),
        "objective": objective,
        "max_residual": residual,
    }
    if "ft_stats" in result:
        row["ft_stats"] = result["ft_stats"]
    return row


def _run_instance(name: str, directory: Path) -> dict[str, Any]:
    data = load_instance(directory / f"lp_{name}.mat")
    prepared = _prepare(data)
    runs = {str(bfrt): _solve_one(data, prepared, bfrt) for bfrt in (0, 1)}
    return {
        "instance": name,
        "shape": list(data["A_scipy"].shape),
        "nnz": int(data["A_scipy"].nnz),
        "presolve": prepared["presolve"],
        "runs": runs,
    }


def _relative_objective_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    base_obj = baseline.get("objective")
    cand_obj = candidate.get("objective")
    if base_obj is None or cand_obj is None:
        return None
    return abs(float(cand_obj) - float(base_obj)) / max(1.0, abs(float(base_obj)))


def _ratio(new: int, old: int) -> float | None:
    if old == 0:
        return None
    return new / old


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["instance"]: row for row in rows}
    criteria: dict[str, Any] = {}

    green = by_name["greenbea"]["runs"]
    green_ratio = _ratio(green["1"]["iterations"], green["0"]["iterations"])
    green_reduction = None if green_ratio is None else 1.0 - green_ratio
    criteria["greenbea_iteration_reduction"] = {
        "value": green_reduction,
        "threshold": 0.15,
        "pass": green_reduction is not None and green_reduction >= 0.15,
    }

    guard_details = {}
    guard_pass = True
    for name in ("woodw", "80bau3b"):
        runs = by_name[name]["runs"]
        ratio = _ratio(runs["1"]["iterations"], runs["0"]["iterations"])
        passed = ratio is not None and ratio <= 1.10
        guard_details[name] = {"ratio": ratio, "threshold": 1.10, "pass": passed}
        guard_pass = guard_pass and passed
    criteria["guard_pivot_regression"] = {"instances": guard_details, "pass": guard_pass}

    status_details = {}
    residual_details = {}
    objective_details = {}
    status_pass = True
    residual_pass = True
    objective_pass = True
    for row in rows:
        name = row["instance"]
        runs = row["runs"]
        for bfrt, run in runs.items():
            key = f"{name}:bfrt={bfrt}"
            status_ok = run.get("status") == "optimal"
            residual = run.get("max_residual")
            residual_ok = residual is not None and float(residual) <= RESIDUAL_LIMIT
            status_details[key] = {"status": run.get("status"), "pass": status_ok}
            residual_details[key] = {
                "value": residual,
                "threshold": RESIDUAL_LIMIT,
                "pass": residual_ok,
            }
            status_pass = status_pass and status_ok
            residual_pass = residual_pass and residual_ok

        rel_delta = _relative_objective_delta(runs["0"], runs["1"])
        obj_ok = rel_delta is not None and rel_delta <= OBJECTIVE_REL_LIMIT
        objective_details[name] = {
            "relative_delta": rel_delta,
            "threshold": OBJECTIVE_REL_LIMIT,
            "pass": obj_ok,
        }
        objective_pass = objective_pass and obj_ok

    criteria["all_statuses_optimal"] = {"runs": status_details, "pass": status_pass}
    criteria["all_residuals"] = {"runs": residual_details, "pass": residual_pass}
    criteria["objectives_match_baseline"] = {
        "instances": objective_details,
        "pass": objective_pass,
    }

    passed = all(item["pass"] for item in criteria.values())
    return {"verdict": "PASS" if passed else "KILL", "criteria": criteria}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [_run_instance(name, args.directory) for name in INSTANCES]
    payload = {
        "probe": "ds-bfrt-post-ft-post-suhl",
        "settings": {
            "instances": list(INSTANCES),
            "max_iter": MAX_ITER,
            "tol": TOL,
            "expand": 1,
            "bfrt_values": [0, 1],
        },
        "results": rows,
        "evaluation": evaluate(rows),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["evaluation"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
