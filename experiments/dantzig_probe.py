"""DS plain-Dantzig leaving-rule probe on selected LP suite instances.

Runs the native CSR dual simplex path directly on the same presolved
equality-plus-bounds problem shape used by SparseSolver, comparing the default
Devex-weighted leaving rule against plain max-violation leaving selection.
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

INSTANCES = ("greenbea", "woodw", "stocfor3", "80bau3b", "cre_d")
CONFIGS = {
    "default-devex": 0,
    "plain-dantzig": 1,
}
DEFAULT_LPSUITE = Path("/tmp/lpsuite")
DEFAULT_OUT = Path("probe_out/dantzig-probe.json")
DEFAULT_MAX_ITER = 50_000
CRE_D_MAX_ITER = 300_000
TOL = 1e-8
RESIDUAL_LIMIT = 2e-5
OBJECTIVE_REL_LIMIT = 2e-5
GREENBEA_MIN_REDUCTION = 0.15
GUARD_MAX_COUNT_RATIO = 1.10
GATED_GUARD_MAX_COUNT_RATIO = 1.13


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


def _max_iter_for(name: str) -> int:
    return CRE_D_MAX_ITER if name == "cre_d" else DEFAULT_MAX_ITER


def _solve_one(
    data: dict[str, Any], prepared: dict[str, Any], leaving_rule: int, max_iter: int
) -> dict[str, Any]:
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
        "objective": objective,
        "max_residual": residual,
    }
    if "bound_flips" in result:
        row["bound_flips"] = int(result.get("bound_flips", 0))
    if "ft_stats" in result:
        row["ft_stats"] = result["ft_stats"]
    return row


def _run_instance(name: str, directory: Path) -> dict[str, Any]:
    data = load_instance(directory / f"lp_{name}.mat")
    prepared = _prepare(data)
    max_iter = _max_iter_for(name)
    runs = {
        label: _solve_one(data, prepared, leaving_rule, max_iter)
        for label, leaving_rule in CONFIGS.items()
    }
    return {
        "instance": name,
        "shape": list(data["A_scipy"].shape),
        "nnz": int(data["A_scipy"].nnz),
        "max_iter": max_iter,
        "presolve": prepared["presolve"],
        "runs": runs,
    }


def _relative_objective_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    base_obj = baseline.get("objective")
    cand_obj = candidate.get("objective")
    if base_obj is None or cand_obj is None:
        return None
    return abs(float(cand_obj) - float(base_obj)) / max(1.0, abs(float(base_obj)))


def _ratio(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return new / old


def _reduction(new: float, old: float) -> float | None:
    ratio = _ratio(new, old)
    if ratio is None:
        return None
    return 1.0 - ratio


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["instance"]: row for row in rows}
    criteria: dict[str, Any] = {}

    green_runs = by_name["greenbea"]["runs"]
    green_base = green_runs["default-devex"]
    green_candidate = green_runs["plain-dantzig"]
    green_iter_reduction = _reduction(green_candidate["iterations"], green_base["iterations"])
    green_wall_reduction = _reduction(green_candidate["wall"], green_base["wall"])
    green_rel_delta = _relative_objective_delta(green_base, green_candidate)
    green_cert_ok = (
        green_candidate.get("status") == "optimal"
        and green_candidate.get("max_residual") is not None
        and float(green_candidate["max_residual"]) <= RESIDUAL_LIMIT
        and green_rel_delta is not None
        and green_rel_delta <= OBJECTIVE_REL_LIMIT
    )
    green_lift_ok = (
        green_iter_reduction is not None and green_iter_reduction >= GREENBEA_MIN_REDUCTION
    ) or (green_wall_reduction is not None and green_wall_reduction >= GREENBEA_MIN_REDUCTION)
    criteria["greenbea_lift"] = {
        "iteration_reduction": green_iter_reduction,
        "wall_reduction": green_wall_reduction,
        "threshold": GREENBEA_MIN_REDUCTION,
        "pass": green_lift_ok,
    }
    criteria["greenbea_certificate"] = {
        "status": green_candidate.get("status"),
        "max_residual": green_candidate.get("max_residual"),
        "objective_relative_delta": green_rel_delta,
        "residual_threshold": RESIDUAL_LIMIT,
        "objective_relative_threshold": OBJECTIVE_REL_LIMIT,
        "pass": green_cert_ok,
    }

    guard_details = {}
    guard_pass = True
    for row in rows:
        name = row["instance"]
        if name == "greenbea":
            continue
        runs = row["runs"]
        ratio = _ratio(
            runs["plain-dantzig"]["iterations"],
            runs["default-devex"]["iterations"],
        )
        passed = ratio is not None and ratio <= GUARD_MAX_COUNT_RATIO
        guard_details[name] = {
            "count_ratio": ratio,
            "count_regression": None if ratio is None else ratio - 1.0,
            "threshold": GUARD_MAX_COUNT_RATIO,
            "pass": passed,
        }
        guard_pass = guard_pass and passed
    criteria["guard_count_regression"] = {
        "instances": guard_details,
        "pass": guard_pass,
    }

    status_details = {}
    residual_details = {}
    objective_details = {}
    status_pass = True
    residual_pass = True
    objective_pass = True
    for row in rows:
        name = row["instance"]
        runs = row["runs"]
        for label, run in runs.items():
            key = f"{name}:{label}"
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

        rel_delta = _relative_objective_delta(runs["default-devex"], runs["plain-dantzig"])
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

    green_pass = green_lift_ok and green_cert_ok
    all_certificates_pass = status_pass and residual_pass and objective_pass
    gated_guard_pass = all(
        detail["count_ratio"] is not None
        and 1.0 < detail["count_ratio"] <= GATED_GUARD_MAX_COUNT_RATIO
        for detail in guard_details.values()
        if not detail["pass"]
    )
    criteria["old_note_gated_band"] = {
        "max_regression_threshold": GATED_GUARD_MAX_COUNT_RATIO,
        "pass": gated_guard_pass,
    }

    if not green_pass or not all_certificates_pass:
        verdict = "KILL"
    elif guard_pass:
        verdict = "PASS"
    elif gated_guard_pass:
        verdict = "GATED-PASS"
    else:
        verdict = "KILL"

    return {"verdict": verdict, "criteria": criteria}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [_run_instance(name, args.directory) for name in INSTANCES]
    payload = {
        "probe": "ds-plain-dantzig-post-ft-post-suhl",
        "settings": {
            "instances": list(INSTANCES),
            "max_iter_default": DEFAULT_MAX_ITER,
            "max_iter_cre_d": CRE_D_MAX_ITER,
            "tol": TOL,
            "expand": 1,
            "configs": CONFIGS,
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
