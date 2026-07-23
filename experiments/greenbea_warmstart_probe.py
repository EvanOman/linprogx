"""G2 falsifier: crossover partial greenbea IPM iterates into warm dual simplex.

All crossover behavior is diagnostic-only.  The IPM extraction path requires
``LINPROGX_IPM_CROSSOVER_SLICE=1`` and the basis injection requires
``LINPROGX_DS_WARM_START=1``; neither changes a default solver route.
"""

from __future__ import annotations

import heapq
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from experiments.greenbea_pivot_gap_probe import prepare
from experiments.suite_bench import load_instance
from linprogx.presolve import postsolve_x
from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

EPS = 2e-5
DS_TOL = 1e-8
K_VALUES = (20, 30, 35, 40, 45, 48, 49, 50, 55, 59, 60)
FIXTURE_DIR = Path("/tmp/lpsuite")
RESULTS = Path("/tmp/greenbea-warmstart/results.json")


def _distance_to_nearest_bound(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lower_distance = np.where(np.isfinite(lo), np.abs(x - lo), np.inf)
    upper_distance = np.where(np.isfinite(hi), np.abs(hi - x), np.inf)
    distance = np.minimum(lower_distance, upper_distance)
    # Free columns have no nearest bound and are natural basic candidates.
    distance[~np.isfinite(distance)] = np.finfo(np.float64).max
    return distance


def _superbasic_basis(prepared: dict[str, Any], x: np.ndarray) -> tuple[list[int], dict[str, int]]:
    """Attempt 1: the m structural columns farthest from a bound."""
    m, n = prepared["A_scipy"].shape
    distance = _distance_to_nearest_bound(x, prepared["lo"], prepared["hi"])
    order = np.lexsort((np.arange(n), -distance))
    basis = order[:m].astype(int).tolist()
    return basis, {"structural_columns": m, "artificial_columns": 0}


def _bixby_iterate_crash(
    prepared: dict[str, Any], x: np.ndarray
) -> tuple[list[int], dict[str, int]]:
    """Attempt 2: iterate-prioritized triangular crash with artificial fill.

    A structural column enters only when it has one uncovered row.  Covering
    that row therefore produces a triangular structural block; uncovered rows
    receive their identity artificials.  Among available singleton columns,
    the iterate's distance to its nearest bound supplies the priority.
    """
    csc = prepared["A_scipy"].tocsc()
    csr = prepared["A_scipy"].tocsr()
    m, n = csc.shape
    distance = _distance_to_nearest_bound(x, prepared["lo"], prepared["hi"])
    uncovered_count = np.diff(csc.indptr).astype(np.int32)
    covered = np.zeros(m, dtype=np.bool_)
    done = np.zeros(n, dtype=np.bool_)
    col_max = np.zeros(n, dtype=np.float64)
    for j in range(n):
        start, end = int(csc.indptr[j]), int(csc.indptr[j + 1])
        if start < end:
            col_max[j] = float(np.max(np.abs(csc.data[start:end])))

    queue: list[tuple[float, int]] = []
    for j in np.flatnonzero(uncovered_count == 1):
        heapq.heappush(queue, (-float(distance[j]), int(j)))

    basis: list[int] = []
    while queue:
        _, j = heapq.heappop(queue)
        if done[j] or uncovered_count[j] != 1:
            continue
        done[j] = True
        start, end = int(csc.indptr[j]), int(csc.indptr[j + 1])
        rows = csc.indices[start:end]
        values = np.abs(csc.data[start:end])
        live = ~covered[rows]
        if int(np.count_nonzero(live)) != 1:
            continue
        row = int(rows[live][0])
        pivot = float(values[live][0])
        # Match the native crash's single global stability guard.
        if pivot < 0.5 * col_max[j]:
            continue
        basis.append(j)
        covered[row] = True
        row_start, row_end = int(csr.indptr[row]), int(csr.indptr[row + 1])
        for q_raw in csr.indices[row_start:row_end]:
            q = int(q_raw)
            if not done[q] and uncovered_count[q] > 0:
                uncovered_count[q] -= 1
                if uncovered_count[q] == 1:
                    heapq.heappush(queue, (-float(distance[q]), q))

    uncovered_rows = np.flatnonzero(~covered)
    basis.extend(n + int(row) for row in uncovered_rows)
    if len(basis) != m or len(set(basis)) != m:
        raise RuntimeError("iterate crash failed to produce a complete unique basis")
    structural = m - len(uncovered_rows)
    return basis, {
        "structural_columns": structural,
        "artificial_columns": len(uncovered_rows),
    }


def _bound_status(prepared: dict[str, Any], x: np.ndarray, basis: list[int]) -> list[int]:
    """Map nonbasics to their nearest finite bound using DS hook codes."""
    m, n = prepared["A_scipy"].shape
    basic = set(basis)
    status: list[int] = []
    for j, (lo, hi) in enumerate(zip(prepared["lo"], prepared["hi"], strict=True)):
        if j in basic:
            status.append(4)  # basic
        elif np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) < 1e-14:
            status.append(3)  # fixed
        elif not np.isfinite(lo) and not np.isfinite(hi):
            status.append(2)  # free
        elif not np.isfinite(hi):
            status.append(0)  # lower
        elif not np.isfinite(lo):
            status.append(1)  # upper
        else:
            status.append(0 if x[j] - lo <= hi - x[j] else 1)
    status.extend(4 if n + i in basic else 3 for i in range(m))
    return status


def _original_metrics(
    original: dict[str, Any], prepared: dict[str, Any], reduced_x: list[float]
) -> dict[str, float]:
    x = np.asarray(postsolve_x(reduced_x, prepared["reduction"]), dtype=np.float64)
    equality = float(np.max(np.abs(original["A_scipy"] @ x - original["b"])))
    bound = float(
        max(
            np.max(np.maximum(original["lo"] - x, 0.0)),
            np.max(np.maximum(x - original["hi"], 0.0)),
        )
    )
    return {
        "objective_original": float(np.dot(original["c"], x)),
        "max_equality_residual_original": equality,
        "max_bound_violation_original": bound,
    }


def _run_ds(
    original: dict[str, Any],
    prepared: dict[str, Any],
    basis: list[int] | None,
    bound_status: list[int] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if basis is not None:
        kwargs["initial_basis"] = basis
        kwargs["initial_bound_status"] = bound_status
    started = time.perf_counter()
    result = prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["c"].tolist(),
        prepared["b"].tolist(),
        prepared["lo"].tolist(),
        prepared["hi"].tolist(),
        max_iter=50_000,
        tol=DS_TOL,
        expand=1,
        leaving_rule=1,
        bfrt=0,
        **kwargs,
    )
    wall = time.perf_counter() - started
    iterations = int(result["iterations"])
    row = {
        "status": result["status"],
        "ds_pivots": iterations,
        "ds_wall_seconds": wall,
        "us_per_pivot": wall * 1e6 / iterations if iterations else 0.0,
        "warm_start": result.get("warm_start"),
        "ftran_mean_density": result.get("ftran_mean_density"),
        "btran_mean_density": result.get("btran_mean_density"),
        "phase_us": result.get("phase_us"),
        **_original_metrics(original, prepared, result["x"]),
    }
    return row


def _run_ipm(prepared: dict[str, Any], k: int) -> tuple[dict[str, Any], np.ndarray, float]:
    started = time.perf_counter()
    result = prepared["matrix"].solve_eq_box_ipm(
        prepared["c"].tolist(),
        prepared["b"].tolist(),
        prepared["lo"].tolist(),
        prepared["hi"].tolist(),
        max_iter=k,
        tol=1e-9,
        feas_tol=EPS,
        threads=0,
    )
    wall = time.perf_counter() - started
    return result, np.asarray(result["x"], dtype=np.float64), wall


def _fixture_rule_checks() -> list[dict[str, Any]]:
    """Check that normal IPM fixtures certify before extraction is needed."""
    rows: list[dict[str, Any]] = []
    slice_gate = os.environ.pop("LINPROGX_IPM_CROSSOVER_SLICE", None)
    try:
        for name in ("woodw", "80bau3b", "cre_a"):
            data = load_instance(FIXTURE_DIR / f"lp_{name}.mat")
            problem = SparseLPProblem(
                c=data["c"].tolist(),
                A_eq=from_scipy_sparse(data["A_scipy"]),
                b_eq=data["b"].tolist(),
                objective="min",
                bounds=[
                    (
                        None if not np.isfinite(lo) else float(lo),
                        None if not np.isfinite(hi) else float(hi),
                    )
                    for lo, hi in zip(data["lo"], data["hi"], strict=True)
                ],
                name=name,
            )
            result = SparseSolver(algorithm="ipm", eps=EPS, max_iterations=200, threads=0).solve(
                problem
            )
            rows.append(
                {
                    "fixture": name,
                    "status": result.solution.status.value,
                    "iterations": result.solution.iterations,
                    "wall_seconds": result.seconds,
                    "extraction_triggered": result.solution.status.value != "optimal",
                }
            )
    finally:
        if slice_gate is not None:
            os.environ["LINPROGX_IPM_CROSSOVER_SLICE"] = slice_gate
    return rows


def main() -> int:
    required = (
        "LINPROGX_DS_WARM_START",
        "LINPROGX_DS_EXPORT_BASIS",
        "LINPROGX_IPM_CROSSOVER_SLICE",
        "LINPROGX_IPM_SLICE",
    )
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError(f"set diagnostic environment gates to 1: {missing}")

    original, prepared = prepare()
    # One unmeasured C-path warmup, then the board-equivalent cold reference.
    prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["c"].tolist(),
        prepared["b"].tolist(),
        prepared["lo"].tolist(),
        prepared["hi"].tolist(),
        max_iter=1,
        tol=DS_TOL,
        expand=1,
        leaving_rule=1,
        bfrt=0,
    )
    cold = _run_ds(original, prepared, None, None)
    oracle = cold["objective_original"]

    methods: dict[str, Callable[[dict[str, Any], np.ndarray], tuple[list[int], dict[str, int]]]] = {
        "superbasic_top_m": _superbasic_basis,
        "bixby_iterate_crash": _bixby_iterate_crash,
    }
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        ipm, x, ipm_wall = _run_ipm(prepared, k)
        for method_name, method in methods.items():
            started = time.perf_counter()
            basis, basis_summary = method(prepared, x)
            status = _bound_status(prepared, x, basis)
            crossover_wall = time.perf_counter() - started
            ds = _run_ds(original, prepared, basis, status)
            objective_rel = abs(ds["objective_original"] - oracle) / max(1.0, abs(oracle))
            certificate_ok = (
                ds["status"] == "optimal"
                and ds["max_equality_residual_original"] <= EPS
                and ds["max_bound_violation_original"] <= EPS
                and objective_rel <= EPS
            )
            rows.append(
                {
                    "method": method_name,
                    "k": k,
                    "ipm_returned_iterations": int(ipm["iterations"]),
                    "ipm_status": ipm["status"],
                    "ipm_mu": float(ipm["mu"]),
                    "ipm_wall_seconds": ipm_wall,
                    "ipm_slice_us": ipm.get("ipm_slice_us"),
                    "crossover_wall_seconds": crossover_wall,
                    "total_wall_seconds": ipm_wall + crossover_wall + ds["ds_wall_seconds"],
                    "objective_relative_error": objective_rel,
                    "certificate_ok": certificate_ok,
                    **basis_summary,
                    **ds,
                }
            )

    # The global extraction rule is the native safety termination (nonfinite
    # merit, 60-iteration pace watchdog, or 200-iteration cap), followed by
    # extraction of the already-maintained best merit snapshot.  greenbea's
    # safety-stop datum is the k=60 row; ordinary fixtures below should certify
    # and therefore never invoke crossover.
    rule_checks = _fixture_rule_checks()
    accepted = [row for row in rows if row["certificate_ok"]]
    best = min(accepted, key=lambda row: row["total_wall_seconds"], default=None)
    verdict = "LIVE" if best and best["total_wall_seconds"] < 0.30 else "KILLED"
    payload = {
        "fixture": str(FIXTURE_DIR / "lp_greenbea.mat"),
        "eps": EPS,
        "k_values": list(K_VALUES),
        "extraction_rule": (
            "On a non-optimal IPM safety termination (first nonfinite merit, "
            "the native iteration-60 pace watchdog, or the global 200-iteration "
            "cap), extract the native best max(pres,dres,mu) snapshot; converged "
            "IPM solves never crossover."
        ),
        "cold_baseline": cold,
        "oracle_objective_original": oracle,
        "rows": rows,
        "fixture_rule_checks": rule_checks,
        "best_certificate_backed_row": best,
        "verdict": verdict,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
