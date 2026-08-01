"""W2-A: machine-readable arm matrix over the 11 simplex-routed LPnetlib cases.

Measures every existing simplex mechanism and their funded cross-products at
HEAD fc2f86e.  Production defaults are never modified: experimental arms are
driven by replicating the route in ``linprogx.sparse._solve_eq_box`` and calling
the kernels directly, with env flags scoped to a single call.

Measurement protocol (shared, heavily-loaded host -- loadavg ~35 on 12 cores):
  * BLAS/OMP pinned to one thread so ``time.process_time`` is a clean total-CPU
    measure rather than a sum over spin-waiting worker threads.
  * Arms are INTERLEAVED inside each round and the arm order is rotated per
    round, so every arm sees the same load conditions and a paired sign test
    against the shipped arm is valid.
  * >= 9 rounds; the verdict statistic is the median of per-round CPU time.
    Wall time is recorded as context only.
  * Certificate gate: status optimal AND original-units max equality residual
    <= 2e-5 after postsolve.  An arm that cannot certify is recorded
    KILLED-uncertified and its timings are not used for any verdict.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
        uv run python experiments/close6_w2a_arm_matrix.py --rounds 11 \
        --out /tmp/linprogx-close6/wave2/w2a/arm_matrix.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from close6_w2a_lib import (  # noqa: E402
    CASES,
    EPS,
    FIXTURE_DIR,
    EnvScope,
    Prepared,
    certify,
    load_instance,
    problem_of,
)

MAX_ITER = 50_000

# --------------------------------------------------------------------------
# Arm definitions.
#
# kind      : "route" (full SparseSolver) | "ds" (legacy dual simplex kernel)
#             | "ds2" (the DS2 rewrite kernel)
# bundle    : which reduction the kernel is driven on --
#             "base" = presolve-only (what the shipped Dantzig DS sees),
#             "agg"  = the shipped 20%/5% aggregation (DS2 composition input),
#             "fagg" = the SAME aggregation with the global gate bypassed
#                      (not shippable; records what the gate is declining).
# kw        : kernel keyword arguments
# env       : environment scoped to the single kernel call
# --------------------------------------------------------------------------

DS_BASE_KW = {"max_iter": MAX_ITER, "expand": 1}

ARMS: list[dict[str, Any]] = [
    # ---- reference: the shipped production route, end to end -------------
    {"arm": "prod-auto", "kind": "route", "env": {}},
    # ---- legacy dual simplex on the presolve-only reduction --------------
    {
        "arm": "ds-dantzig-churn",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 1},
        "env": {},
        "shipped_kernel_for": ("shortcut-noagg", "ipm-rescue"),
    },
    {
        "arm": "ds-dantzig-nochurn",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 1},
        "env": {"LINPROGX_DS_CHURN_DANTZIG": "0"},
    },
    {
        "arm": "ds-dse",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {},
    },
    {
        "arm": "ds-dse-churn",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {"LINPROGX_DS_CHURN_DSE": "1"},
    },
    {
        "arm": "ds-dse-bfrt",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 5, "bfrt": 1},
        "env": {},
    },
    # ---- two-phase variants and the DSE x two-phase cross-product --------
    {
        "arm": "ds-dantzig-phase1",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 1},
        "env": {"LINPROGX_DS_PHASE1": "1"},
    },
    {
        "arm": "ds-dse-phase1",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {"LINPROGX_DS_PHASE1": "1"},
    },
    {
        "arm": "ds-dse-churn-phase1",
        "kind": "ds",
        "bundle": "base",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {"LINPROGX_DS_CHURN_DSE": "1", "LINPROGX_DS_PHASE1": "1"},
    },
    # ---- DS2 driven directly on the un-aggregated reduction --------------
    {"arm": "ds2-direct", "kind": "ds2", "bundle": "base", "kw": {"max_iter": MAX_ITER}, "env": {}},
    # ---- the shipped DS2 composition (gate-accepting cells only) ---------
    {
        "arm": "agg-ds2",
        "kind": "ds2",
        "bundle": "agg",
        "kw": {"max_iter": MAX_ITER},
        "env": {},
        "shipped_kernel_for": ("shortcut-agg",),
    },
    # ---- DSE x aggregation cross-products (gate-accepting cells) ---------
    {
        "arm": "agg-ds-dantzig-churn",
        "kind": "ds",
        "bundle": "agg",
        "kw": {**DS_BASE_KW, "leaving_rule": 1},
        "env": {},
    },
    {
        "arm": "agg-ds-dse",
        "kind": "ds",
        "bundle": "agg",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {},
    },
    {
        "arm": "agg-ds-dse-churn",
        "kind": "ds",
        "bundle": "agg",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {"LINPROGX_DS_CHURN_DSE": "1"},
    },
    {
        "arm": "agg-ds-dse-phase1",
        "kind": "ds",
        "bundle": "agg",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {"LINPROGX_DS_PHASE1": "1"},
    },
    # ---- aggregation-gate-off: the whole route with the composition off --
    {
        "arm": "prod-auto-aggoff",
        "kind": "route",
        "env": {"LINPROGX_DS2_COMPOSITION": "0"},
    },
    # ---- gate-bypassed aggregation (cells the 20%/5% gate declines) ------
    {
        "arm": "fagg-ds-dantzig-churn",
        "kind": "ds",
        "bundle": "fagg",
        "kw": {**DS_BASE_KW, "leaving_rule": 1},
        "env": {},
    },
    {
        "arm": "fagg-ds-dse",
        "kind": "ds",
        "bundle": "fagg",
        "kw": {**DS_BASE_KW, "leaving_rule": 5},
        "env": {},
    },
    {"arm": "fagg-ds2", "kind": "ds2", "bundle": "fagg", "kw": {"max_iter": MAX_ITER}, "env": {}},
]


def applicable(arm: dict[str, Any], prep: Prepared) -> bool:
    kind = arm["kind"]
    if kind == "route":
        # the aggregation-gate-off route arm is only meaningful where the gate
        # actually accepts; elsewhere it is identical to prod-auto.
        return arm["arm"] != "prod-auto-aggoff" or prep.gate_accepts
    bundle = arm["bundle"]
    if bundle == "base":
        return True
    if bundle == "agg":
        return prep.agg is not None
    if bundle == "fagg":
        # only informative where the shipped gate DECLINES an available
        # aggregation; where it accepts, "fagg" is the same object as "agg".
        return prep.forced_agg is not None and not prep.gate_accepts
    return False


def run_once(arm: dict[str, Any], prep: Prepared, data: dict[str, Any]) -> dict[str, Any]:
    """Execute one arm once.  Returns timings + certificate outcome."""
    kind = arm["kind"]
    if kind == "route":
        from linprogx.sparse import SparseSolver

        problem = problem_of(data)
        with EnvScope(arm["env"]):
            c0, w0 = time.process_time(), time.perf_counter()
            result = SparseSolver(
                algorithm="auto", max_iterations=MAX_ITER, eps=EPS, check_interval=MAX_ITER
            ).solve(problem)
            cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        import numpy as np

        x = np.array(result.solution.x, dtype=float)
        residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))
        status = result.solution.status.value
        message = result.solution.message or ""
        return {
            "cpu": cpu,
            "wall": wall,
            "pivots": result.solution.iterations,
            "status": status,
            "objective": result.solution.objective_value,
            "residual": residual,
            "certified": status == "optimal" and residual <= EPS,
            "message": message,
        }

    bundle = {"base": prep.base, "agg": prep.agg, "fagg": prep.forced_agg}[arm["bundle"]]
    matrix, reduction, sc, sb, slo, shi = bundle
    call = matrix.solve_eq_box_ds2 if kind == "ds2" else matrix.solve_eq_box_dual_simplex
    try:
        with EnvScope(arm["env"]):
            c0, w0 = time.process_time(), time.perf_counter()
            raw = call(sc, sb, slo, shi, **arm["kw"])
            cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    except Exception as exc:  # noqa: BLE001 - a crashing arm is a killed arm
        return {
            "cpu": None,
            "wall": None,
            "pivots": None,
            "status": f"exception: {type(exc).__name__}: {exc}",
            "objective": None,
            "residual": None,
            "certified": False,
        }
    cert = certify(prep, reduction, raw)
    return {
        "cpu": cpu,
        "wall": wall,
        "pivots": int(raw["iterations"]),
        **cert,
    }


def measure_instance(name: str, rounds: int) -> list[dict[str, Any]]:
    data = load_instance(FIXTURE_DIR / f"{name}.mat")
    prep = Prepared(data)
    arms = [a for a in ARMS if applicable(a, prep)]

    samples: dict[str, list[dict[str, Any]]] = {a["arm"]: [] for a in arms}
    for r in range(rounds):
        # rotate the arm order every round so no arm is systematically
        # advantaged or disadvantaged by its position under varying load
        order = arms[r % len(arms) :] + arms[: r % len(arms)]
        for arm in order:
            samples[arm["arm"]].append(run_once(arm, prep, data))
        print(f"  {name}: round {r + 1}/{rounds} done", file=sys.stderr, flush=True)

    records = []
    for arm in arms:
        runs = samples[arm["arm"]]
        certified = [s for s in runs if s["certified"]]
        cpus = [s["cpu"] for s in certified]
        walls = [s["wall"] for s in certified]
        pivots = sorted({s["pivots"] for s in runs if s["pivots"] is not None})
        bundle_name = arm.get("bundle", "route")
        shape = {
            "route": [*prep.raw_matrix.shape, prep.raw_matrix.nnz],
            "base": list(prep.shape("base")),
            "agg": list(prep.shape("agg")),
            "fagg": list(prep.shape("forced_agg")),
        }[bundle_name]
        rec: dict[str, Any] = {
            "instance": name,
            "arm": arm["arm"],
            "kind": arm["kind"],
            "bundle": bundle_name,
            "env": arm["env"],
            "kernel_kwargs": {k: v for k, v in arm.get("kw", {}).items() if k != "max_iter"},
            "scope": "full-route" if arm["kind"] == "route" else "kernel-only",
            "rounds": len(runs),
            "certified_rounds": len(certified),
            "status": runs[0]["status"],
            "objective": runs[0]["objective"],
            "residual": runs[0]["residual"],
            "pivots": pivots[0] if len(pivots) == 1 else None,
            "pivots_observed": pivots,
            "deterministic_pivots": len(pivots) == 1,
            "reduced_rows": shape[0],
            "reduced_cols": shape[1],
            "reduced_nnz": shape[2],
            "presolve_cpu": prep.presolve_cpu,
            "gate_accepts_aggregation": prep.gate_accepts,
        }
        if certified:
            rec.update(
                {
                    "cpu_time_median": statistics.median(cpus),
                    "cpu_time_all": cpus,
                    "cpu_time_min": min(cpus),
                    "wall_median": statistics.median(walls),
                    "wall_all": walls,
                    "verdict": "measured",
                }
            )
        else:
            rec.update(
                {
                    "cpu_time_median": None,
                    "cpu_time_all": [],
                    "cpu_time_min": None,
                    "wall_median": None,
                    "wall_all": [],
                    "verdict": "KILLED-uncertified",
                    "kill_reason": runs[0]["status"]
                    if runs[0]["status"] != "optimal"
                    else f"residual {runs[0]['residual']:.3e} > eps {EPS:.0e}",
                }
            )
        if arm["kind"] == "route":
            msg = runs[0].get("message", "")
            rec["route_realised"] = (
                "shortcut"
                if "stall predictor" in msg
                else "ipm-rescue"
                if "after the IPM stalled" in msg
                else "ipm"
                if "IPM converged" in msg
                else "other"
            )
            rec["message"] = msg
        records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instances", nargs="*", default=list(CASES))
    args = ap.parse_args()

    all_records: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.instances:
        print(f"[{name}] starting", file=sys.stderr, flush=True)
        recs = measure_instance(name, args.rounds)
        all_records.extend(recs)
        # write incrementally so a long run is never lost
        args.out.write_text(json.dumps({"records": all_records}, indent=1))
        print(f"[{name}] {len(recs)} arms recorded", file=sys.stderr, flush=True)

    payload = {
        "campaign": "close-six wave 2A -- simplex arm matrix",
        "ref": "fc2f86e9ba71f81cb4d496fbaeb0179dcac4f699",
        "eps": EPS,
        "rounds": args.rounds,
        "protocol": {
            "statistic": "median of per-round time.process_time over certified rounds",
            "threads": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
            "interleaved": True,
            "arm_order_rotated_per_round": True,
            "wall_time": "context only -- shared host, loadavg ~35 on 12 cores",
            "certificate": "status optimal AND original-units max equality residual <= 2e-5",
        },
        "records": all_records,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {len(all_records)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
