"""Close-six W2-B: exact-DSE full-cost attribution on the simplex-routed losses.

Two modes.

``round``
    One paired round: for every arm of one instance, run the *production*
    ``SparseSolver.solve`` whole cell (exactly the shape ``suite_bench.py``
    times) once, under that arm's environment, and report CPU seconds.
    Arms are rotated per round so ordering effects cancel.  CPU time
    (``time.process_time``) is the metric because the host is shared -- see
    ``docs/`` measurement doctrine.

``instrument``
    One un-timed run per arm that replicates the production route
    (``presolve_matrix`` -> aggregation gate -> DS/DS2) and returns the raw C
    result dict: pivots, refactorizations, per-phase microseconds, FTRAN/BTRAN
    counters and the pivot-trace digest.

Every experimental knob is env-gated and OFF by default; the production route
at HEAD is byte-identical with no environment set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.suite_bench import bounds_of, load_instance  # noqa: E402

# --- arm table -------------------------------------------------------------
# Each arm is a name -> environment overlay.  {} is the shipped production cell.

DS_ARMS: dict[str, dict[str, str]] = {
    "shipped": {},
    "dse": {"LINPROGX_W2B_DS_RULE": "5"},
    "dse_churn": {"LINPROGX_W2B_DS_RULE": "5", "LINPROGX_DS_CHURN_DSE": "1"},
    # DS_PHASE1 is only consumed under the logical form (_csparse.c:14766), so
    # the two-phase arm needs both and `dse_logical` isolates the form itself.
    "dse_logical": {"LINPROGX_W2B_DS_RULE": "5", "LINPROGX_DS_LOGICAL_FORM": "1"},
    "dse_twophase": {
        "LINPROGX_W2B_DS_RULE": "5",
        "LINPROGX_DS_LOGICAL_FORM": "1",
        "LINPROGX_DS_PHASE1": "1",
    },
    "ds2": {"LINPROGX_W2B_FORCE_DS2": "1"},
    "ds2_pair": {"LINPROGX_W2B_FORCE_DS2": "1", "LINPROGX_DS2_DSE_PAIR": "1"},
    "ds2_agg": {"LINPROGX_W2B_FORCE_AGG": "1"},
    "ds2_agg_pair": {"LINPROGX_W2B_FORCE_AGG": "1", "LINPROGX_DS2_DSE_PAIR": "1"},
}

DS2_ARMS: dict[str, dict[str, str]] = {
    "shipped": {},
    "ds2_pair": {"LINPROGX_DS2_DSE_PAIR": "1"},
    "ds2_churn": {"LINPROGX_DS2_CHURN": "1"},
    "ds2_refac250": {"LINPROGX_DS2_REFAC": "250"},
    "ds2_refac60": {"LINPROGX_DS2_REFAC": "60"},
    "ds2_pair_refac250": {"LINPROGX_DS2_DSE_PAIR": "1", "LINPROGX_DS2_REFAC": "250"},
}

ARMS_BY_INSTANCE: dict[str, dict[str, dict[str, str]]] = {
    "lp_25fv47": DS_ARMS,
    "lp_degen2": DS_ARMS,
    "lp_greenbeb": DS2_ARMS,
    "lp_greenbea": DS2_ARMS,
}

ARM_KEYS = sorted(
    {k for arms in (DS_ARMS, DS2_ARMS) for arm in arms.values() for k in arm}
    | {"LINPROGX_DS2_TRACE_HASH"}
)


def clear_arm_env() -> None:
    for key in ARM_KEYS:
        os.environ.pop(key, None)


def apply_arm(overlay: dict[str, str]) -> None:
    clear_arm_env()
    os.environ.update(overlay)


def whole_cell(data: dict[str, Any], eps: float = 2e-5) -> dict[str, Any]:
    """Run the production whole cell and return CPU seconds plus certificate."""
    import numpy as np

    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    matrix = from_scipy_sparse(data["A_scipy"])
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    result = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=eps, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=matrix,
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )
    )
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))
    return {
        "cpu": cpu,
        "wall": wall,
        "status": result.solution.status.value,
        "objective": result.solution.objective_value,
        "residual": residual,
        "pivots": result.solution.iterations,
        "backend": result.backend.rsplit("-", 2)[-1],
        "message": result.solution.message,
    }


def replicate_route(data: dict[str, Any]) -> dict[str, Any]:
    """Replicate the production route and return the raw C result dict.

    Mirrors ``sparse.py``'s stall-predictor shortcut: presolve, aggregation
    gate, then DS2 or the dual simplex with the arm's leaving rule.
    """
    import numpy as np

    from linprogx.presolve import aggressive_aggregate_for_ds2, postsolve_x, presolve_matrix
    from linprogx.sparse import (
        _ds2_composition_enabled,
        _ipm_stall_risk,
        _w2b_ds_leaving_rule,
        _w2b_force_ds2,
        csr_matrix,
        from_scipy_sparse,
    )

    matrix = from_scipy_sparse(data["A_scipy"])
    b = data["b"].tolist()
    c = data["c"].tolist()
    lo = [float(v) for v in data["lo"]]
    hi = [float(v) for v in data["hi"]]

    reduction = presolve_matrix(matrix, b, c, lo, hi, algorithm="auto")
    if reduction is None:
        raise SystemExit("presolve returned no reduction; route replication invalid")
    work = (
        reduction._matrix
        if reduction._matrix is not None
        else csr_matrix(
            reduction.rows, reduction.cols, reduction.indptr, reduction.indices, reduction.data
        )
    )
    solve_c, solve_b, solve_lo, solve_hi = (
        reduction.c,
        reduction.b,
        reduction.lo,
        reduction.hi,
    )
    ps_rows, ps_cols = work.shape
    stall = (
        ps_rows <= 4000
        and ps_cols <= 30_000
        and _ipm_stall_risk(solve_c, solve_lo, solve_hi, work.nnz, ps_cols)
    )
    if not stall:
        raise SystemExit("stall predictor false: this instance is not on the shortcut route")

    aggregated = False
    if _ds2_composition_enabled():
        aggressive = aggressive_aggregate_for_ds2(reduction)
        if aggressive is not None:
            reduction = aggressive
            work = (
                aggressive._matrix
                if aggressive._matrix is not None
                else csr_matrix(
                    aggressive.rows,
                    aggressive.cols,
                    aggressive.indptr,
                    aggressive.indices,
                    aggressive.data,
                )
            )
            solve_c, solve_b = aggressive.c, aggressive.b
            solve_lo, solve_hi = aggressive.lo, aggressive.hi
            aggregated = True
    ds2_route = aggregated or _w2b_force_ds2()

    shape = list(work.shape) + [work.nnz]
    cpu0 = time.process_time()
    if ds2_route:
        raw = work.solve_eq_box_ds2(solve_c, solve_b, solve_lo, solve_hi, max_iter=50_000)
    else:
        raw = work.solve_eq_box_dual_simplex(
            solve_c,
            solve_b,
            solve_lo,
            solve_hi,
            max_iter=50_000,
            leaving_rule=_w2b_ds_leaving_rule(),
            expand=1,
        )
    solve_cpu = time.process_time() - cpu0

    x = postsolve_x([float(v) for v in raw["x"]], reduction)
    objective = sum(v * coef for v, coef in zip(x, c, strict=True))
    xa = np.array(x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ xa - data["b"])))
    out = {k: v for k, v in raw.items() if k not in ("x", "y")}
    out.update(
        {
            "route": "ds2" if ds2_route else "dual_simplex",
            "aggregated": aggregated,
            "solve_shape": shape,
            "solve_cpu": solve_cpu,
            "objective_original_units": objective,
            "residual_original_units": residual,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("round", "instrument"))
    ap.add_argument("--instance", required=True)
    ap.add_argument("--fixtures", default="/tmp/lpsuite")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    arms = ARMS_BY_INSTANCE[args.instance]
    names = [n.strip() for n in args.arms.split(",") if n.strip()] or list(arms)
    data = load_instance(Path(args.fixtures) / f"{args.instance}.mat")

    if args.mode == "round":
        # Rotate arm order per round so no arm keeps a warm-cache advantage.
        shift = args.round % len(names)
        order = names[shift:] + names[:shift]
        for name in order:
            apply_arm(arms[name])
            row = whole_cell(data)
            row.update({"instance": args.instance, "arm": name, "round": args.round})
            print(json.dumps(row), flush=True)
        clear_arm_env()
        return

    for name in names:
        overlay = dict(arms[name])
        overlay["LINPROGX_DS2_TRACE_HASH"] = "1"
        apply_arm(overlay)
        # Instrumentation only: these timers are wall-based and add overhead,
        # so they never run in the timed rounds.
        os.environ["LINPROGX_DS_PIVOT_TRACE"] = "1"
        os.environ["LINPROGX_DS_SOLVE_SLICE"] = "1"
        os.environ["LINPROGX_DS2_SOLVE_SLICE"] = "1"
        row = replicate_route(data)
        row.update({"instance": args.instance, "arm": name})
        print(json.dumps(row), flush=True)
    clear_arm_env()
    for key in ("LINPROGX_DS_PIVOT_TRACE", "LINPROGX_DS_SOLVE_SLICE", "LINPROGX_DS2_SOLVE_SLICE"):
        os.environ.pop(key, None)


if __name__ == "__main__":
    main()
