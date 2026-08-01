"""Supplementary W2-A probes, merged into arm_matrix.json.

Three things the main matrix run did not capture:

1. **Phase timing.** ``solve_eq_box_dual_simplex`` reports ``phase_us`` and
   ``solve_eq_box_ds2`` reports ``phase1_iterations``; both report
   refactorisation and degeneracy counters.  Recorded per (instance, arm).

2. **lp_sierra stage decomposition.** sierra reaches the dual simplex through
   the post-IPM rescue (``sparse.py:404``), so its production cost contains a
   failed IPM plus two floored retries.  This times each stage separately to
   attribute the wall honestly.

3. **lp_cycle's DSE certificate failure.** Every exact-DSE arm returns
   ``dual_infeasible`` on lp_cycle.  In production that does not corrupt the
   answer -- the shortcut declines and the route falls through to the IPM --
   but the wasted simplex plus the IPM is what shipping DSE would actually
   cost on that control.  Measured here.
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

from close6_w2a_arm_matrix import ARMS, MAX_ITER, applicable  # noqa: E402
from close6_w2a_lib import EPS, FIXTURE_DIR, EnvScope, Prepared, load_instance  # noqa: E402


def phase_probe(name: str) -> dict[str, dict[str, Any]]:
    """One run per applicable kernel arm, keeping the kernel's own counters."""
    data = load_instance(FIXTURE_DIR / f"{name}.mat")
    prep = Prepared(data)
    out: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if arm["kind"] == "route" or not applicable(arm, prep):
            continue
        bundle = {"base": prep.base, "agg": prep.agg, "fagg": prep.forced_agg}[arm["bundle"]]
        matrix, _red, sc, sb, slo, shi = bundle
        call = matrix.solve_eq_box_ds2 if arm["kind"] == "ds2" else matrix.solve_eq_box_dual_simplex
        try:
            with EnvScope(arm["env"]):
                raw = call(sc, sb, slo, shi, **arm["kw"])
        except Exception as exc:  # noqa: BLE001
            out[arm["arm"]] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        keep = (
            "phase_us",
            "phase1_iterations",
            "phase1_dual_objective",
            "refactorizations",
            "refac_time",
            "refac_factorize_time",
            "degenerate_pivots",
            "bound_flips",
            "cost_shifts",
            "max_degenerate_streak",
            "ftran_mean_density",
            "btran_mean_density",
            "audit_rounds",
        )
        rec: dict[str, Any] = {"status": str(raw["status"]), "iterations": int(raw["iterations"])}
        for key in keep:
            if key in raw:
                value = raw[key]
                rec[key] = dict(value) if isinstance(value, dict) else value
        out[arm["arm"]] = rec
    return out


def sierra_stages(rounds: int) -> dict[str, Any]:
    """Time each stage of lp_sierra's post-IPM-failure rescue route."""
    data = load_instance(FIXTURE_DIR / "lp_sierra.mat")
    prep = Prepared(data)
    matrix, _red, sc, sb, slo, shi = prep.base
    ipm_kw = {
        "max_iter": 200,
        "tol": min(EPS, 1e-9),
        "threads": 0,
        "feas_tol": EPS,
    }
    stages: dict[str, list[float]] = {
        "presolve": [],
        "ipm_primary": [],
        "ipm_floored_retry_presolved": [],
        "ipm_floored_retry_unpresolved": [],
        "dual_simplex_rescue": [],
    }
    statuses: dict[str, str] = {}
    for _ in range(rounds):
        t = time.process_time()
        prep2 = Prepared(data)
        stages["presolve"].append(prep2.presolve_cpu)

        t = time.process_time()
        r1 = matrix.solve_eq_box_ipm(sc, sb, slo, shi, **ipm_kw)
        stages["ipm_primary"].append(time.process_time() - t)
        statuses["ipm_primary"] = str(r1["status"])

        t = time.process_time()
        r2 = matrix.solve_eq_box_ipm(sc, sb, slo, shi, blas=False, **ipm_kw)
        stages["ipm_floored_retry_presolved"].append(time.process_time() - t)
        statuses["ipm_floored_retry_presolved"] = str(r2["status"])

        t = time.process_time()
        r3 = prep.raw_matrix.solve_eq_box_ipm(
            prep.c, prep.b, prep.lo, prep.hi, blas=False, **ipm_kw
        )
        stages["ipm_floored_retry_unpresolved"].append(time.process_time() - t)
        statuses["ipm_floored_retry_unpresolved"] = str(r3["status"])

        t = time.process_time()
        r4 = matrix.solve_eq_box_dual_simplex(
            sc, sb, slo, shi, max_iter=MAX_ITER, expand=1, leaving_rule=1
        )
        stages["dual_simplex_rescue"].append(time.process_time() - t)
        statuses["dual_simplex_rescue"] = f"{r4['status']} ({int(r4['iterations'])} pivots)"

    medians = {k: statistics.median(v) for k, v in stages.items()}
    total = sum(medians.values())
    return {
        "stage_cpu_median": medians,
        "stage_cpu_all": stages,
        "stage_status": statuses,
        "stage_share_of_sum": {k: v / total for k, v in medians.items()},
        "sum_of_stage_medians": total,
        "note": (
            "The dual simplex is the LAST stage of lp_sierra's route; everything "
            "before it is work the route discards."
        ),
    }


def cycle_dse_fallback(rounds: int) -> dict[str, Any]:
    """What shipping exact DSE would cost on the lp_cycle control.

    Exact DSE returns dual_infeasible on lp_cycle, so the stall-predictor
    shortcut would decline and the route would fall through to the IPM.  The
    realised cost is therefore the wasted DSE simplex PLUS the IPM.
    """
    data = load_instance(FIXTURE_DIR / "lp_cycle.mat")
    prep = Prepared(data)
    matrix, _red, sc, sb, slo, shi = prep.base
    dse: list[float] = []
    ipm: list[float] = []
    shipped: list[float] = []
    statuses: dict[str, Any] = {}
    for _ in range(rounds):
        t = time.process_time()
        r = matrix.solve_eq_box_dual_simplex(
            sc, sb, slo, shi, max_iter=MAX_ITER, expand=1, leaving_rule=5
        )
        dse.append(time.process_time() - t)
        statuses["dse"] = f"{r['status']} ({int(r['iterations'])} pivots)"

        t = time.process_time()
        ri = matrix.solve_eq_box_ipm(
            sc, sb, slo, shi, max_iter=200, tol=min(EPS, 1e-9), threads=0, feas_tol=EPS
        )
        ipm.append(time.process_time() - t)
        statuses["ipm"] = f"{ri['status']} ({int(ri['iterations'])} iters)"

        t = time.process_time()
        rs = matrix.solve_eq_box_dual_simplex(
            sc, sb, slo, shi, max_iter=MAX_ITER, expand=1, leaving_rule=1
        )
        shipped.append(time.process_time() - t)
        statuses["shipped_dantzig_churn"] = f"{rs['status']} ({int(rs['iterations'])} pivots)"

    med_dse, med_ipm, med_ship = (statistics.median(x) for x in (dse, ipm, shipped))
    return {
        "statuses": statuses,
        "wasted_dse_cpu_median": med_dse,
        "ipm_fallback_cpu_median": med_ipm,
        "shipped_kernel_cpu_median": med_ship,
        "dse_route_cost_cpu_median": med_dse + med_ipm,
        "dse_route_ratio_vs_shipped_kernel": (med_dse + med_ipm) / med_ship,
        "sign_test_dse_route_beats_shipped": sum(
            1 for a, b, c in zip(dse, ipm, shipped, strict=True) if a + b < c
        ),
        "rounds": rounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()

    payload = json.loads(args.matrix.read_text())
    instances = sorted({r["instance"] for r in payload["records"]})

    phases: dict[str, dict[str, Any]] = {}
    for name in instances:
        print(f"[phase] {name}", file=sys.stderr, flush=True)
        phases[name] = phase_probe(name)

    for rec in payload["records"]:
        got = phases.get(rec["instance"], {}).get(rec["arm"])
        if got:
            rec["kernel_counters"] = got

    print("[stages] lp_sierra", file=sys.stderr, flush=True)
    payload["sierra_stage_decomposition"] = sierra_stages(args.rounds)
    print("[fallback] lp_cycle", file=sys.stderr, flush=True)
    payload["cycle_dse_fallback"] = cycle_dse_fallback(args.rounds)

    args.matrix.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["sierra_stage_decomposition"]["stage_share_of_sum"], indent=1))
    print(json.dumps(payload["cycle_dse_fallback"], indent=1))


if __name__ == "__main__":
    main()
