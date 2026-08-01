"""Close-six W2-B analysis: gate arithmetic, sign tests, causal decomposition.

Consumes the paired CPU-time JSONL from ``w2b_drive.py`` plus the per-arm
instrumentation JSONL from ``w2b_dse_cost.py instrument`` and emits
``dse_cost.json`` -- one record per (instance, arm) carrying pivots, median CPU
time, refactorizations, FTRAN/BTRAN or phase timing, status, objective and
residual, together with the whole-cell candidate/shipped ratio, its exact
one-sided sign-test p-value and the pivot x per-pivot decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# Funding gates (candidate/shipped whole-cell) for a 0.97 final ratio.
GATES = {"lp_25fv47": 0.275, "lp_degen2": 0.268, "lp_greenbeb": 0.705}
# Control: must not regress its certified 0.986 win.
CONTROL = "lp_greenbea"


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def sign_test(wins: int, trials: int) -> float:
    """Exact one-sided binomial p-value for `wins` successes under p=0.5."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return tail / (2**trials)


def decision_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-target verdict: does ANY measured composition meet the gate?"""
    table = []
    control_by_arm = {
        r["arm"]: r for r in records if r["instance"] == CONTROL and r["arm"] != "shipped"
    }
    for inst, gate in GATES.items():
        arms = [r for r in records if r["instance"] == inst and r["arm"] != "shipped"]
        # An arm only counts if it certifies optimal on the shipped simplex
        # backend (an IPM fallback is a different cell, not a DSE composition).
        eligible = [r for r in arms if r["status"] == "optimal" and r["backend"] == "simplex"]
        best = min(eligible, key=lambda r: r["whole_cell_ratio_vs_shipped"]) if eligible else None
        ctrl = control_by_arm.get(best["arm"]) if best else None
        table.append(
            {
                "instance": inst,
                "gate": gate,
                "any_composition_meets_gate": any(
                    r["whole_cell_ratio_vs_shipped"] <= gate and r["sign_test_p_one_sided"] <= 0.05
                    for r in eligible
                ),
                "best_arm": best["arm"] if best else None,
                "best_ratio": best["whole_cell_ratio_vs_shipped"] if best else None,
                "best_sign_test": (
                    f"{best['sign_test_wins']}/{best['sign_test_trials']} "
                    f"p={best['sign_test_p_one_sided']:.4f}"
                    if best
                    else None
                ),
                "pivot_ratio": best["pivot_ratio_vs_shipped"] if best else None,
                "per_pivot_ratio": best["per_pivot_ratio_vs_shipped"] if best else None,
                "shortfall_x": (best["whole_cell_ratio_vs_shipped"] / gate) if best else None,
                "control_greenbea_ratio_same_arm": (
                    ctrl["whole_cell_ratio_vs_shipped"] if ctrl else None
                ),
                "control_regressed": (
                    bool(ctrl["whole_cell_ratio_vs_shipped"] > 1.0) if ctrl else None
                ),
            }
        )
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timed", required=True)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    timed: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for line in Path(args.timed).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        timed[(row["instance"], row["arm"])][row["round"]] = row

    instr: dict[tuple[str, str], dict[str, Any]] = {}
    for line in Path(args.instrument).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instr[(row["instance"], row["arm"])] = row

    instances = sorted({inst for inst, _ in timed})
    records: list[dict[str, Any]] = []
    for inst in instances:
        base_rounds = timed[(inst, "shipped")]
        base_cpu = median([r["cpu"] for r in base_rounds.values()])
        base_pivots = next(iter(base_rounds.values()))["pivots"]
        arms = sorted({a for i, a in timed if i == inst})
        for arm in arms:
            rounds = timed[(inst, arm)]
            cpus = [r["cpu"] for r in rounds.values()]
            med = median(cpus)
            paired = [
                (rounds[k]["cpu"], base_rounds[k]["cpu"])
                for k in sorted(rounds)
                if k in base_rounds
            ]
            wins = sum(1 for a, b in paired if a < b)
            sample = next(iter(rounds.values()))
            pivots = sample["pivots"]
            statuses = sorted({r["status"] for r in rounds.values()})
            objectives = sorted({round(r["objective"], 6) for r in rounds.values()})
            piv_set = sorted({r["pivots"] for r in rounds.values()})
            ins = instr.get((inst, arm), {})
            ratio = med / base_cpu if base_cpu else float("nan")
            piv_ratio = pivots / base_pivots if base_pivots else float("nan")
            rec: dict[str, Any] = {
                "instance": inst,
                "arm": arm,
                "pivots": pivots,
                "pivots_all_rounds": piv_set,
                "deterministic_pivots": len(piv_set) == 1,
                "cpu_time_median_s": med,
                "cpu_times_s": [rounds[k]["cpu"] for k in sorted(rounds)],
                "wall_median_s": median([r["wall"] for r in rounds.values()]),
                "rounds": len(cpus),
                "status": statuses[0] if len(statuses) == 1 else statuses,
                "backend": sample["backend"],
                "objective": sample["objective"],
                "objective_stable": len(objectives) == 1,
                "residual": max(r["residual"] for r in rounds.values()),
                "whole_cell_ratio_vs_shipped": ratio,
                "sign_test_wins": wins,
                "sign_test_trials": len(paired),
                "sign_test_p_one_sided": sign_test(wins, len(paired)),
                "pivot_ratio_vs_shipped": piv_ratio,
                "per_pivot_ratio_vs_shipped": (ratio / piv_ratio if piv_ratio else float("nan")),
                "cpu_us_per_pivot": (med / pivots * 1e6) if pivots else None,
                "gate": GATES.get(inst),
                "meets_gate": (None if inst not in GATES else bool(ratio <= GATES[inst])),
            }
            if inst in GATES and arm != "shipped":
                gate = GATES[inst]
                ppr = rec["per_pivot_ratio_vs_shipped"]
                # What the arm would additionally need to reach its gate,
                # holding the other factor at its measured value.
                rec["gate_arithmetic"] = {
                    "gate": gate,
                    "measured": ratio,
                    "shortfall_x": ratio / gate,
                    "required_pivot_ratio_at_measured_per_pivot": (gate / ppr if ppr else None),
                    "required_pivots_at_measured_per_pivot": (
                        int(round(base_pivots * gate / ppr)) if ppr else None
                    ),
                    "required_per_pivot_ratio_at_measured_pivots": (
                        gate / piv_ratio if piv_ratio else None
                    ),
                }
            if inst == CONTROL:
                rec["control_note"] = "must not regress the certified 0.986 win"
            # Instrumentation (route replication, separate un-timed run).
            if ins:
                rec["instrument"] = {
                    "route": ins.get("route"),
                    "aggregated": ins.get("aggregated"),
                    "solve_shape_rows_cols_nnz": ins.get("solve_shape"),
                    "pivots": ins.get("iterations"),
                    "status": ins.get("status"),
                    "objective_original_units": ins.get("objective_original_units"),
                    "residual_original_units": ins.get("residual_original_units"),
                    "refactorizations": ins.get("refactorizations"),
                    "bound_flips": ins.get("bound_flips"),
                    "degenerate_pivots": ins.get("degenerate_pivots"),
                    "phase1_iterations": ins.get("phase1_iterations"),
                    "solve_cpu_s": ins.get("solve_cpu"),
                    "pivot_trace_hash": ins.get("pivot_trace_hash"),
                    "phase_us": ins.get("phase_us"),
                    "solve_slice_us": ins.get("solve_slice_us"),
                    "ft_stats": ins.get("ft_stats"),
                }
            records.append(rec)

    payload = {
        "campaign": "close-six W2-B: exact-DSE full-cost attribution",
        "date": "2026-07-31",
        "host": "shared, loaded; CPU time (time.process_time) is the metric",
        "protocol": {
            "rounds": max((r["rounds"] for r in records), default=0),
            "pairing": "one subprocess per (round, instance) solving every arm back to back; "
            "arm order rotates with the round index",
            "statistic": "per-arm median CPU seconds + exact one-sided paired sign test "
            "vs the shipped arm",
            "threads": "OMP/OPENBLAS/MKL pinned to 1",
            "eps": 2e-5,
            "cell": "production SparseSolver.solve whole cell, same shape suite_bench.py times",
        },
        "gates": GATES,
        "opportunity_ceiling": json.loads(Path(args.floor).read_text()),
        "decision": decision_table(records),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}: {len(records)} records")


if __name__ == "__main__":
    main()
