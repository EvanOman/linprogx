"""Funding arithmetic and paired sign tests over the W2-A arm matrix.

For every (instance, arm) the candidate's *end-to-end* cost is reconstructed
per round as

    est_r = prod_auto_cpu_r - shipped_kernel_cpu_r + candidate_kernel_cpu_r

i.e. the shipped route with only its simplex kernel swapped.  This is the
honest denominator for the funding gates, because presolve, postsolve, the
residual check and -- on lp_sierra -- the failed IPM prefix and its two floored
retries are all charged to the candidate as well.  The sign test counts rounds
in which the candidate kernel beat the shipped kernel on the SAME round, which
is exactly the paired comparison the interleaved schedule was built for.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

GATES = {"lp_25fv47": 0.275, "lp_degen2": 0.268, "lp_greenbeb": 0.705, "lp_sierra": 0.354}


def shipped_kernel_arm(recs: list[dict[str, Any]]) -> str:
    """Which kernel arm reproduces this instance's shipped production solve."""
    prod = next(r for r in recs if r["arm"] == "prod-auto")
    if prod.get("route_realised") == "shortcut" and prod["gate_accepts_aggregation"]:
        return "agg-ds2"
    return "ds-dantzig-churn"


def sign_test(cand: list[float], base: list[float]) -> tuple[int, int, int]:
    """Return (wins, losses, ties) of candidate against baseline, paired."""
    wins = losses = ties = 0
    for a, b in zip(cand, base, strict=False):
        if a < b:
            wins += 1
        elif a > b:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_inst: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_inst.setdefault(r["instance"], []).append(r)

    out: dict[str, Any] = {}
    for inst, recs in by_inst.items():
        prod = next(r for r in recs if r["arm"] == "prod-auto")
        ship_name = shipped_kernel_arm(recs)
        ship = next((r for r in recs if r["arm"] == ship_name), None)
        prod_cpu = prod["cpu_time_all"]
        ship_cpu = ship["cpu_time_all"] if ship else []

        # sanity: the shipped kernel arm must reproduce production's pivot count
        pivot_match = bool(ship) and ship["pivots"] == prod["pivots"]

        rows = []
        for r in recs:
            row = {
                "arm": r["arm"],
                "bundle": r["bundle"],
                "scope": r["scope"],
                "pivots": r["pivots"],
                "verdict": r["verdict"],
                "cpu_median": r["cpu_time_median"],
                "objective": r["objective"],
                "residual": r["residual"],
            }
            if r["verdict"] != "measured":
                row["kill_reason"] = r.get("kill_reason")
                rows.append(row)
                continue
            if r["scope"] == "kernel-only" and ship and ship_cpu and prod_cpu:
                n = min(len(prod_cpu), len(ship_cpu), len(r["cpu_time_all"]))
                est = [
                    prod_cpu[i] - ship_cpu[i] + r["cpu_time_all"][i] for i in range(n)
                ]
                w, lo, t = sign_test(r["cpu_time_all"][:n], ship_cpu[:n])
                row.update(
                    {
                        "kernel_ratio_vs_shipped": (
                            r["cpu_time_median"] / ship["cpu_time_median"]
                        ),
                        "endtoend_ratio_vs_prod": statistics.median(est)
                        / statistics.median(prod_cpu[:n]),
                        "pivot_ratio_vs_shipped": (
                            r["pivots"] / ship["pivots"]
                            if r["pivots"] and ship["pivots"]
                            else None
                        ),
                        "sign_test_wins": w,
                        "sign_test_losses": lo,
                        "sign_test_ties": t,
                        "sign_test_n": n,
                    }
                )
            elif r["scope"] == "full-route" and prod_cpu and r["arm"] != "prod-auto":
                n = min(len(prod_cpu), len(r["cpu_time_all"]))
                w, lo, t = sign_test(r["cpu_time_all"][:n], prod_cpu[:n])
                row.update(
                    {
                        "endtoend_ratio_vs_prod": r["cpu_time_median"]
                        / prod["cpu_time_median"],
                        "sign_test_wins": w,
                        "sign_test_losses": lo,
                        "sign_test_ties": t,
                        "sign_test_n": n,
                    }
                )
            rows.append(row)

        rows.sort(key=lambda x: x.get("endtoend_ratio_vs_prod") or 9e9)
        out[inst] = {
            "shipped_kernel_arm": ship_name,
            "shipped_kernel_reproduces_production_pivots": pivot_match,
            "production_pivots": prod["pivots"],
            "production_cpu_median": prod["cpu_time_median"],
            "production_route": prod.get("route_realised"),
            "gate": GATES.get(inst),
            "arms": rows,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads(args.matrix.read_text())
    analysis = analyze(payload["records"])
    args.out.write_text(json.dumps(analysis, indent=1))

    for inst, blk in analysis.items():
        gate = blk["gate"]
        tag = f"  GATE {gate}" if gate else ""
        print(
            f"\n=== {inst}  route={blk['production_route']} "
            f"piv={blk['production_pivots']} "
            f"prod_cpu={blk['production_cpu_median']:.5f}"
            f"  shipped_kernel={blk['shipped_kernel_arm']}"
            f" (pivot match: {blk['shipped_kernel_reproduces_production_pivots']}){tag}"
        )
        for row in blk["arms"]:
            if row["verdict"] != "measured":
                print(f"  {row['arm']:24s} KILLED-uncertified: {row.get('kill_reason')}")
                continue
            e2e = row.get("endtoend_ratio_vs_prod")
            e2e_s = f"{e2e:6.3f}" if e2e is not None else "   ref"
            st = (
                f" sign {row['sign_test_wins']}/{row['sign_test_n']}"
                if "sign_test_wins" in row
                else ""
            )
            pr = row.get("pivot_ratio_vs_shipped")
            pr_s = f" piv_ratio {pr:5.3f}" if pr else ""
            print(
                f"  {row['arm']:24s} piv={str(row['pivots']):>6s} "
                f"cpu={row['cpu_median']:.5f} e2e={e2e_s}{pr_s}{st}"
            )


if __name__ == "__main__":
    main()
