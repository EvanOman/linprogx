"""Roll the per-cell arm matrix up into GLOBAL policy verdicts.

An arm is a measurement; a *policy* is what shipping that arm would mean.  The
distinction matters because the route decides which cells a policy can touch:
changing ``leaving_rule`` on the legacy dual simplex cannot affect lp_greenbea,
lp_greenbeb or lp_tuff at all, because the aggregation gate sends those three
to the DS2 composition instead.

Each policy therefore declares the cells it actually reaches, and is judged on
every one of them -- wins on the losses, and regressions on the controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# cells whose production route runs the LEGACY dual simplex
LEGACY_DS_CELLS = [
    "lp_25fv47",
    "lp_degen2",
    "lp_sierra",
    "lp_agg2",
    "lp_agg3",
    "lp_cycle",
    "lp_fffff800",
    "lp_israel",
]
# cells whose production route runs the DS2 composition (aggregation accepted)
DS2_CELLS = ["lp_greenbeb", "lp_greenbea", "lp_tuff"]
# cells where the aggregation gate DECLINES an available reduction
GATE_DECLINED_CELLS = ["lp_25fv47", "lp_cycle", "lp_fffff800"]

LOSSES = {"lp_25fv47": 0.275, "lp_degen2": 0.268, "lp_greenbeb": 0.705, "lp_sierra": 0.354}
BOARD = {"lp_25fv47": 3.530948, "lp_degen2": 3.619711, "lp_greenbeb": 1.375663, "lp_sierra": 2.737341}

POLICIES: list[dict[str, Any]] = [
    {
        "policy": "dse-legacy",
        "description": "leaving_rule 1 -> 5 (exact DSE) on the legacy dual simplex route",
        "arm": "ds-dse",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "dse-churn-legacy",
        "description": "exact DSE + the churn penalty applied to the DSE score",
        "arm": "ds-dse-churn",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "dse-phase1-legacy",
        "description": "exact DSE x in-place two-phase bound handling (cross-product)",
        "arm": "ds-dse-phase1",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "dse-bfrt-legacy",
        "description": "exact DSE x bound-flipping ratio test (cross-product)",
        "arm": "ds-dse-bfrt",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "churn-off",
        "description": "disable the shipped Dantzig churn penalty",
        "arm": "ds-dantzig-nochurn",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "phase1-dantzig",
        "description": "shipped Dantzig + in-place two-phase bound handling",
        "arm": "ds-dantzig-phase1",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "ds2-for-gate-declined",
        "description": "route gate-declined stall-shortcut cells to DS2 instead of the legacy DS",
        "arm": "ds2-direct",
        "cells": LEGACY_DS_CELLS,
    },
    {
        "policy": "aggregation-off",
        "description": "LINPROGX_DS2_COMPOSITION=0 -- retire the aggregation + DS2 composition",
        "arm": "prod-auto-aggoff",
        "cells": DS2_CELLS,
    },
    {
        "policy": "dse-on-aggregated",
        "description": "exact DSE x controlled aggregation (cross-product), replacing DS2",
        "arm": "agg-ds-dse",
        "cells": DS2_CELLS,
    },
    {
        "policy": "dse-churn-on-aggregated",
        "description": "exact DSE + churn x controlled aggregation (cross-product)",
        "arm": "agg-ds-dse-churn",
        "cells": DS2_CELLS,
    },
    {
        "policy": "dantzig-on-aggregated",
        "description": "shipped Dantzig+churn x controlled aggregation, replacing DS2",
        "arm": "agg-ds-dantzig-churn",
        "cells": DS2_CELLS,
    },
    {
        "policy": "fillneg-gate-then-ds2",
        "description": (
            "widen the aggregation gate from '>=20% rows AND <=+5% nnz' to "
            "'rows reduce AND nnz does not grow', then run DS2 on the result"
        ),
        "arm": "fagg-ds2",
        "cells": GATE_DECLINED_CELLS,
    },
    {
        "policy": "fillneg-gate-then-dse",
        "description": "the same widened gate, then exact DSE on the legacy dual simplex",
        "arm": "fagg-ds-dse",
        "cells": GATE_DECLINED_CELLS,
    },
    {
        "policy": "fillneg-gate-then-dantzig",
        "description": "the same widened gate, keeping the shipped Dantzig+churn kernel",
        "arm": "fagg-ds-dantzig-churn",
        "cells": GATE_DECLINED_CELLS,
    },
]

#: Paired sign-test thresholds for n = 11 rounds (binomial, p = 0.5).
#:
#: The test is deliberately ASYMMETRIC: strict about claiming an improvement,
#: lenient about detecting damage to a control.
#:   * a claimed WIN needs >= 10/11, two-sided p = 0.0117
#:   * a control REGRESSION is called at <= 2/11, one-sided p = 0.0327
#:   * 3-4/11 with a median > 5% worse is a SUSPECTED regression that n = 11
#:     cannot resolve; it is reported, never silently passed.
SIGNIFICANT_WINS = 10
SIGNIFICANT_LOSSES = 2
SUSPECT_LOSSES = 4
SUSPECT_MEDIAN_MARGIN = 1.05


def build(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for pol in POLICIES:
        cells = {}
        verdict = "FUNDED"
        kills: list[str] = []
        suspects: list[str] = []
        for inst in pol["cells"]:
            blk = analysis.get(inst)
            if blk is None:
                continue
            row = next((r for r in blk["arms"] if r["arm"] == pol["arm"]), None)
            if row is None:
                continue
            if row["verdict"] != "measured":
                cells[inst] = {
                    "verdict": "KILLED-uncertified",
                    "kill_reason": row.get("kill_reason"),
                    "pivots": row.get("pivots"),
                }
                kills.append(f"{inst}: cannot certify ({row.get('kill_reason')})")
                continue
            e2e = row.get("endtoend_ratio_vs_prod")
            wins = row.get("sign_test_wins")
            n = row.get("sign_test_n")
            entry = {
                "endtoend_ratio_vs_prod": e2e,
                "pivots": row["pivots"],
                "pivot_ratio_vs_shipped": row.get("pivot_ratio_vs_shipped"),
                "sign_test": f"{wins}/{n}",
                "is_loss_cell": inst in LOSSES,
            }
            if inst in LOSSES:
                entry["gate"] = LOSSES[inst]
                entry["closes_gate"] = bool(e2e is not None and e2e <= LOSSES[inst])
                entry["projected_board_ratio"] = (
                    BOARD[inst] * e2e if e2e is not None else None
                )
            else:
                regressed = (
                    e2e is not None
                    and e2e > 1.0
                    and wins is not None
                    and wins <= SIGNIFICANT_LOSSES
                )
                suspect = (
                    not regressed
                    and e2e is not None
                    and e2e > SUSPECT_MEDIAN_MARGIN
                    and wins is not None
                    and wins <= SUSPECT_LOSSES
                )
                entry["control_regressed"] = regressed
                entry["control_suspect"] = suspect
                if regressed:
                    kills.append(f"{inst}: control regressed to {e2e:.3f} ({wins}/{n})")
                elif suspect:
                    suspects.append(
                        f"{inst}: control suspected regressed to {e2e:.3f} ({wins}/{n}) "
                        "-- not resolvable at n=11"
                    )
            cells[inst] = entry
        if kills:
            verdict = "KILLED"
        elif suspects:
            verdict = "FUNDED-WITH-SUSPECT-CONTROL"
        else:
            gained = [
                c
                for i, c in cells.items()
                if i in LOSSES
                and c.get("endtoend_ratio_vs_prod") is not None
                and c["endtoend_ratio_vs_prod"] < 1.0
                and int(c["sign_test"].split("/")[0]) >= SIGNIFICANT_WINS
            ]
            if not gained:
                verdict = "NO-EFFECT"
        out.append(
            {
                **{k: v for k, v in pol.items() if k != "cells"},
                "cells_reached": pol["cells"],
                "verdict": verdict,
                "kills": kills,
                "suspected_control_regressions": suspects,
                "closes_any_gate": any(c.get("closes_gate") for c in cells.values()),
                "per_cell": cells,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    analysis = json.loads(args.analysis.read_text())
    policies = build(analysis)
    args.out.write_text(json.dumps({"policies": policies}, indent=1))
    for p in policies:
        print(f"\n{p['policy']:28s} {p['verdict']:20s} closes_gate={p['closes_any_gate']}")
        for inst, c in p["per_cell"].items():
            if c["verdict"] == "KILLED-uncertified" if "verdict" in c else False:
                print(f"   {inst:14s} KILLED-uncertified: {c['kill_reason']}")
                continue
            e = c.get("endtoend_ratio_vs_prod")
            tag = ""
            if c.get("is_loss_cell"):
                tag = f"  gate {c['gate']}  -> board {c['projected_board_ratio']:.3f}"
            elif c.get("control_regressed"):
                tag = "  <-- CONTROL REGRESSED"
            elif c.get("control_suspect"):
                tag = "  <-- control suspect"
            print(f"   {inst:14s} e2e={e:.3f} sign={c['sign_test']}{tag}")
        for k in p["kills"]:
            print(f"   KILL: {k}")
        for k in p.get("suspected_control_regressions", []):
            print(f"   SUSPECT: {k}")


if __name__ == "__main__":
    main()
