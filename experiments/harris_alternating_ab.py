"""Alternating within-process A/B for the Harris cheapest-filter-first early-outs.

METHODOLOGY (required by the campaign's LOADED-BOX doctrine).  This box is
shared and cross-process phase minima drift 4-19% between runs, which is larger
than the effect under test.  So both arms are measured in ONE process,
strictly alternating B,A,B,A,... so that any contention drift is shared by both
arms and cancels in the paired difference.  We report:

  * the paired per-repetition ratio (arm B / arm A) -- robust to drift
  * the median of those ratios, which is the headline number
  * the untouched control phases, which MUST show a ratio near 1.000; if a
    control phase moves as much as ratio_test, the result is contention and
    must be discarded.

Arm A (LINPROGX_DS_HARRIS_FASTPATH=0) reproduces the shipped kernel.
Arm B (=1) enables the two bit-identical early-outs.  Both arms compile the
same branch, so arm A is if anything slightly penalised -- conservative.

BIT-IDENTITY is asserted on every repetition: identical pivot count and
identical objective repr across all arms and reps.  Any mismatch aborts.

Usage:
    PYTHONPATH=. uv run python experiments/harris_alternating_ab.py --pairs 11
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
ARM_ENV = "LINPROGX_DS_HARRIS_FASTPATH"
CONTROL_PHASES_ALL = ("btran_rho", "ftran_col", "pivot_row", "refactor",
                      "lu_update", "ratio_test", "rcost_update")


def load_instance(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A_scipy": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def solve_once(data: dict[str, Any], arm: str) -> dict[str, Any]:
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    for _var in ARM_ENV.split(","):
        os.environ[_var.strip()] = "1" if arm == "B" else "0"

    original = from_scipy_sparse(data["A_scipy"])
    reduction = presolve_matrix(
        original, data["b"].tolist(), data["c"].tolist(),
        data["lo"].tolist(), data["hi"].tolist(), algorithm="auto",
    )
    matrix = reduction._matrix
    if matrix is None:
        matrix = csr_matrix(
            reduction.rows, reduction.cols, reduction.indptr,
            reduction.indices, reduction.data,
        )
    out = matrix.solve_eq_box_dual_simplex(
        reduction.c, reduction.b, reduction.lo, reduction.hi,
        max_iter=50_000, leaving_rule=1, expand=1,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--instance", default="lp_greenbea")
    parser.add_argument("--out", default="/tmp/harris_alternating_ab.json")
    parser.add_argument("--env", default="LINPROGX_DS_HARRIS_FASTPATH",
                        help="env var(s), comma-separated, toggled between arms")
    parser.add_argument("--treatment", default="ratio_test",
                        help="phase expected to change")
    args = parser.parse_args()

    global ARM_ENV
    ARM_ENV = args.env

    data = load_instance(SUITE / f"{args.instance}.mat")

    # Warm-up (page-ins, first-touch) -- discarded.
    solve_once(data, "B")
    solve_once(data, "A")

    rows: list[dict[str, Any]] = []
    signatures: set[tuple[int, str]] = set()
    for _ in range(args.pairs):
        for arm in ("B", "A"):
            out = solve_once(data, arm)
            phases = out.get("phase_us") or {}
            signatures.add((int(out["iterations"]), repr(float(out["objective"]))))
            rows.append({"arm": arm, "phases": phases,
                         "total": sum(phases.values()),
                         "iterations": int(out["iterations"])})

    if len(signatures) != 1:
        raise SystemExit(f"BIT-IDENTITY VIOLATED across arms/reps: {sorted(signatures)}")
    iters, obj_repr = next(iter(signatures))
    print(f"bit-identity     OK  ({iters} pivots, objective {obj_repr})")
    print(f"pairs            {args.pairs}")

    b_rows = [r for r in rows if r["arm"] == "B"]
    a_rows = [r for r in rows if r["arm"] == "A"]

    def paired_ratios(key: str) -> list[float]:
        out = []
        for rb, ra in zip(b_rows, a_rows, strict=True):
            av = ra["phases"].get(key, 0.0) if key != "TOTAL" else ra["total"]
            bv = rb["phases"].get(key, 0.0) if key != "TOTAL" else rb["total"]
            if av > 0:
                out.append(bv / av)
        return out

    print(f"\n{'phase':18s} {'B/A median':>11s} {'B/A min':>9s} {'B/A max':>9s}  note")
    result: dict[str, Any] = {"pairs": args.pairs, "iterations": iters,
                              "objective_repr": obj_repr, "ratios": {}}
    CONTROL_PHASES = tuple(p for p in CONTROL_PHASES_ALL if p != args.treatment)
    ordered = [args.treatment, *CONTROL_PHASES, "TOTAL"]
    for key in ordered:
        ratios = paired_ratios(key)
        if not ratios:
            continue
        med = statistics.median(ratios)
        note = "<-- TREATMENT" if key == args.treatment else (
            "control" if key in CONTROL_PHASES else "")
        print(f"{key:18s} {med:11.4f} {min(ratios):9.4f} {max(ratios):9.4f}  {note}")
        result["ratios"][key] = {"median": med, "min": min(ratios),
                                 "max": max(ratios), "all": ratios}

    treat = statistics.median(paired_ratios(args.treatment))
    controls = [statistics.median(paired_ratios(p)) for p in CONTROL_PHASES]
    worst_control_drift = max(abs(1.0 - c) for c in controls)
    print(f"\ntreatment effect on {args.treatment} : {100.0 * (treat - 1.0):+.2f}%")
    print(f"worst control-phase drift      : {100.0 * worst_control_drift:.2f}%")

    a_share = statistics.median(
        [ra["phases"][args.treatment] / ra["total"] for ra in a_rows])
    saved_share = a_share * (1.0 - treat)
    print(f"{args.treatment} share of wall (arm A): {100.0 * a_share:.2f}%")
    print(f"=> whole-wall saving           : {100.0 * saved_share:.2f}%")
    result["treatment_ratio"] = treat
    result["worst_control_drift"] = worst_control_drift
    result["whole_wall_saving"] = saved_share

    if worst_control_drift >= abs(1.0 - treat):
        print("\nVERDICT: INCONCLUSIVE — control drift is as large as the effect.")
        result["verdict"] = "INCONCLUSIVE"
    elif treat < 1.0:
        print("\nVERDICT: REAL — treatment beats controls; bit-identical.")
        result["verdict"] = "REAL"
    else:
        print("\nVERDICT: NO EFFECT or REGRESSION.")
        result["verdict"] = "NO_EFFECT"

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"artifact: {args.out}")


if __name__ == "__main__":
    main()
