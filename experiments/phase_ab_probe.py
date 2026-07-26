"""Load-robust per-phase A/B probe for the greenbea dual simplex.

The box this campaign runs on is shared (load ~11/12 cores), and the measured
whole-wall noise floor is +-3%.  MEDIANS are not robust under that load, but
MINIMA are: contention can only ADD time to a phase, never remove it, so the
minimum over N repetitions is a consistent lower-bound estimator of the
uncontended cost.  This probe reports per-phase minima so an A/B on a single
phase can be read even on a loaded box.

This is still NOT a ship gate.  The campaign's ship standard is alternating A/B
median-of-9 on a QUIET box plus a v3 paired cert.  Use this to size a mechanism
and to decide whether a quiet-box run is warranted.

Usage:
    PYTHONPATH=. uv run python experiments/phase_ab_probe.py --repeats 15 --label baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--instance", default="lp_greenbea")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    data = load_instance(SUITE / f"{args.instance}.mat")
    c, b = data["c"].tolist(), data["b"].tolist()
    lo, hi = data["lo"].tolist(), data["hi"].tolist()

    per_phase: dict[str, list[float]] = {}
    totals: list[float] = []
    iters = 0
    objective_reprs: set[str] = set()

    for _ in range(args.repeats):
        original = from_scipy_sparse(data["A_scipy"])
        reduction = presolve_matrix(original, b, c, lo, hi, algorithm="auto")
    if reduction is None:
        raise SystemExit("presolve returned no reduction")
        matrix = reduction._matrix
        if matrix is None:
            matrix = csr_matrix(
                reduction.rows,
                reduction.cols,
                reduction.indptr,
                reduction.indices,
                reduction.data,
            )
        out = matrix.solve_eq_box_dual_simplex(
            reduction.c,
            reduction.b,
            reduction.lo,
            reduction.hi,
            max_iter=50_000,
            leaving_rule=1,
            expand=1,
        )
        iters = int(out["iterations"])
        phases = out.get("phase_us") or {}
        for name, us in phases.items():
            per_phase.setdefault(name, []).append(us)
        totals.append(sum(phases.values()))
        objective_reprs.add(repr(float(out["objective"])) if "objective" in out else "n/a")

    print(f"label            {args.label}")
    print(f"repeats          {args.repeats}")
    print(f"iterations       {iters}")
    print(f"objective reprs  {sorted(objective_reprs)}")
    print(f"\n{'phase':18s} {'min us':>10s} {'median us':>11s} {'min/pivot':>10s}")
    result: dict[str, Any] = {"label": args.label, "iterations": iters, "phases": {}}
    for name, samples in sorted(per_phase.items(), key=lambda kv: -min(kv[1])):
        lo_us, med_us = min(samples), statistics.median(samples)
        print(f"{name:18s} {lo_us / 1e3:10.3f} {med_us / 1e3:11.3f} {lo_us / max(1, iters):10.3f}")
        result["phases"][name] = {"min_us": lo_us, "median_us": med_us, "all_us": samples}
    total_min = min(totals)
    print(f"\n{'TOTAL (min)':18s} {total_min / 1e3:10.3f} ms")
    print(f"{'TOTAL (median)':18s} {statistics.median(totals) / 1e3:10.3f} ms")
    result["total_min_us"] = total_min
    result["total_median_us"] = statistics.median(totals)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"artifact: {args.out}")


if __name__ == "__main__":
    main()
