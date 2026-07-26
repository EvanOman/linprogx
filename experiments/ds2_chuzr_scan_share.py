"""How much of a pivot does the shipped CHUZR dense scan actually cost?

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

The DS2 CHUZR brief funds a cheaper leaving-row scan on the argument that
HiGHS affords dual steepest edge because its CHUZR is cheap (a maintained
hyper-sparse infeasibility list) where linprogx rescans all m rows every
pivot.  Before building anything, price the thing being replaced.

The shipped dual simplex already reports `phase_us`, an intra-process
per-phase wall profile, with `leaving_scan` as its own bucket.  A ratio of
two buckets inside one process is not a cross-process wall comparison, so
the 4-19% load drift that makes those unusable does not apply here: both
numbers are perturbed by the same load, and the ratio survives.

Usage:
  PYTHONPATH=src uv run python experiments/ds2_chuzr_scan_share.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scipy.io import loadmat

SUITE = Path("/tmp/lpsuite")
INF = float("inf")


def load(name: str):
    raw = loadmat(SUITE / f"{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return (
        raw["A"].tocsr().astype(float),
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def run(name: str, leaving_rule: int) -> dict:
    from linprogx.sparse import from_scipy_sparse

    A, b, c, lo, hi = load(name)
    M = from_scipy_sparse(A)
    t0 = time.perf_counter()
    r = M.solve_eq_box_dual_simplex(
        c.tolist(),
        b.tolist(),
        lo.tolist(),
        hi.tolist(),
        max_iter=200_000,
        leaving_rule=leaving_rule,
        expand=1,
    )
    wall_ms = (time.perf_counter() - t0) * 1e3
    ph = r.get("phase_us", {})
    iters = max(1, int(r["iterations"]))
    total_us = sum(ph.values()) if ph else float("nan")
    return {
        "instance": name,
        "rows": int(A.shape[0]),
        "cols": int(A.shape[1]),
        "rule": leaving_rule,
        "status": r["status"],
        "iters": iters,
        "wall_ms": wall_ms,
        "us_per_pivot": 1e3 * wall_ms / iters,
        "scan_us": ph.get("leaving_scan", float("nan")),
        "scan_us_per_pivot": ph.get("leaving_scan", float("nan")) / iters,
        "scan_share": ph.get("leaving_scan", float("nan")) / max(total_us, 1e-9),
        "phase_us_per_pivot": {k: v / iters for k, v in sorted(ph.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="lp_greenbea,lp_25fv47,lp_degen2")
    ap.add_argument("--rules", default="1,5")
    ap.add_argument("--out", default="/tmp/ds2_chuzr_scan_share.json")
    a = ap.parse_args()

    out = []
    for name in a.instances.split(","):
        for rule in (int(x) for x in a.rules.split(",")):
            try:
                rec = run(name, rule)
            except Exception as exc:  # noqa: BLE001 - probe script
                print(f"{name} rule={rule}: FAILED {type(exc).__name__}: {exc}")
                continue
            out.append(rec)
            print(
                f"{rec['instance']:12s} rule={rule} {rec['status']:9s} "
                f"iters={rec['iters']:6d}  {rec['us_per_pivot']:7.1f} us/pivot  "
                f"scan {rec['scan_us_per_pivot']:7.3f} us/pivot "
                f"({100 * rec['scan_share']:5.2f}% of profiled time)",
                flush=True,
            )
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nartifact: {a.out}")
    if out:
        print("\nper-pivot phase breakdown (us/pivot):")
        keys = sorted(out[0]["phase_us_per_pivot"])
        hdr = "  ".join(f"{k[:12]:>12s}" for k in keys)
        print(f"{'instance/rule':>18s}  {hdr}")
        for rec in out:
            row = "  ".join(f"{rec['phase_us_per_pivot'].get(k, float('nan')):12.2f}" for k in keys)
            print(f"{rec['instance'] + '/' + str(rec['rule']):>18s}  {row}")


if __name__ == "__main__":
    main()
