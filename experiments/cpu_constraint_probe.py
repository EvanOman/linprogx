"""Is greenbea's BOARD loss a CPU-CONSTRAINT effect rather than a trajectory one?

The ledger records a paradox: on this 12-core box linprogx SOLVES GREENBEA
FASTER than HiGHS (377 ms vs 421 ms), yet the board -- Modal AWS us-west-2,
4-vCPU containers -- records a 1.156x LOSS, with the ratio varying 1.16-1.47
across three hosts of the SAME class and HiGHS's own wall varying 54%.

That is the signature of a resource-contention effect, not an algorithmic one.

Candidate mechanism: linprogx sizes its thread pool from
`sysconf(_SC_NPROCESSORS_ONLN)` (_csparse.c:7274, :1694), which reports the
HOST's online CPUs -- NOT the cgroup CPU quota a container is throttled to. On a
4-vCPU Modal container running on a large host, that would oversubscribe badly.
HiGHS would not, being single-threaded here.

This probe emulates the constraint with taskset and measures the RATIO under
each core count. The ratio is self-normalising against box load to first order,
which matters because this box runs several agents.

Usage:
    PYTHONPATH=. uv run python experiments/cpu_constraint_probe.py --reps 7
    taskset -c 0-3 env PYTHONPATH=. uv run python experiments/cpu_constraint_probe.py --reps 7
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import time
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
EPS = 2e-5


def load(name: str) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(SUITE / f"{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def lx_once(d: dict[str, Any]) -> tuple[float, float, str]:
    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    bounds = [
        (None if lo == -INF else float(lo), None if hi == INF else float(hi))
        for lo, hi in zip(d["lo"], d["hi"], strict=True)
    ]
    p = SparseLPProblem(
        c=d["c"].tolist(),
        A_eq=from_scipy_sparse(d["A"]),
        b_eq=d["b"].tolist(),
        objective="min",
        bounds=bounds,
    )
    t0, c0 = time.perf_counter(), time.process_time()
    r = SparseSolver(algorithm="auto", max_iterations=50_000, eps=EPS, check_interval=50_000).solve(
        p
    )
    return (
        (time.perf_counter() - t0) * 1e3,
        (time.process_time() - c0) * 1e3,
        r.solution.status.value,
    )


def hx_once(d: dict[str, Any]) -> tuple[float, float, str]:
    import numpy as np
    from scipy.optimize import linprog

    bounds = list(
        zip(
            np.where(np.isfinite(d["lo"]), d["lo"], -np.inf),
            np.where(np.isfinite(d["hi"]), d["hi"], np.inf),
            strict=True,
        )
    )
    t0, c0 = time.perf_counter(), time.process_time()
    r = linprog(d["c"], A_eq=d["A"], b_eq=d["b"], bounds=bounds, method="highs")
    return (
        (time.perf_counter() - t0) * 1e3,
        (time.process_time() - c0) * 1e3,
        "optimal" if r.success else "fail",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--instance", default="lp_greenbea")
    args = ap.parse_args()

    affinity = os.sched_getaffinity(0)
    print(
        f"instance={args.instance}  visible CPUs={os.cpu_count()}  "
        f"affinity={len(affinity)} cpus  reps={args.reps}"
    )

    d = load(args.instance)
    lx_once(d), hx_once(d)  # warm

    lxw, lxc, hxw, hxc = [], [], [], []
    for i in range(args.reps):
        w, c, s1 = lx_once(d)
        lxw.append(w)
        lxc.append(c)
        w2, c2, s2 = hx_once(d)
        hxw.append(w2)
        hxc.append(c2)
        print(
            f"  rep {i + 1}: lx {w:8.1f} ms (cpu {c:8.1f})  "
            f"hx {w2:8.1f} ms (cpu {c2:8.1f})  {s1}/{s2}",
            flush=True,
        )

    mlw, mhw = st.median(lxw), st.median(hxw)
    mlc, mhc = st.median(lxc), st.median(hxc)
    print(f"\n  median WALL  lx {mlw:8.1f}  hx {mhw:8.1f}   ratio lx/hx = {mlw / mhw:.3f}")
    print(f"  median CPU   lx {mlc:8.1f}  hx {mhc:8.1f}   ratio lx/hx = {mlc / mhc:.3f}")
    par = mlc / mlw
    print(f"\n  linprogx CPU/wall = {par:.2f}  (>1.15 means it is using threads)")
    print(f"  HiGHS    CPU/wall = {mhc / mhw:.2f}")
    print("\n  Board ratio to beat: 1.156 (Modal 4-vCPU, protocol v3)")


if __name__ == "__main__":
    main()
