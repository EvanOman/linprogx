"""Generalization check: race the auto-routed solver against HiGHS and
Clarabel on Netlib instances that were never used while tuning.

Usage: PYTHONPATH=. uv run python experiments/generalization_bench.py /tmp/lpgen
The directory must hold LPnetlib .mat files (lp_*.mat).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse


def load_instance(path: Path) -> dict[str, Any]:
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": from_scipy_sparse(raw["A"]),
        "A_scipy": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(float),
        "c": aux["c"].ravel().astype(float),
        "lo": aux["lo"].ravel().astype(float),
        "hi": aux["hi"].ravel().astype(float),
    }


def bounds_of(data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    return [
        (
            None if low == float("-inf") else float(low),
            None if up == float("inf") else float(up),
        )
        for low, up in zip(data["lo"], data["hi"], strict=True)
    ]


def run_linprogx(data: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=data["A"],
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )
    )
    seconds = time.perf_counter() - start
    x = np.array(result.solution.x, dtype=float)
    res = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))
    return {
        "solver": f"linprogx[{result.backend.rsplit('-', 1)[-1]}]",
        "status": result.solution.status.value,
        "objective": result.solution.objective_value,
        "seconds": seconds,
        "residual": res,
        "iters": result.solution.iterations,
    }


def run_highs(data: dict[str, Any]) -> dict[str, Any]:
    from scipy.optimize import linprog

    start = time.perf_counter()
    result = linprog(
        data["c"],
        A_eq=data["A_scipy"],
        b_eq=data["b"],
        bounds=bounds_of(data),
        method="highs",
    )
    seconds = time.perf_counter() - start
    return {
        "solver": "HiGHS",
        "status": "optimal" if result.success else f"status_{result.status}",
        "objective": float(result.fun) if result.success else None,
        "seconds": seconds,
        "residual": None,
        "iters": None,
    }


def run_clarabel(data: dict[str, Any]) -> dict[str, Any]:
    import clarabel
    from scipy import sparse

    c = data["c"]
    lo = data["lo"]
    hi = data["hi"]
    finite_hi = np.isfinite(hi)
    finite_lo = np.isfinite(lo)
    eye = sparse.eye(len(c), format="csc")
    A = sparse.vstack([data["A_scipy"], eye[finite_hi], -eye[finite_lo]], format="csc")
    b = np.concatenate([data["b"], hi[finite_hi], -lo[finite_lo]])
    api = vars(clarabel)
    cones = [
        api["ZeroConeT"](data["A_scipy"].shape[0]),
        api["NonnegativeConeT"](int(finite_hi.sum() + finite_lo.sum())),
    ]
    P = sparse.csc_matrix((len(c), len(c)))
    settings: Any = api["DefaultSettings"]()
    settings.verbose = False
    start = time.perf_counter()
    result: Any = api["DefaultSolver"](P, c, A, b, cones, settings).solve()
    seconds = time.perf_counter() - start
    status = str(result.status)
    objective = float(c @ np.array(result.x)) if status in {"Solved", "AlmostSolved"} else None
    return {
        "solver": "Clarabel",
        "status": status,
        "objective": objective,
        "seconds": seconds,
        "residual": None,
        "iters": None,
    }


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/lpgen")
    for path in sorted(directory.glob("lp_*.mat")):
        data = load_instance(path)
        rows, cols = data["A"].shape
        print(f"\n== {path.stem} ({rows}x{cols}, nnz {data['A'].nnz}) ==", flush=True)
        reference = None
        for runner in (run_highs, run_clarabel, run_linprogx):
            try:
                row = runner(data)
            except Exception as exc:  # pragma: no cover - exploratory harness
                print(f"  {runner.__name__}: FAILED {str(exc)[:90]}", flush=True)
                continue
            if row["solver"] == "HiGHS" and row["objective"] is not None:
                reference = row["objective"]
            delta = (
                abs(row["objective"] - reference)
                if reference is not None and row["objective"] is not None
                else None
            )
            extras = []
            if delta is not None:
                extras.append(f"delta_vs_HiGHS={delta:.3e}")
            if row["residual"] is not None:
                extras.append(f"residual={row['residual']:.3e}")
            if row["iters"] is not None:
                extras.append(f"iters={row['iters']}")
            print(
                f"  {row['solver']:>18}: {row['status']:<12} {row['seconds']:7.3f}s  "
                + " ".join(extras),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
