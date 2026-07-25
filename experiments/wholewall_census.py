"""Whole-wall census: charge every millisecond of SparseSolver.solve() on greenbea.

Every prior campaign kill measured a slice INSIDE the simplex loop (pivots,
BTRAN/FTRAN, factor, pricing, trajectory).  The board gate is on the complete
solve() wall.  This probe charges the complement: presolve, Python marshalling,
the C dual-simplex call itself, postsolve, objective, and the residual check.

Usage:
    PYTHONPATH=. uv run python experiments/wholewall_census.py [--repeats 9] [--instance lp_greenbea]
"""

from __future__ import annotations

import argparse
import statistics
import time
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


def bounds_of(data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    return [
        (
            None if low == float("-inf") else float(low),
            None if up == float("inf") else float(up),
        )
        for low, up in zip(data["lo"], data["hi"], strict=True)
    ]


def instrument() -> dict[str, list[float]]:
    """Monkeypatch the phase boundaries inside sparse.py and record wall time."""
    from linprogx import presolve as presolve_mod
    from linprogx import sparse as sparse_mod

    marks: dict[str, list[float]] = {}

    def record(name: str, elapsed: float) -> None:
        marks.setdefault(name, []).append(elapsed)

    def wrap(module: Any, name: str, label: str) -> None:
        original = getattr(module, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                record(label, time.perf_counter() - begin)

        timed.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(module, name, timed)

    wrap(sparse_mod, "presolve_matrix", "presolve")
    wrap(sparse_mod, "postsolve_x", "postsolve")
    wrap(sparse_mod, "_max_equality_residual", "residual_check")
    wrap(presolve_mod, "postsolve_x", "postsolve_inner")

    return marks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--instance", default="lp_greenbea")
    parser.add_argument("--no-instrument", action="store_true")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="deterministic profile of one solve, charging every Python and C entry point",
    )
    args = parser.parse_args()

    import numpy as np

    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    data = load_instance(SUITE / f"{args.instance}.mat")

    def build() -> Any:
        matrix = from_scipy_sparse(data["A_scipy"])
        return SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=matrix,
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )

    if args.cprofile:
        import cProfile
        import pstats

        problem = build()
        solver = SparseSolver(
            algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
        )
        profiler = cProfile.Profile()
        profiler.enable()
        solver.solve(problem)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime")
        print("--- cProfile tottime (includes C entry points) ---")
        stats.print_stats(30)
        return

    marks = {} if args.no_instrument else instrument()

    walls: list[float] = []
    iterations = 0
    status = ""
    objective = 0.0
    for _ in range(args.repeats):
        matrix = from_scipy_sparse(data["A_scipy"])
        problem = SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=matrix,
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )
        begin = time.perf_counter()
        result = SparseSolver(
            algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
        ).solve(problem)
        walls.append(time.perf_counter() - begin)
        iterations = result.solution.iterations
        status = result.solution.status.value
        objective = float(result.solution.objective_value)

    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))

    median_wall = statistics.median(walls)
    print(f"instance         {args.instance}")
    print(f"rows x cols      {data['A_scipy'].shape[0]} x {data['A_scipy'].shape[1]}")
    print(f"nnz              {data['A_scipy'].nnz}")
    print(f"status           {status}")
    print(f"objective        {objective!r}")
    print(f"iterations       {iterations}")
    print(f"residual         {residual:.3e}")
    print(f"repeats          {args.repeats}")
    print(f"wall median      {median_wall * 1e3:.3f} ms")
    print(f"wall min         {min(walls) * 1e3:.3f} ms")
    print(f"wall all         {[round(w * 1e3, 2) for w in walls]}")

    if marks:
        print("\n--- charged phases (median over repeats, ms) ---")
        charged = 0.0
        rows = []
        for name, samples in marks.items():
            med = statistics.median(samples)
            rows.append((med, name, len(samples)))
        for med, name, count in sorted(rows, reverse=True):
            share = 100.0 * med / median_wall
            print(f"{name:26s} {med * 1e3:9.3f} ms  {share:6.2f}%  (n={count})")
            # postsolve_inner double-counts postsolve; residual_check may fire twice
            if name not in {"postsolve_inner"}:
                charged += med
        print(f"{'CHARGED TOTAL':26s} {charged * 1e3:9.3f} ms  {100.0 * charged / median_wall:6.2f}%")
        gap = median_wall - charged
        print(f"{'UNCHARGED (python glue)':26s} {gap * 1e3:9.3f} ms  {100.0 * gap / median_wall:6.2f}%")


if __name__ == "__main__":
    main()
