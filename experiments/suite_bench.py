"""LPnetlib suite benchmark: linprogx (auto) vs SciPy/HiGHS vs Clarabel.

Driver mode runs every (instance, solver) pair in an isolated subprocess
with a timeout, appends rows to a JSONL file incrementally, and writes a
markdown summary at the end. Objective deltas are reported against HiGHS
as the reference.

Usage:
    PYTHONPATH=. uv run python experiments/suite_bench.py /tmp/lpsuite \
        --out /tmp/lpsuite_results.jsonl --markdown /tmp/lpsuite_summary.md
    PYTHONPATH=. uv run python experiments/suite_bench.py --worker FILE SOLVER
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 180.0
SOLVERS = ("highs", "clarabel", "linprogx")


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


def solve_linprogx(data: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    matrix = from_scipy_sparse(data["A_scipy"])
    start = time.perf_counter()
    result = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=matrix,
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
        )
    )
    seconds = time.perf_counter() - start
    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))
    row: dict[str, Any] = {
        "status": result.solution.status.value,
        "objective": result.solution.objective_value,
        "seconds": seconds,
        "residual": residual,
        "backend": result.backend.rsplit("-", 1)[-1],
        "iterations": result.solution.iterations,
    }
    if result.ipm_slice_us is not None:
        row["ipm_slice_us"] = result.ipm_slice_us
    return row


def solve_highs(data: dict[str, Any]) -> dict[str, Any]:
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
        "status": "optimal" if result.success else f"status_{result.status}",
        "objective": float(result.fun) if result.success else None,
        "seconds": seconds,
        "residual": None,
        "backend": "highs",
        "iterations": None,
    }


def solve_clarabel(data: dict[str, Any]) -> dict[str, Any]:
    import clarabel
    import numpy as np
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
    solved = status in {"Solved", "AlmostSolved"}
    objective = float(c @ np.array(result.x)) if solved else None
    residual = None
    if solved:
        x = np.array(result.x, dtype=float)
        residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"])))
    return {
        "status": "optimal" if solved else status,
        "objective": objective,
        "seconds": seconds,
        "residual": residual,
        "backend": "clarabel",
        "iterations": None,
    }


def run_worker(path: Path, solver: str) -> int:
    data = load_instance(path)
    runner = {"linprogx": solve_linprogx, "highs": solve_highs, "clarabel": solve_clarabel}[solver]
    print(json.dumps(runner(data)))
    return 0


def run_driver(directory: Path, out: Path, markdown: Path) -> int:
    instances = sorted(directory.glob("lp_*.mat"), key=lambda p: p.stat().st_size)
    done: set[tuple[str, str]] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["instance"], row["solver"]))
    for path in instances:
        for solver in SOLVERS:
            if (path.stem, solver) in done:
                continue
            row: dict[str, Any] = {"instance": path.stem, "solver": solver}
            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--worker", str(path), solver],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
                else:
                    row.update(
                        {
                            "status": "crashed",
                            "seconds": time.perf_counter() - started,
                            "error": proc.stderr.strip()[-300:],
                        }
                    )
            except subprocess.TimeoutExpired:
                row.update({"status": "timeout", "seconds": TIMEOUT_SECONDS})
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"{row['instance']:>14} {row['solver']:>9}: {row.get('status'):<12} "
                f"{row.get('seconds', 0.0):8.2f}s {row.get('backend', '')}",
                flush=True,
            )
    write_markdown(out, markdown)
    return 0


def write_markdown(out: Path, markdown: Path) -> None:
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    by_instance: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], {})[row["solver"]] = row
    lines = [
        "| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in sorted(by_instance):
        cells = by_instance[name]

        def fmt(solver: str, cells: dict[str, dict[str, Any]] = cells) -> str:
            row = cells.get(solver)
            if row is None:
                return "n/a"
            if row.get("status") not in ("optimal",):
                return f"{row.get('status')}"
            return f"{row['seconds']:.2f}s"

        lx = cells.get("linprogx", {})
        ref = cells.get("highs", {})
        delta = (
            f"{abs(lx['objective'] - ref['objective']):.2e}"
            if lx.get("objective") is not None and ref.get("objective") is not None
            else "n/a"
        )
        residual = f"{lx['residual']:.1e}" if lx.get("residual") is not None else "n/a"
        lines.append(
            f"| {name} | {fmt('linprogx')} | {fmt('highs')} | {fmt('clarabel')} "
            f"| {delta} | {residual} | {lx.get('backend', 'n/a')} |"
        )
    markdown.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("/tmp/lpsuite"))
    parser.add_argument("--worker", nargs=2, metavar=("FILE", "SOLVER"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/lpsuite_results.jsonl"))
    parser.add_argument("--markdown", type=Path, default=Path("/tmp/lpsuite_summary.md"))
    args = parser.parse_args()
    if args.worker:
        return run_worker(Path(args.worker[0]), args.worker[1])
    return run_driver(args.directory, args.out, args.markdown)


if __name__ == "__main__":
    raise SystemExit(main())
