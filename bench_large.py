from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

EXPECTED_DFL001_OBJECTIVE = 11_266_396.047
DATA_PATH = Path("benchmark_data/netlib_dfl001/lp_dfl001.mat")


@dataclass(frozen=True)
class LargeProblem:
    name: str
    rows: int
    cols: int
    nonzeros: int
    expected_objective: float
    source_url: str


@dataclass(frozen=True)
class LargeBenchRow:
    solver: str
    status: str
    objective: float | None
    objective_delta: float | None
    seconds: float | None
    notes: str


@dataclass(frozen=True)
class LargeBenchResult:
    problem: LargeProblem
    rows: list[LargeBenchRow]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the large Netlib DFL001 benchmark.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=Path("assets/large_dfl001_results.json"))
    parser.add_argument("--plot", type=Path, default=Path("assets/large_dfl001_runtime.png"))
    parser.add_argument("--markdown", type=Path, default=Path("assets/large_dfl001_summary.md"))
    args = parser.parse_args()

    result = run_benchmark(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    write_markdown(result, args.markdown)
    write_plot(result, args.plot)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


def run_benchmark(path: Path) -> LargeBenchResult:
    problem_data = load_dfl001(path)
    problem = LargeProblem(
        name="Netlib DFL001",
        rows=problem_data["A"].shape[0],
        cols=problem_data["A"].shape[1],
        nonzeros=problem_data["A"].nnz,
        expected_objective=EXPECTED_DFL001_OBJECTIVE,
        source_url="https://sparse.tamu.edu/mat/LPnetlib/lp_dfl001.mat",
    )
    return LargeBenchResult(
        problem=problem,
        rows=[
            _run_linprogx_sparse(problem_data),
            _run_scipy(problem_data),
            _run_clarabel(problem_data),
        ],
    )


def load_dfl001(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        msg = "SciPy is required to load SuiteSparse .mat benchmark data"
        raise RuntimeError(msg) from exc

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


def write_markdown(result: LargeBenchResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Solver | Status | Objective | Delta vs published | Runtime | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in result.rows:
        objective = "n/a" if row.objective is None else f"{row.objective:.6f}"
        delta = "n/a" if row.objective_delta is None else f"{row.objective_delta:.3e}"
        seconds = "n/a" if row.seconds is None else f"{row.seconds:.3f}s"
        lines.append(
            f"| {row.solver} | {row.status} | {objective} | {delta} | {seconds} | {row.notes} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_plot(result: LargeBenchResult, path: Path) -> None:
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")

    solved = [row for row in result.rows if row.seconds is not None]
    fig, ax = pyplot.subplots(figsize=(9, 4.8))
    color_map = {"linprogx-sparse": "#28536b", "SciPy/HiGHS": "#c44536", "Clarabel": "#f3a712"}
    colors = [color_map.get(row.solver, "#6c757d") for row in solved]
    seconds = [float(row.seconds) for row in solved if row.seconds is not None]
    ax.bar([row.solver for row in solved], seconds, color=colors)
    ax.set_ylabel("seconds")
    ax.set_title("Large Netlib DFL001 Runtime")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    for index, row in enumerate(solved):
        ax.text(index, row.seconds or 0.0, f"{row.seconds:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    pyplot.close(fig)


def _run_linprogx_sparse(problem_data: dict[str, Any]) -> LargeBenchRow:
    result = SparseSolver(
        algorithm="pdhg",
        max_iterations=200_000,
        eps=2e-5,
        objective_scale=50_000.0,
        check_interval=200_000,
    ).solve(
        SparseLPProblem(
            c=problem_data["c"].tolist(),
            A_eq=problem_data["A"],
            b_eq=problem_data["b"].tolist(),
            objective="min",
            bounds=_bounds(problem_data),
            name="dfl001",
        )
    )
    objective = result.solution.objective_value
    return LargeBenchRow(
        solver="linprogx-sparse",
        status=result.solution.status.value,
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_DFL001_OBJECTIVE),
        seconds=result.seconds,
        notes=(
            f"C CSR matrix with {result.backend}; equality+bounds PDHG, "
            f"objective_scale=5e4; {result.solution.message}"
        ),
    )


def _run_scipy(problem_data: dict[str, Any]) -> LargeBenchRow:
    from scipy.optimize import linprog

    start = time.perf_counter()
    result = linprog(
        problem_data["c"],
        A_eq=problem_data["A_scipy"],
        b_eq=problem_data["b"],
        bounds=_bounds(problem_data),
        method="highs",
    )
    seconds = time.perf_counter() - start
    objective = float(result.fun) if result.success else None
    return LargeBenchRow(
        solver="SciPy/HiGHS",
        status="optimal" if result.success else f"status_{result.status}",
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_DFL001_OBJECTIVE),
        seconds=seconds,
        notes=str(result.message).replace("|", "/"),
    )


def _run_clarabel(problem_data: dict[str, Any]) -> LargeBenchRow:
    import clarabel
    import numpy as np
    from scipy import sparse

    c = problem_data["c"]
    lo = problem_data["lo"]
    hi = problem_data["hi"]
    finite_hi = np.isfinite(hi)
    finite_lo = np.isfinite(lo)
    eye = sparse.eye(len(c), format="csc")
    rows = [problem_data["A_scipy"], eye[finite_hi], -eye[finite_lo]]
    rhs = [problem_data["b"], hi[finite_hi], -lo[finite_lo]]
    clarabel_api = vars(clarabel)
    cones = [
        clarabel_api["ZeroConeT"](problem_data["A_scipy"].shape[0]),
        clarabel_api["NonnegativeConeT"](int(finite_hi.sum() + finite_lo.sum())),
    ]
    A = sparse.vstack(rows, format="csc")
    b = np.concatenate(rhs)
    P = sparse.csc_matrix((len(c), len(c)))
    settings: Any = clarabel_api["DefaultSettings"]()
    settings.verbose = False
    settings.max_iter = 1000

    start = time.perf_counter()
    result: Any = clarabel_api["DefaultSolver"](P, c, A, b, cones, settings).solve()
    seconds = time.perf_counter() - start
    status = str(result.status)
    objective = float(result.obj_val) if status in {"Solved", "AlmostSolved"} else None
    return LargeBenchRow(
        solver="Clarabel",
        status=_clarabel_status(status),
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_DFL001_OBJECTIVE),
        seconds=seconds,
        notes=f"Clarabel status: {status}",
    )


def _linprogx_skip(problem: LargeProblem) -> LargeBenchRow:
    estimated_dense_entries = problem.rows * problem.cols
    return LargeBenchRow(
        solver="linprogx",
        status="skipped",
        objective=None,
        objective_delta=None,
        seconds=None,
        notes=(
            "Dense Python tableau skipped; raw A alone would materialize "
            f"{estimated_dense_entries:,} coefficients before slacks/artificials."
        ),
    )


def _bounds(problem_data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    lo = problem_data["lo"]
    hi = problem_data["hi"]
    return [
        (
            None if lower == float("-inf") else float(lower),
            None if upper == float("inf") else float(upper),
        )
        for lower, upper in zip(lo, hi, strict=True)
    ]


def _clarabel_status(status: str) -> str:
    normalized = status.lower()
    if normalized in {"solved", "almostsolved"}:
        return "optimal"
    if normalized in {"dualinfeasible", "almostdualinfeasible"}:
        return "reported_dual_infeasible"
    if normalized in {"primalinfeasible", "almostprimalinfeasible"}:
        return "reported_primal_infeasible"
    return normalized


def _jsonable(result: LargeBenchResult) -> dict[str, Any]:
    return {
        "problem": asdict(result.problem),
        "rows": [asdict(row) for row in result.rows],
    }


if __name__ == "__main__":
    raise SystemExit(main())
