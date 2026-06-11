from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

EXPECTED_CYCLE_OBJECTIVE = -5.2263930249
DATA_PATH = Path("benchmark_data/netlib_cycle/lp_cycle.mat")


@dataclass(frozen=True)
class CycleProblem:
    name: str
    rows: int
    cols: int
    nonzeros: int
    expected_objective: float
    source_url: str


@dataclass(frozen=True)
class CycleBenchRow:
    solver: str
    status: str
    objective: float | None
    objective_delta: float | None
    seconds: float | None
    notes: str


@dataclass(frozen=True)
class CycleBenchResult:
    problem: CycleProblem
    rows: list[CycleBenchRow]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Netlib CYCLE sparse benchmark.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=Path("assets/cycle_results.json"))
    parser.add_argument("--markdown", type=Path, default=Path("assets/cycle_summary.md"))
    args = parser.parse_args()

    result = run_benchmark(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    write_markdown(result, args.markdown)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


def run_benchmark(path: Path) -> CycleBenchResult:
    problem_data = load_cycle(path)
    problem = CycleProblem(
        name="Netlib CYCLE",
        rows=problem_data["A"].shape[0],
        cols=problem_data["A"].shape[1],
        nonzeros=problem_data["A"].nnz,
        expected_objective=EXPECTED_CYCLE_OBJECTIVE,
        source_url="https://sparse.tamu.edu/mat/LPnetlib/lp_cycle.mat",
    )
    return CycleBenchResult(
        problem=problem,
        rows=[
            _run_linprogx_sparse(problem_data),
            _run_scipy(problem_data),
            _run_clarabel(problem_data),
        ],
    )


def load_cycle(path: Path) -> dict[str, Any]:
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


def write_markdown(result: CycleBenchResult, path: Path) -> None:
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


def _run_linprogx_sparse(problem_data: dict[str, Any]) -> CycleBenchRow:
    result = SparseSolver(
        algorithm="auto",
        max_iterations=50_000,
        eps=2e-5,
        check_interval=50_000,
    ).solve(
        SparseLPProblem(
            c=problem_data["c"].tolist(),
            A_eq=problem_data["A"],
            b_eq=problem_data["b"].tolist(),
            objective="min",
            bounds=_bounds(problem_data),
            name="cycle",
        )
    )
    objective = result.solution.objective_value
    return CycleBenchRow(
        solver="linprogx-sparse",
        status=result.solution.status.value,
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_CYCLE_OBJECTIVE),
        seconds=result.seconds,
        notes=(f"C CSR matrix with {result.backend}; equality+bounds; {result.solution.message}"),
    )


def _run_scipy(problem_data: dict[str, Any]) -> CycleBenchRow:
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
    return CycleBenchRow(
        solver="SciPy/HiGHS",
        status="optimal" if result.success else f"status_{result.status}",
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_CYCLE_OBJECTIVE),
        seconds=seconds,
        notes=str(result.message).replace("|", "/"),
    )


def _run_clarabel(problem_data: dict[str, Any]) -> CycleBenchRow:
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
    settings.max_iter = 2000
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10

    start = time.perf_counter()
    result: Any = clarabel_api["DefaultSolver"](P, c, A, b, cones, settings).solve()
    seconds = time.perf_counter() - start
    status = str(result.status)
    objective = None
    notes = f"Clarabel status: {status}"
    if status in {"Solved", "AlmostSolved"}:
        x = np.array(result.x, dtype=float)
        objective = float(c @ x)
        max_eq_residual = float(np.max(np.abs(problem_data["A_scipy"] @ x - problem_data["b"])))
        notes = f"{notes}; max equality residual {max_eq_residual:.3e}"
    return CycleBenchRow(
        solver="Clarabel",
        status=_clarabel_status(status),
        objective=objective,
        objective_delta=None if objective is None else abs(objective - EXPECTED_CYCLE_OBJECTIVE),
        seconds=seconds,
        notes=notes,
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


def _jsonable(result: CycleBenchResult) -> dict[str, Any]:
    return {
        "problem": asdict(result.problem),
        "rows": [asdict(row) for row in result.rows],
    }


if __name__ == "__main__":
    raise SystemExit(main())
