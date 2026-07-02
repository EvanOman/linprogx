from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Literal

try:
    CSRMatrix = importlib.import_module("linprogx._csparse").CSRMatrix
except ImportError:  # pragma: no cover - source tree before extension build
    CSRMatrix = None  # type: ignore[assignment]

from linprogx.presolve import postsolve_x, presolve_eq_box
from linprogx.types import ObjectiveSense, Solution, Status

SparseSense = Literal["<=", ">=", "="]


def _max_equality_residual(matrix: Any, x: list[float], b: list[float]) -> float:
    ax = matrix.matvec(x)
    return max((abs(float(lhs) - rhs) for lhs, rhs in zip(ax, b, strict=True)), default=0.0)


@dataclass(frozen=True)
class SparseLPProblem:
    c: list[float]
    A_eq: Any | None = None
    b_eq: list[float] | None = None
    G_ub: Any | None = None
    h_ub: list[float] | None = None
    objective: ObjectiveSense = "min"
    bounds: list[tuple[float | None, float | None]] | None = None
    name: str = "sparse-lp"

    def __post_init__(self) -> None:
        width = len(self.c)
        if width == 0:
            msg = "objective must contain at least one coefficient"
            raise ValueError(msg)
        if self.A_eq is not None:
            rows, cols = self.A_eq.shape
            if cols != width:
                msg = "A_eq column count must match objective width"
                raise ValueError(msg)
            if self.b_eq is None or len(self.b_eq) != rows:
                msg = "b_eq length must match A_eq row count"
                raise ValueError(msg)
        if self.G_ub is not None:
            rows, cols = self.G_ub.shape
            if cols != width:
                msg = "G_ub column count must match objective width"
                raise ValueError(msg)
            if self.h_ub is None or len(self.h_ub) != rows:
                msg = "h_ub length must match G_ub row count"
                raise ValueError(msg)
        if self.bounds is not None and len(self.bounds) != width:
            msg = "bounds width must match objective width"
            raise ValueError(msg)


@dataclass(frozen=True)
class SparseSolveResult:
    solution: Solution
    backend: str
    seconds: float


@dataclass
class _PreparedSparse:
    c_max: list[float]
    rows: list[dict[int, float]]
    senses: list[SparseSense]
    rhs: list[float]
    reconstruct: list[tuple[int | None, int | None, float]]


@dataclass
class _SparseTableau:
    rows: list[dict[int, float]]
    basis: list[int]
    artificial: set[int]
    width: int
    original_width: int


class SparseSolver:
    """Dependency-free sparse two-phase simplex over linprogx's C CSR type."""

    #: reduced problems at or below this many rows attempt the interior
    #: point method under ``algorithm="auto"``; direct factorization wins
    #: whenever the Cholesky factor is affordable, and the C side aborts
    #: back to PDHG if the ordering or the factor turns out too expensive
    #: (minimum-degree work budget and a factor-flops cap).
    AUTO_IPM_MAX_ROWS = 50_000
    DEFAULT_PDHG_THREADS = 4

    def __init__(
        self,
        *,
        eps: float = 1e-9,
        max_iterations: int = 50_000,
        algorithm: Literal["simplex", "pdhg", "ipm", "auto"] = "simplex",
        objective_scale: float | None = None,
        check_interval: int | None = None,
        presolve: bool = True,
        threads: int | None = None,
    ) -> None:
        if threads is not None and threads < 0:
            msg = "threads must be nonnegative"
            raise ValueError(msg)
        self.eps = eps
        self.max_iterations = max_iterations
        self.algorithm = algorithm
        self.objective_scale = objective_scale
        self.check_interval = check_interval
        self.presolve = presolve
        self.threads = threads

    def solve(self, problem: SparseLPProblem) -> SparseSolveResult:
        start = time.perf_counter()
        if self.algorithm in ("pdhg", "ipm", "auto"):
            solution, backend = self._solve_eq_box(problem, self.algorithm)
        else:
            solution = self._solve(problem)
            backend = "native-sparse-simplex"
        return SparseSolveResult(solution, backend, time.perf_counter() - start)

    def _solve_eq_box(self, problem: SparseLPProblem, algorithm: str) -> tuple[Solution, str]:
        backend = f"native-c-sparse-{algorithm}"
        if problem.objective != "min":
            return Solution(
                Status.INFEASIBLE, message="PDHG sparse path currently expects minimization"
            ), backend
        if problem.A_eq is None or problem.b_eq is None:
            return Solution(
                Status.INFEASIBLE, message="PDHG sparse path expects equality constraints"
            ), backend
        if problem.G_ub is not None:
            return Solution(
                Status.INFEASIBLE, message="PDHG sparse path expects bounds instead of G_ub"
            ), backend
        bounds = problem.bounds or [(0.0, None) for _ in problem.c]
        lo = [float("-inf") if lower is None else float(lower) for lower, _ in bounds]
        hi = [float("inf") if upper is None else float(upper) for _, upper in bounds]
        c = [float(value) for value in problem.c]
        b = [float(value) for value in problem.b_eq]

        matrix = problem.A_eq
        reduction = None
        if self.presolve:
            rows, cols = matrix.shape
            indptr, indices, data = matrix.to_components()
            reduction = presolve_eq_box(rows, cols, indptr, indices, data, b, c, lo, hi)
        if reduction is not None:
            matrix = csr_matrix(
                reduction.rows,
                reduction.cols,
                reduction.indptr,
                reduction.indices,
                reduction.data,
            )
            solve_c, solve_b = reduction.c, reduction.b
            solve_lo, solve_hi = reduction.lo, reduction.hi
        else:
            solve_c, solve_b, solve_lo, solve_hi = c, b, lo, hi

        chosen = algorithm
        if algorithm == "auto":
            rows, _ = matrix.shape
            chosen = "ipm" if rows <= self.AUTO_IPM_MAX_ROWS else "pdhg"

        result = None
        feasible_ipm_candidate: tuple[list[float], float, int, float] | None = None
        if chosen == "ipm":
            result = matrix.solve_eq_box_ipm(
                solve_c,
                solve_b,
                solve_lo,
                solve_hi,
                max_iter=min(self.max_iterations, 200),
                tol=min(self.eps, 1e-9),
                threads=0 if self.threads is None else self.threads,
                feas_tol=self.eps,
            )
            if result["status"] == "optimal":
                candidate_x = [float(value) for value in result["x"]]
                if reduction is not None:
                    candidate_x = postsolve_x(candidate_x, reduction)
                if _max_equality_residual(problem.A_eq, candidate_x, b) > self.eps:
                    result["status"] = "raw_feasibility_failure"
            elif result["status"] != "factor_too_dense":
                candidate_x = [float(value) for value in result["x"]]
                if reduction is not None:
                    candidate_x = postsolve_x(candidate_x, reduction)
                candidate_residual = _max_equality_residual(problem.A_eq, candidate_x, b)
                if candidate_residual <= self.eps:
                    feasible_ipm_candidate = (
                        candidate_x,
                        sum(v * coef for v, coef in zip(candidate_x, problem.c, strict=True)),
                        int(result["iterations"]),
                        candidate_residual,
                    )
            if result["status"] != "optimal" and result["status"] != "factor_too_dense":
                # The fast BLAS dpotrf tail factor lacks the hand
                # kernel's per-pivot floor, so on degenerate endgames it
                # can land a point the Lagrangian certificate can't
                # close. Retry the SAME (presolved) problem with the
                # floored kernel first -- cheap and usually enough --
                # then, only if that also fails, the unpresolved problem
                # (an independent trajectory). The certificate gate
                # keeps every accepted retry sound.
                retries: list[
                    tuple[Any, list[float], list[float], list[float], list[float], str]
                ] = [(matrix, solve_c, solve_b, solve_lo, solve_hi, "floored retry")]
                if reduction is not None:
                    retries.append((problem.A_eq, c, b, lo, hi, "unpresolved floored retry"))
                for rmatrix, rc, rb, rlo, rhi, note in retries:
                    retry_result = rmatrix.solve_eq_box_ipm(
                        rc,
                        rb,
                        rlo,
                        rhi,
                        max_iter=min(self.max_iterations, 200),
                        tol=min(self.eps, 1e-9),
                        threads=0 if self.threads is None else self.threads,
                        blas=False,
                        feas_tol=self.eps,
                    )
                    if retry_result["status"] == "optimal":
                        is_raw = rmatrix is problem.A_eq and reduction is not None
                        rx = [float(value) for value in retry_result["x"]]
                        if is_raw:
                            objective = float(retry_result["objective"])
                        else:
                            rx = postsolve_x(rx, reduction) if reduction is not None else rx
                            objective = sum(v * coef for v, coef in zip(rx, problem.c, strict=True))
                        residual = _max_equality_residual(problem.A_eq, rx, b)
                        if residual > self.eps:
                            continue
                        return Solution(
                            Status.OPTIMAL,
                            x=rx,
                            objective_value=objective,
                            iterations=int(retry_result["iterations"]),
                            message=(
                                f"native sparse IPM converged on the {note}; "
                                f"max equality residual {residual:.3e}"
                            ),
                        ), "native-c-sparse-ipm"
                if algorithm == "auto" and feasible_ipm_candidate is not None:
                    fx, fobj, fiters, fresidual = feasible_ipm_candidate
                    return Solution(
                        Status.ITERATION_LIMIT,
                        fobj,
                        fx,
                        message=(
                            "native sparse auto kept the best feasible IPM candidate; "
                            f"max equality residual {fresidual:.3e}"
                        ),
                        iterations=fiters,
                    ), "native-c-sparse-ipm"
            if result["status"] != "optimal" and (
                algorithm == "auto" or result["status"] == "factor_too_dense"
            ):
                chosen = "pdhg"
                result = None
        if result is None:
            chosen = "pdhg"
            result = matrix.solve_eq_box_pdhg(
                solve_c,
                solve_b,
                solve_lo,
                solve_hi,
                max_iter=self.max_iterations,
                tol=self.eps,
                check_interval=self.check_interval or 250,
                objective_scale=0.0 if self.objective_scale is None else self.objective_scale,
                threads=self.DEFAULT_PDHG_THREADS if self.threads is None else self.threads,
            )
        backend = f"native-c-sparse-{chosen}"

        x = [float(value) for value in result["x"]]
        objective = float(result["objective"])
        if reduction is not None:
            x = postsolve_x(x, reduction)
            objective = sum(value * coef for value, coef in zip(x, problem.c, strict=True))

        residual = _max_equality_residual(problem.A_eq, x, b)
        status = (
            Status.OPTIMAL
            if result["status"] == "optimal" and residual <= self.eps
            else Status.ITERATION_LIMIT
        )
        presolve_note = (
            f"; presolve removed {reduction.removed_rows} rows and {reduction.removed_cols} cols"
            if reduction is not None
            else ""
        )
        if (
            status != Status.OPTIMAL
            and feasible_ipm_candidate is not None
            and residual > feasible_ipm_candidate[3]
        ):
            x, objective, iterations, residual = feasible_ipm_candidate
            return Solution(
                Status.ITERATION_LIMIT,
                objective,
                x,
                message=(
                    "native sparse auto kept the best feasible IPM candidate after "
                    f"fallback failed; max equality residual {residual:.3e}{presolve_note}"
                ),
                iterations=iterations,
            ), "native-c-sparse-ipm"
        if chosen == "ipm":
            verb = "converged" if status == Status.OPTIMAL else "hit the iteration limit"
            message = (
                f"native sparse IPM {verb}; max equality residual {residual:.3e}{presolve_note}"
            )
        else:
            objective_scale = float(result["objective_scale"])
            verb = "converged" if status == Status.OPTIMAL else "hit the iteration limit"
            message = (
                f"native sparse PDHG {verb}; "
                f"max equality residual {residual:.3e}; objective scale {objective_scale:.3g}"
                f"{presolve_note}"
            )
        return Solution(
            status,
            objective,
            x,
            message=message,
            iterations=int(result["iterations"]),
        ), backend

    def _solve(self, problem: SparseLPProblem) -> Solution:
        prepared = self._prepare(problem)
        if not prepared.rows:
            if any(value > self.eps for value in prepared.c_max):
                return Solution(Status.UNBOUNDED, message="objective is unbounded")
            original = [offset for _, _, offset in prepared.reconstruct]
            return Solution(
                Status.OPTIMAL,
                sum(a * b for a, b in zip(problem.c, original, strict=True)),
                original,
            )

        tableau = self._build_tableau(prepared)
        phase_one = self._run_simplex(tableau, blocked_cols=tableau.artificial)
        if phase_one[0] == Status.ITERATION_LIMIT:
            return Solution(Status.ITERATION_LIMIT, message="phase I hit the iteration limit")
        if tableau.rows[-1].get(tableau.width, 0.0) > self.eps:
            return Solution(Status.INFEASIBLE, message="phase I found no feasible basis")

        self._remove_artificial_columns(tableau)
        self._set_objective(tableau, prepared.c_max)
        phase_two = self._run_simplex(tableau, blocked_cols=set())
        iterations = phase_one[1] + phase_two[1]
        if phase_two[0] != Status.OPTIMAL:
            return Solution(phase_two[0], message=phase_two[2], iterations=iterations)

        transformed = [0.0] * len(prepared.c_max)
        for row_index, basic_col in enumerate(tableau.basis):
            if basic_col < len(transformed):
                transformed[basic_col] = tableau.rows[row_index].get(tableau.width, 0.0)
        original = [
            (transformed[pos] if pos is not None else 0.0)
            - (transformed[neg] if neg is not None else 0.0)
            + offset
            for pos, neg, offset in prepared.reconstruct
        ]
        objective = sum(a * b for a, b in zip(problem.c, original, strict=True))
        return Solution(
            Status.OPTIMAL,
            objective,
            original,
            message="optimal solution found",
            iterations=iterations,
        )

    def _prepare(self, problem: SparseLPProblem) -> _PreparedSparse:
        source_rows: list[dict[int, float]] = []
        senses: list[SparseSense] = []
        rhs: list[float] = []
        if problem.A_eq is not None:
            eq_rows = _csr_rows(problem.A_eq)
            source_rows.extend(eq_rows)
            senses.extend(["="] * len(eq_rows))
            rhs.extend([float(value) for value in problem.b_eq or []])
        if problem.G_ub is not None:
            ub_rows = _csr_rows(problem.G_ub)
            source_rows.extend(ub_rows)
            senses.extend(["<="] * len(ub_rows))
            rhs.extend([float(value) for value in problem.h_ub or []])

        objective = [float(value) for value in problem.c]
        c_max = [-value for value in objective] if problem.objective == "min" else objective
        bounds = problem.bounds or [(0.0, None) for _ in c_max]
        offsets = [0.0] * len(c_max)
        transformed_c: list[float] = []
        reconstruct: list[tuple[int | None, int | None, float]] = []
        transformed_rows: list[dict[int, float]] = [dict() for _ in source_rows]

        for var_index, ((lower, upper), coefficient) in enumerate(zip(bounds, c_max, strict=True)):
            lower_value = 0.0 if lower is None else float(lower)
            upper_value = None if upper is None else float(upper)
            if upper_value is not None and upper_value < lower_value - self.eps:
                msg = f"upper bound is lower than lower bound for variable {var_index}"
                raise ValueError(msg)
            if lower is None:
                pos = len(transformed_c)
                transformed_c.append(coefficient)
                neg = len(transformed_c)
                transformed_c.append(-coefficient)
                reconstruct.append((pos, neg, 0.0))
                for row_index, source in enumerate(source_rows):
                    value = source.get(var_index)
                    if value is not None:
                        transformed_rows[row_index][pos] = value
                        transformed_rows[row_index][neg] = -value
            else:
                pos = len(transformed_c)
                transformed_c.append(coefficient)
                reconstruct.append((pos, None, lower_value))
                offsets[var_index] = lower_value
                for row_index, source in enumerate(source_rows):
                    value = source.get(var_index)
                    if value is not None:
                        transformed_rows[row_index][pos] = value
                        rhs[row_index] -= value * lower_value
            if upper_value is not None:
                bound_row: dict[int, float] = {}
                last = reconstruct[-1]
                if last[0] is not None:
                    bound_row[last[0]] = 1.0
                if last[1] is not None:
                    bound_row[last[1]] = -1.0
                transformed_rows.append(bound_row)
                senses.append("<=")
                rhs.append(upper_value - lower_value)
        return _PreparedSparse(transformed_c, transformed_rows, senses, rhs, reconstruct)

    def _build_tableau(self, prepared: _PreparedSparse) -> _SparseTableau:
        rows: list[dict[int, float]] = []
        rhs_values: list[float] = []
        basis: list[int] = []
        artificial: set[int] = set()
        width = len(prepared.c_max)
        for source, sense, raw_rhs in zip(
            prepared.rows, prepared.senses, prepared.rhs, strict=True
        ):
            row = _clean_row(dict(source), self.eps)
            rhs = raw_rhs
            if rhs < -self.eps:
                row = {col: -value for col, value in row.items()}
                rhs = -rhs
                sense = ">=" if sense == "<=" else "<=" if sense == ">=" else "="
            if sense == "<=":
                row[width] = 1.0
                basis.append(width)
                width += 1
            elif sense == ">=":
                row[width] = -1.0
                width += 1
                row[width] = 1.0
                artificial.add(width)
                basis.append(width)
                width += 1
            else:
                row[width] = 1.0
                artificial.add(width)
                basis.append(width)
                width += 1
            rows.append(row)
            rhs_values.append(rhs)

        rhs_col = width
        for row, rhs in zip(rows, rhs_values, strict=True):
            row[rhs_col] = rhs
        objective: dict[int, float] = {col: -1.0 for col in artificial}
        objective[rhs_col] = 0.0
        rows.append(objective)
        tableau = _SparseTableau(rows, basis, artificial, rhs_col, len(prepared.c_max))
        for row_index, basic_col in enumerate(basis):
            if basic_col in artificial:
                self._add_scaled(tableau.rows[-1], tableau.rows[row_index], 1.0)
        return tableau

    def _run_simplex(
        self, tableau: _SparseTableau, *, blocked_cols: set[int]
    ) -> tuple[Status, int, str]:
        iterations = 0
        while iterations < self.max_iterations:
            entering = None
            entering_value = self.eps
            for col, value in tableau.rows[-1].items():
                if col == tableau.width or col in blocked_cols:
                    continue
                if value > entering_value:
                    entering = col
                    entering_value = value
            if entering is None:
                return Status.OPTIMAL, iterations, "optimal"
            ratios: list[tuple[float, int]] = []
            for row_index, row in enumerate(tableau.rows[:-1]):
                coefficient = row.get(entering, 0.0)
                if coefficient > self.eps:
                    ratios.append((row.get(tableau.width, 0.0) / coefficient, row_index))
            if not ratios:
                return Status.UNBOUNDED, iterations, "objective is unbounded"
            leaving_row = min(ratios, key=lambda item: (item[0], tableau.basis[item[1]]))[1]
            self._pivot(tableau, leaving_row, entering)
            tableau.basis[leaving_row] = entering
            iterations += 1
        return Status.ITERATION_LIMIT, iterations, "sparse simplex hit the iteration limit"

    def _pivot(self, tableau: _SparseTableau, pivot_row: int, pivot_col: int) -> None:
        row = tableau.rows[pivot_row]
        pivot_value = row.get(pivot_col, 0.0)
        if abs(pivot_value) <= self.eps:
            msg = "pivot value is too close to zero"
            raise ZeroDivisionError(msg)
        tableau.rows[pivot_row] = {col: value / pivot_value for col, value in row.items()}
        pivot = tableau.rows[pivot_row]
        for row_index, target in enumerate(tableau.rows):
            if row_index == pivot_row:
                continue
            factor = target.get(pivot_col, 0.0)
            if abs(factor) <= self.eps:
                target.pop(pivot_col, None)
                continue
            self._add_scaled(target, pivot, -factor)

    def _add_scaled(self, target: dict[int, float], source: dict[int, float], scale: float) -> None:
        for col, value in source.items():
            updated = target.get(col, 0.0) + scale * value
            if abs(updated) <= self.eps:
                target.pop(col, None)
            else:
                target[col] = updated

    def _remove_artificial_columns(self, tableau: _SparseTableau) -> None:
        drop_rows: set[int] = set()
        for row_index, basic_col in enumerate(list(tableau.basis)):
            if basic_col not in tableau.artificial:
                continue
            replacement = None
            for col, value in tableau.rows[row_index].items():
                if col != tableau.width and col not in tableau.artificial and abs(value) > self.eps:
                    replacement = col
                    break
            if replacement is None:
                drop_rows.add(row_index)
            else:
                self._pivot(tableau, row_index, replacement)
                tableau.basis[row_index] = replacement
        if drop_rows:
            tableau.rows = [
                row for row_index, row in enumerate(tableau.rows) if row_index not in drop_rows
            ]
            tableau.basis = [
                col for row_index, col in enumerate(tableau.basis) if row_index not in drop_rows
            ]
        for row in tableau.rows:
            for col in tableau.artificial:
                row.pop(col, None)
        tableau.artificial = set()

    def _set_objective(self, tableau: _SparseTableau, objective: list[float]) -> None:
        row = {col: value for col, value in enumerate(objective) if abs(value) > self.eps}
        row[tableau.width] = 0.0
        tableau.rows[-1] = row
        for row_index, basic_col in enumerate(tableau.basis):
            coefficient = tableau.rows[-1].get(basic_col, 0.0)
            if abs(coefficient) > self.eps:
                self._add_scaled(tableau.rows[-1], tableau.rows[row_index], -coefficient)


def csr_matrix(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
) -> Any:
    if CSRMatrix is None:
        msg = "linprogx._csparse extension is not available"
        raise RuntimeError(msg)
    return CSRMatrix(rows, cols, indptr, indices, data)


def from_scipy_sparse(matrix: Any) -> Any:
    csr = matrix.tocsr()
    return csr_matrix(
        int(csr.shape[0]),
        int(csr.shape[1]),
        [int(value) for value in csr.indptr.tolist()],
        [int(value) for value in csr.indices.tolist()],
        [float(value) for value in csr.data.tolist()],
    )


def solve_sparse(problem: SparseLPProblem) -> SparseSolveResult:
    return SparseSolver().solve(problem)


def solve_sparse_canonical(
    c: list[float],
    A: Any,
    b: list[float],
    G: Any | None = None,
    h: list[float] | None = None,
    *,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> SparseSolveResult:
    return solve_sparse(
        SparseLPProblem(c, A, b, G, h, "min", bounds or [(None, None)] * len(c)),
    )


def _csr_rows(matrix: Any) -> list[dict[int, float]]:
    indptr, indices, data = matrix.to_components()
    rows, _ = matrix.shape
    result: list[dict[int, float]] = []
    for row_index in range(rows):
        row: dict[int, float] = {}
        for offset in range(indptr[row_index], indptr[row_index + 1]):
            row[int(indices[offset])] = float(data[offset])
        result.append(row)
    return result


def _clean_row(row: dict[int, float], eps: float) -> dict[int, float]:
    return {col: value for col, value in row.items() if abs(value) > eps}
