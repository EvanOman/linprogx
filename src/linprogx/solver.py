from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from linprogx._fast import dot, pivot
from linprogx.types import Constraint, LPProblem, Sensitivity, Solution, Status


@dataclass
class _Prepared:
    c_max: list[float]
    constraints: list[Constraint]
    names: list[str]
    reconstruct: list[tuple[int | None, int | None, float]]
    constant: float


@dataclass
class _Tableau:
    rows: list[list[float]]
    basis: list[int]
    names: list[str]
    artificial: set[int]
    original_width: int
    constraint_count: int


class Solver:
    """Two-phase primal simplex solver.

    The implementation favors clarity and inspectability. The Gauss-Jordan
    pivot and vector dot product are dispatched to a tiny C extension when it is
    installed, and fall back to Python otherwise.
    """

    def __init__(self, *, eps: float = 1e-9, max_iterations: int = 10_000) -> None:
        self.eps = eps
        self.max_iterations = max_iterations

    def solve(self, problem: LPProblem) -> Solution:
        prepared = self._prepare(problem)
        if not prepared.constraints:
            if any(value > self.eps for value in prepared.c_max):
                return Solution(Status.UNBOUNDED, message="objective is unbounded")
            original = [offset for _, _, offset in prepared.reconstruct]
            return Solution(
                Status.OPTIMAL,
                objective_value=dot(problem.c, original),
                x=original,
                message="optimal solution found",
            )
        tableau = self._build_tableau(prepared)
        iterations = 0

        phase_one = self._run_simplex(tableau.rows, tableau.basis, "min", tableau.artificial)
        iterations += phase_one[1]
        if phase_one[0] == Status.ITERATION_LIMIT:
            return Solution(Status.ITERATION_LIMIT, message="phase I hit the iteration limit")
        if tableau.rows[-1][-1] > self.eps:
            return Solution(Status.INFEASIBLE, message="phase I found no feasible basis")

        self._remove_artificial_columns(tableau)
        self._set_objective(tableau.rows, tableau.basis, prepared.c_max, "max")

        phase_two = self._run_simplex(tableau.rows, tableau.basis, "max", set())
        iterations += phase_two[1]
        if phase_two[0] != Status.OPTIMAL:
            return Solution(phase_two[0], message=phase_two[2], iterations=iterations)

        transformed = [0.0] * prepared.c_max.__len__()
        for row_index, basic_col in enumerate(tableau.basis):
            if basic_col < len(transformed):
                transformed[basic_col] = tableau.rows[row_index][-1]

        original = [
            (transformed[pos] if pos is not None else 0.0)
            - (transformed[neg] if neg is not None else 0.0)
            + offset
            for pos, neg, offset in prepared.reconstruct
        ]
        objective = dot(problem.c, original)
        if problem.objective == "min":
            objective = dot(problem.c, original)

        slacks = self._slacks(problem, original)
        sensitivity = self._sensitivity(tableau, prepared)
        return Solution(
            status=Status.OPTIMAL,
            objective_value=objective,
            x=original,
            slacks=slacks,
            message="optimal solution found",
            iterations=iterations,
            sensitivity=sensitivity,
        )

    def _prepare(self, problem: LPProblem) -> _Prepared:
        c = [float(value) for value in problem.c]
        if problem.objective == "min":
            c = [-value for value in c]

        constraints = [
            Constraint(
                [float(value) for value in item.coefficients],
                item.sense,
                float(item.rhs),
                item.name,
            )
            for item in problem.constraints
        ]
        bounds = problem.bounds or [(0.0, None) for _ in c]
        transformed_c: list[float] = []
        transformed_names: list[str] = []
        reconstruct: list[tuple[int | None, int | None, float]] = []
        offsets = [0.0 for _ in c]
        constant = 0.0

        rows: list[list[float]] = []
        for item in constraints:
            rows.append(list(item.coefficients))

        for var_index, ((lower, upper), coefficient) in enumerate(zip(bounds, c, strict=True)):
            lower_value = 0.0 if lower is None else float(lower)
            upper_value = None if upper is None else float(upper)
            if upper_value is not None and upper_value < lower_value - self.eps:
                msg = f"upper bound is lower than lower bound for variable {var_index}"
                raise ValueError(msg)

            if lower is None:
                pos = len(transformed_c)
                transformed_c.append(coefficient)
                transformed_names.append(f"x{var_index}_pos")
                neg = len(transformed_c)
                transformed_c.append(-coefficient)
                transformed_names.append(f"x{var_index}_neg")
                for row in rows:
                    row.append(row[var_index])
                    row.append(-row[var_index])
                reconstruct.append((pos, neg, 0.0))
            else:
                pos = len(transformed_c)
                transformed_c.append(coefficient)
                transformed_names.append(f"x{var_index}")
                for row in rows:
                    row.append(row[var_index])
                reconstruct.append((pos, None, lower_value))
                offsets[var_index] = lower_value
                constant += coefficient * lower_value

            if upper_value is not None:
                width = len(transformed_c)
                coeffs = [0.0] * width
                coeffs[reconstruct[-1][0] or 0] = 1.0
                if reconstruct[-1][1] is not None:
                    coeffs[reconstruct[-1][1]] = -1.0
                constraints.append(
                    Constraint(coeffs, "<=", upper_value - lower_value, f"upper_x{var_index}")
                )

        normalized: list[Constraint] = []
        old_width = len(c)
        for source, row in zip(problem.constraints, rows, strict=True):
            rhs = float(source.rhs) - dot(row[:old_width], offsets)
            normalized.append(Constraint(row[old_width:], source.sense, rhs, source.name))
        for item in constraints[len(problem.constraints) :]:
            missing = len(transformed_c) - len(item.coefficients)
            normalized.append(
                Constraint(item.coefficients + [0.0] * missing, item.sense, item.rhs, item.name)
            )

        return _Prepared(transformed_c, normalized, transformed_names, reconstruct, constant)

    def _build_tableau(self, prepared: _Prepared) -> _Tableau:
        rows: list[list[float]] = []
        basis: list[int] = []
        names = list(prepared.names)
        artificial: set[int] = set()
        width = len(prepared.c_max)

        for index, raw in enumerate(prepared.constraints):
            coeffs = list(raw.coefficients)
            rhs = float(raw.rhs)
            sense = raw.sense
            if rhs < -self.eps:
                coeffs = [-value for value in coeffs]
                rhs = -rhs
                sense = ">=" if sense == "<=" else "<=" if sense == ">=" else "="

            row = coeffs + [0.0] * (len(names) - width)
            if sense == "<=":
                row.append(1.0)
                names.append(raw.name or f"s{index}")
                basis.append(len(names) - 1)
            elif sense == ">=":
                row.append(-1.0)
                names.append(raw.name or f"surplus{index}")
                row.append(1.0)
                names.append(f"artificial{index}")
                artificial.add(len(names) - 1)
                basis.append(len(names) - 1)
            else:
                row.append(1.0)
                names.append(f"artificial{index}")
                artificial.add(len(names) - 1)
                basis.append(len(names) - 1)
            rows.append(row + [rhs])

        total_width = len(names)
        for row in rows:
            missing = total_width + 1 - len(row)
            if missing > 0:
                row[-1:-1] = [0.0] * missing

        objective = [0.0] * total_width + [0.0]
        for col in artificial:
            objective[col] = -1.0
        rows.append(objective)
        for row_index, basic_col in enumerate(basis):
            if basic_col in artificial:
                rows[-1] = [a + b for a, b in zip(rows[-1], rows[row_index], strict=True)]

        return _Tableau(
            rows, basis, names, artificial, len(prepared.c_max), len(prepared.constraints)
        )

    def _run_simplex(
        self,
        rows: list[list[float]],
        basis: list[int],
        sense: Literal["min", "max"],
        blocked_cols: set[int],
    ) -> tuple[Status, int, str]:
        iterations = 0
        while iterations < self.max_iterations:
            obj = rows[-1][:-1]
            candidates = [
                (col, value)
                for col, value in enumerate(obj)
                if col not in blocked_cols and value > self.eps
            ]
            if not candidates:
                return Status.OPTIMAL, iterations, "optimal"
            entering = max(candidates, key=lambda item: (item[1], -item[0]))[0]
            ratios: list[tuple[float, int]] = []
            for row_index, row in enumerate(rows[:-1]):
                coefficient = row[entering]
                if coefficient > self.eps:
                    ratios.append((row[-1] / coefficient, row_index))
            if not ratios:
                return Status.UNBOUNDED, iterations, "objective is unbounded"
            leaving_row = min(ratios, key=lambda item: (item[0], basis[item[1]]))[1]
            pivot(rows, leaving_row, entering, self.eps)
            basis[leaving_row] = entering
            iterations += 1
        return Status.ITERATION_LIMIT, iterations, f"{sense} simplex hit the iteration limit"

    def _set_objective(
        self,
        rows: list[list[float]],
        basis: list[int],
        objective: list[float],
        sense: Literal["max"],
    ) -> None:
        del sense
        width = len(rows[0]) - 1
        rows[-1] = [0.0] * (width + 1)
        for col, coefficient in enumerate(objective):
            rows[-1][col] = coefficient
        for row_index, basic_col in enumerate(basis):
            coefficient = rows[-1][basic_col]
            if abs(coefficient) <= self.eps:
                continue
            rows[-1] = [
                current - coefficient * value
                for current, value in zip(rows[-1], rows[row_index], strict=True)
            ]

    def _remove_artificial_columns(self, tableau: _Tableau) -> None:
        drop_rows: set[int] = set()
        for row_index, basic_col in enumerate(list(tableau.basis)):
            if basic_col not in tableau.artificial:
                continue
            replacement = None
            for col in range(len(tableau.names)):
                if col not in tableau.artificial and abs(tableau.rows[row_index][col]) > self.eps:
                    replacement = col
                    break
            if replacement is not None:
                pivot(tableau.rows, row_index, replacement, self.eps)
                tableau.basis[row_index] = replacement
            else:
                drop_rows.add(row_index)

        keep = [col for col in range(len(tableau.names)) if col not in tableau.artificial]
        tableau.rows = [
            [row[col] for col in keep] + [row[-1]]
            for row_index, row in enumerate(tableau.rows)
            if row_index not in drop_rows
        ]
        remap = {old: new for new, old in enumerate(keep)}
        tableau.basis = [
            remap[col] for row_index, col in enumerate(tableau.basis) if row_index not in drop_rows
        ]
        tableau.names = [tableau.names[col] for col in keep]
        tableau.artificial = set()

    def _slacks(self, problem: LPProblem, x: list[float]) -> list[float]:
        slacks: list[float] = []
        for item in problem.constraints:
            activity = dot(item.coefficients, x)
            if item.sense == "<=":
                slacks.append(item.rhs - activity)
            elif item.sense == ">=":
                slacks.append(activity - item.rhs)
            else:
                slacks.append(abs(activity - item.rhs))
        return slacks

    def _sensitivity(self, tableau: _Tableau, prepared: _Prepared) -> Sensitivity:
        reduced = [0.0] * tableau.original_width
        for col in range(min(tableau.original_width, len(tableau.rows[-1]) - 1)):
            reduced[col] = -tableau.rows[-1][col]
        shadow = []
        for col in range(tableau.original_width, len(tableau.rows[-1]) - 1):
            shadow.append(-tableau.rows[-1][col])
        while len(shadow) < tableau.constraint_count:
            shadow.append(0.0)
        return Sensitivity(
            reduced_costs=reduced,
            shadow_prices=shadow[: tableau.constraint_count],
            basis=[tableau.names[col] for col in tableau.basis],
        )


def solve(
    c: list[float],
    A: list[list[float]],
    b: list[float],
    *,
    senses: list[Literal["<=", ">=", "="]] | None = None,
    objective: Literal["min", "max"] = "max",
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> Solution:
    resolved_senses: list[Literal["<=", ">=", "="]]
    if senses is None:
        resolved_senses = ["<=" for _ in b]
    else:
        resolved_senses = senses
    if len(A) != len(b) or len(resolved_senses) != len(b):
        msg = "A, b, and senses must describe the same number of constraints"
        raise ValueError(msg)
    if any(not isfinite(value) for value in c):
        msg = "objective coefficients must be finite"
        raise ValueError(msg)
    constraints = [
        Constraint(A[index], resolved_senses[index], b[index]) for index in range(len(b))
    ]
    return Solver().solve(LPProblem(c, constraints, objective, bounds))
