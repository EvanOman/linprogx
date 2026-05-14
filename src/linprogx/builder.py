from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from linprogx.solver import Solver
from linprogx.types import Constraint, LPProblem, Solution


@dataclass(frozen=True)
class Variable:
    index: int
    name: str


class Model:
    def __init__(self, name: str = "lp") -> None:
        self.name = name
        self._variables: list[Variable] = []
        self._bounds: list[tuple[float | None, float | None]] = []
        self._constraints: list[Constraint] = []
        self._objective: list[float] | None = None
        self._sense: Literal["min", "max"] = "max"

    def variable(
        self,
        name: str | None = None,
        *,
        lower: float | None = 0.0,
        upper: float | None = None,
    ) -> Variable:
        variable = Variable(len(self._variables), name or f"x{len(self._variables)}")
        self._variables.append(variable)
        self._bounds.append((lower, upper))
        return variable

    def maximize(self, coefficients: dict[Variable, float] | list[float]) -> None:
        self._sense = "max"
        self._objective = self._expand(coefficients)

    def minimize(self, coefficients: dict[Variable, float] | list[float]) -> None:
        self._sense = "min"
        self._objective = self._expand(coefficients)

    def add_constraint(
        self,
        coefficients: dict[Variable, float] | list[float],
        sense: Literal["<=", ">=", "="],
        rhs: float,
        *,
        name: str | None = None,
    ) -> None:
        self._constraints.append(Constraint(self._expand(coefficients), sense, rhs, name))

    def problem(self) -> LPProblem:
        if self._objective is None:
            msg = "set an objective before solving"
            raise ValueError(msg)
        return LPProblem(
            self._objective,
            list(self._constraints),
            self._sense,
            list(self._bounds),
            self.name,
        )

    def solve(self, *, solver: Solver | None = None) -> Solution:
        return (solver or Solver()).solve(self.problem())

    def _expand(self, coefficients: dict[Variable, float] | list[float]) -> list[float]:
        if isinstance(coefficients, list):
            if len(coefficients) != len(self._variables):
                msg = "coefficient list must match variable count"
                raise ValueError(msg)
            return [float(value) for value in coefficients]
        row = [0.0 for _ in self._variables]
        for variable, value in coefficients.items():
            row[variable.index] = float(value)
        return row
