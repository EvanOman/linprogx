from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

Sense = Literal["<=", ">=", "="]
ObjectiveSense = Literal["min", "max"]


class Status(StrEnum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ITERATION_LIMIT = "iteration_limit"


@dataclass(frozen=True)
class Constraint:
    coefficients: list[float]
    sense: Sense
    rhs: float
    name: str | None = None


@dataclass(frozen=True)
class LPProblem:
    c: list[float]
    constraints: list[Constraint]
    objective: ObjectiveSense = "max"
    bounds: list[tuple[float | None, float | None]] | None = None
    name: str = "lp"

    def __post_init__(self) -> None:
        width = len(self.c)
        if width == 0:
            msg = "objective must contain at least one coefficient"
            raise ValueError(msg)
        for item in self.constraints:
            if len(item.coefficients) != width:
                msg = "constraint width must match objective width"
                raise ValueError(msg)
        if self.bounds is not None and len(self.bounds) != width:
            msg = "bounds width must match objective width"
            raise ValueError(msg)


@dataclass(frozen=True)
class Sensitivity:
    reduced_costs: list[float]
    shadow_prices: list[float]
    basis: list[str]


@dataclass(frozen=True)
class Solution:
    status: Status
    objective_value: float | None = None
    x: list[float] = field(default_factory=list)
    slacks: list[float] = field(default_factory=list)
    message: str = ""
    iterations: int = 0
    sensitivity: Sensitivity | None = None

    @property
    def success(self) -> bool:
        return self.status == Status.OPTIMAL
