from __future__ import annotations

from dataclasses import dataclass

from linprogx.types import Constraint, LPProblem


@dataclass(frozen=True)
class SampleProblem:
    name: str
    problem: LPProblem
    expected_status: str
    expected_objective: float | None = None


SAMPLES: tuple[SampleProblem, ...] = (
    SampleProblem(
        "product_mix",
        LPProblem(
            [3, 2],
            [Constraint([1, 1], "<=", 4), Constraint([1, 0], "<=", 2), Constraint([0, 1], "<=", 3)],
            "max",
            name="product_mix",
        ),
        "optimal",
        10.0,
    ),
    SampleProblem(
        "diet_minimum",
        LPProblem(
            [0.5, 0.8, 0.2],
            [
                Constraint([400, 200, 150], ">=", 500),
                Constraint([3, 12, 2], ">=", 12),
                Constraint([2, 1, 0], ">=", 3),
            ],
            "min",
            name="diet_minimum",
        ),
        "optimal",
        1.1428571428571428,
    ),
    SampleProblem(
        "transportation_2x3",
        LPProblem(
            [8, 6, 10, 9, 12, 13],
            [
                Constraint([1, 1, 1, 0, 0, 0], "=", 20),
                Constraint([0, 0, 0, 1, 1, 1], "=", 30),
                Constraint([1, 0, 0, 1, 0, 0], "=", 10),
                Constraint([0, 1, 0, 0, 1, 0], "=", 25),
                Constraint([0, 0, 1, 0, 0, 1], "=", 15),
            ],
            "min",
            name="transportation_2x3",
        ),
        "optimal",
        465.0,
    ),
    SampleProblem(
        "blending",
        LPProblem(
            [4, 3],
            [
                Constraint([1, 1], "=", 100),
                Constraint([0.10, 0.30], ">=", 20),
                Constraint([0.05, 0.02], "<=", 4),
            ],
            "min",
            name="blending",
        ),
        "optimal",
        300.0,
    ),
    SampleProblem(
        "assignment_relaxation",
        LPProblem(
            [14, 5, 8, 7, 2, 12, 6, 5, 3],
            [
                Constraint([1, 1, 1, 0, 0, 0, 0, 0, 0], "=", 1),
                Constraint([0, 0, 0, 1, 1, 1, 0, 0, 0], "=", 1),
                Constraint([0, 0, 0, 0, 0, 0, 1, 1, 1], "=", 1),
                Constraint([1, 0, 0, 1, 0, 0, 1, 0, 0], "=", 1),
                Constraint([0, 1, 0, 0, 1, 0, 0, 1, 0], "=", 1),
                Constraint([0, 0, 1, 0, 0, 1, 0, 0, 1], "=", 1),
            ],
            "min",
            bounds=[(0, 1)] * 9,
            name="assignment_relaxation",
        ),
        "optimal",
        15.0,
    ),
    SampleProblem(
        "free_variable_balance",
        LPProblem(
            [1, -1],
            [Constraint([1, -1], "<=", 3), Constraint([1, 0], "<=", 4)],
            "max",
            bounds=[(None, None), (0, None)],
            name="free_variable_balance",
        ),
        "optimal",
        3.0,
    ),
    SampleProblem(
        "portfolio_allocation",
        LPProblem(
            [0.12, 0.08, 0.10],
            [
                Constraint([1, 1, 1], "=", 1),
                Constraint([1, 0, 0], "<=", 0.5),
                Constraint([0, 1, 0], "<=", 0.7),
            ],
            "max",
            bounds=[(0, 1), (0, 1), (0, 1)],
            name="portfolio_allocation",
        ),
        "optimal",
        0.11,
    ),
    SampleProblem(
        "ad_campaign",
        LPProblem(
            [5, 3, 4],
            [
                Constraint([2, 1, 3], "<=", 100),
                Constraint([1, 1, 1], "<=", 60),
                Constraint([0, 1, 0], ">=", 10),
            ],
            "max",
            name="ad_campaign",
        ),
        "optimal",
        260.0,
    ),
    SampleProblem(
        "workforce_cover",
        LPProblem(
            [160, 180, 170],
            [
                Constraint([1, 0, 1], ">=", 7),
                Constraint([1, 1, 0], ">=", 8),
                Constraint([0, 1, 1], ">=", 6),
            ],
            "min",
            name="workforce_cover",
        ),
        "optimal",
        1775.0,
    ),
    SampleProblem(
        "knapsack_relaxation",
        LPProblem(
            [10, 7, 12, 8],
            [Constraint([4, 3, 5, 2], "<=", 9)],
            "max",
            bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            name="knapsack_relaxation",
        ),
        "optimal",
        25.2,
    ),
    SampleProblem(
        "degenerate_multiple_optima",
        LPProblem(
            [1, 1],
            [Constraint([1, 0], "<=", 1), Constraint([0, 1], "<=", 1), Constraint([1, 1], "<=", 1)],
            "max",
            name="degenerate_multiple_optima",
        ),
        "optimal",
        1.0,
    ),
    SampleProblem(
        "redundant_constraints",
        LPProblem(
            [2, 1],
            [
                Constraint([1, 1], "<=", 5),
                Constraint([2, 2], "<=", 10),
                Constraint([1, 0], "<=", 3),
            ],
            "max",
            name="redundant_constraints",
        ),
        "optimal",
        8.0,
    ),
    SampleProblem(
        "negative_rhs_normalization",
        LPProblem(
            [1, 1],
            [Constraint([-1, -1], ">=", -4), Constraint([1, 0], "<=", 3)],
            "max",
            name="negative_rhs_normalization",
        ),
        "optimal",
        4.0,
    ),
    SampleProblem(
        "lower_bound_shift",
        LPProblem(
            [1, 2],
            [Constraint([1, 1], ">=", 10)],
            "min",
            bounds=[(2, None), (3, None)],
            name="lower_bound_shift",
        ),
        "optimal",
        13.0,
    ),
    SampleProblem(
        "infeasible_window",
        LPProblem(
            [1],
            [Constraint([1], "<=", 1), Constraint([1], ">=", 2)],
            "max",
            name="infeasible_window",
        ),
        "infeasible",
    ),
    SampleProblem(
        "unbounded_ray",
        LPProblem([1], [], "max", name="unbounded_ray"),
        "unbounded",
    ),
)


def get_sample(name: str) -> SampleProblem:
    for sample in SAMPLES:
        if sample.name == name:
            return sample
    msg = f"unknown sample problem: {name}"
    raise KeyError(msg)
