from __future__ import annotations

import json

import pytest

from linprogx import Constraint, LPProblem, Model, Solver, Status, solve, solve_canonical
from linprogx._fast import dot
from linprogx.cli import main


def assert_close_list(actual: list[float], expected: list[float], tol: float = 1e-7) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left == pytest.approx(right, abs=tol)


def test_solves_canonical_max_problem() -> None:
    result = solve(
        [3, 2],
        [[1, 1], [1, 0], [0, 1]],
        [4, 2, 3],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(10)
    assert_close_list(result.x, [2, 2])
    assert_close_list(result.slacks, [0, 0, 1])


def test_solves_minimization_with_greater_equal_constraint() -> None:
    result = solve(
        [1, 1],
        [[1, 2], [4, 2]],
        [4, 12],
        senses=[">=", ">="],
        objective="min",
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(10 / 3)
    assert_close_list(result.x, [8 / 3, 2 / 3])


def test_equality_constraint() -> None:
    result = solve(
        [5, 1],
        [[1, 1], [1, 0]],
        [5, 2],
        senses=["=", "<="],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(13)
    assert_close_list(result.x, [2, 3])


def test_solve_canonical_minimization_form() -> None:
    result = solve_canonical(
        c=[1, 2],
        A=[[1, 1]],
        b=[3],
        G=[[-1, 0], [0, -1], [1, 0]],
        h=[0, 0, 2],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(4)
    assert_close_list(result.x, [2, 1])


def test_solve_canonical_defaults_to_free_variables() -> None:
    result = solve_canonical(
        c=[1, 0],
        A=[[1, -1]],
        b=[-2],
        G=[[0, 1], [0, -1]],
        h=[1, -1],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(-1)
    assert_close_list(result.x, [-1, 1])


def test_solve_canonical_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="G and h"):
        solve_canonical([1], [], [], [[1]], [])


def test_bounds_are_respected() -> None:
    result = solve(
        [1, 1],
        [[1, 1]],
        [10],
        bounds=[(2, 4), (3, None)],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(10)
    assert result.x[0] == pytest.approx(4)
    assert result.x[1] == pytest.approx(6)


def test_free_variable_support() -> None:
    result = solve(
        [1, -1],
        [[1, -1], [1, 0]],
        [3, 4],
        bounds=[(None, None), (0, None)],
    )

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(3)
    assert result.x[0] - result.x[1] == pytest.approx(3)


def test_detects_infeasible_problem() -> None:
    result = solve(
        [1],
        [[1], [1]],
        [1, 2],
        senses=["<=", ">="],
    )

    assert result.status == Status.INFEASIBLE


def test_detects_unbounded_problem() -> None:
    result = solve([1], [], [])

    assert result.status == Status.UNBOUNDED


def test_builder_interface() -> None:
    model = Model("factory")
    chairs = model.variable("chairs", upper=3)
    tables = model.variable("tables")
    model.maximize({chairs: 5, tables: 4})
    model.add_constraint({chairs: 2, tables: 1}, "<=", 8, name="wood")
    model.add_constraint({chairs: 1, tables: 2}, "<=", 8, name="labor")

    result = model.solve()

    assert result.status == Status.OPTIMAL
    assert result.objective_value == pytest.approx(24)
    assert_close_list(result.x, [8 / 3, 8 / 3])


def test_problem_validation() -> None:
    with pytest.raises(ValueError, match="constraint width"):
        LPProblem([1, 2], [Constraint([1], "<=", 1)])


def test_dot_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        dot([1], [1, 2])


def test_cli_outputs_json(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "problem.json"
    path.write_text(
        json.dumps(
            {
                "c": [3, 2],
                "A": [[1, 1], [1, 0], [0, 1]],
                "b": [4, 2, 3],
            }
        )
    )

    code = main([str(path), "--pretty"])
    out = json.loads(capsys.readouterr().out)

    assert code == 0
    assert out["status"] == "optimal"
    assert out["objective_value"] == pytest.approx(10)


def test_iteration_limit() -> None:
    solver = Solver(max_iterations=0)
    result = solver.solve(LPProblem([1], [Constraint([1], "<=", 1)]))

    assert result.status == Status.ITERATION_LIMIT
