from __future__ import annotations

import json
import runpy
import sys
from typing import Any

import pytest

import linprogx._fast as fast
import linprogx.compare as compare
from linprogx import Constraint, LPProblem, Model, Solver, Status, solve, solve_canonical
from linprogx.cli import _bounds, main
from linprogx.compare import Comparison
from linprogx.samples import SAMPLES, get_sample, klee_minty


def test_python_dot_fallback_computes_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fast, "_cfast", None)

    assert fast.dot([1.5, -2.0], [2.0, 4.0]) == pytest.approx(-5.0)
    with pytest.raises(ValueError, match="same length"):
        fast.dot([1.0], [1.0, 2.0])


def test_python_pivot_fallback_reduces_the_tableau(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fast, "_cfast", None)
    tableau = [[2.0, 1.0, 4.0], [4.0, 2.0, 8.0], [1e-12, 3.0, 6.0]]

    fast.pivot(tableau, 0, 0, 1e-9)

    assert tableau == [[1.0, 0.5, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 6.0]]


def test_python_pivot_fallback_rejects_a_zero_pivot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fast, "_cfast", None)

    with pytest.raises(ZeroDivisionError, match="too close to zero"):
        fast.pivot([[0.0, 1.0]], 0, 0, 1e-9)


def test_model_requires_an_objective_and_valid_coefficient_width() -> None:
    model = Model()
    model.variable()
    model.variable()

    with pytest.raises(ValueError, match="set an objective"):
        model.problem()
    with pytest.raises(ValueError, match="coefficient list"):
        model.minimize([1.0])

    model.minimize([2, 3])
    assert model.problem().c == [2.0, 3.0]
    assert model.problem().objective == "min"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not-a-list", "bounds must be a list"),
        ([[0, 1, 2]], "each bound must be"),
        ([1], "each bound must be"),
    ],
)
def test_cli_bounds_reject_malformed_values(raw: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _bounds(raw)


def test_cli_bounds_support_defaults_and_explicit_pairs() -> None:
    assert _bounds(None) is None
    assert _bounds([None, [1, 2], (None, None)]) == [
        (0.0, None),
        (1, 2),
        (None, None),
    ]


@pytest.mark.parametrize(
    "contents",
    [
        "{not JSON",
        json.dumps({"A": [], "b": []}),
        json.dumps({"c": [1], "A": [], "b": [], "bounds": "invalid"}),
    ],
)
def test_cli_reports_input_errors(
    tmp_path: Any, capsys: pytest.CaptureFixture[str], contents: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(contents)

    assert main([str(path)]) == 2
    assert capsys.readouterr().err.startswith("linprogx:")


def test_cli_returns_failure_for_a_valid_unbounded_problem(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "unbounded.json"
    path.write_text(json.dumps({"c": [1], "A": [], "b": []}))

    assert main([str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unbounded"


def test_cli_module_entry_point_exits_with_main_result(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "problem.json"
    path.write_text(json.dumps({"c": [-1], "A": [], "b": []}))
    monkeypatch.setattr(sys, "argv", ["linprogx", str(path)])
    monkeypatch.delitem(sys.modules, "linprogx.cli")

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("linprogx.cli", run_name="__main__")


def _comparison(
    *,
    linprogx_status: str = "optimal",
    solver_status: str = "optimal",
    delta: float | None = 0.0,
    own_objective: float | None = 1.0,
) -> Comparison:
    return Comparison(
        name="test",
        linprogx_status=linprogx_status,
        solver_status=solver_status,
        linprogx_objective=own_objective,
        solver_objective=own_objective,
        objective_delta=delta,
        linprogx_seconds=0.0,
        solver_seconds=0.0,
        solver_name="fake",
    )


def test_scipy_assertion_helper_accepts_optimal_and_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = LPProblem([1], [])
    monkeypatch.setattr(compare, "compare_with_scipy", lambda _problem: _comparison())
    compare.assert_matches_scipy(problem)

    monkeypatch.setattr(
        compare,
        "compare_with_scipy",
        lambda _problem: _comparison(
            linprogx_status="unbounded",
            solver_status=Status.ITERATION_LIMIT.value,
            delta=None,
        ),
    )
    compare.assert_matches_scipy(problem)


def test_scipy_assertion_helper_enforces_objective_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compare, "compare_with_scipy", lambda _problem: _comparison(delta=0.1))

    with pytest.raises(AssertionError):
        compare.assert_matches_scipy(LPProblem([1], []), objective_tol=0.01)


def test_clarabel_assertion_helper_accepts_relative_tolerance_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = LPProblem([1], [])
    monkeypatch.setattr(
        compare,
        "compare_with_clarabel",
        lambda _problem: _comparison(delta=5e-4, own_objective=1_000_000.0),
    )
    compare.assert_matches_clarabel(problem, objective_tol=1e-5, relative_tol=1e-9)

    monkeypatch.setattr(
        compare,
        "compare_with_clarabel",
        lambda _problem: _comparison(solver_status=Status.ITERATION_LIMIT.value, delta=None),
    )
    compare.assert_matches_clarabel(problem)


def test_clarabel_assertion_helper_enforces_status_and_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = LPProblem([1], [])
    monkeypatch.setattr(
        compare,
        "compare_with_clarabel",
        lambda _problem: _comparison(linprogx_status="unbounded"),
    )
    with pytest.raises(AssertionError):
        compare.assert_matches_clarabel(problem)

    monkeypatch.setattr(compare, "compare_with_clarabel", lambda _problem: _comparison(delta=1.0))
    with pytest.raises(AssertionError):
        compare.assert_matches_clarabel(problem)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MaxIterations", Status.ITERATION_LIMIT.value),
        ("MaxTime", Status.ITERATION_LIMIT.value),
        ("NumericalError", "clarabel_numericalerror"),
    ],
)
def test_clarabel_status_maps_remaining_results(raw: str, expected: str) -> None:
    assert compare._clarabel_status(raw) == expected


def test_sample_lookup_and_validation_paths() -> None:
    assert get_sample(SAMPLES[0].name) is SAMPLES[0]
    with pytest.raises(ValueError, match="at least 2"):
        klee_minty(1)
    with pytest.raises(KeyError, match="unknown sample"):
        get_sample("does-not-exist")


def test_problem_rejects_empty_objective_and_wrong_bound_width() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LPProblem([], [])
    with pytest.raises(ValueError, match="bounds width"):
        LPProblem([1, 2], [], bounds=[(0, None)])


def test_no_constraint_bounded_objective_returns_its_lower_bound() -> None:
    result = solve([-2], [], [], bounds=[(3, None)])

    assert result.status == Status.OPTIMAL
    assert result.x == [3.0]
    assert result.objective_value == pytest.approx(-6.0)


def test_invalid_variable_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="upper bound is lower"):
        solve([1], [], [], bounds=[(2, 1)])


def test_upper_bound_on_a_free_variable_is_respected() -> None:
    result = solve([1], [], [], bounds=[(None, 4)])

    assert result.status == Status.OPTIMAL
    assert result.x == pytest.approx([4.0])


def test_constrained_unbounded_problem_reaches_ratio_test() -> None:
    result = solve([1], [[1]], [0], senses=[">="])

    assert result.status == Status.UNBOUNDED
    assert result.message == "objective is unbounded"


def test_phase_two_nonoptimal_status_preserves_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = Solver()
    calls = 0

    def run_simplex(
        _rows: list[list[float]],
        _basis: list[int],
        _sense: str,
        _blocked: set[int],
    ) -> tuple[Status, int, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Status.OPTIMAL, 2, "optimal"
        return Status.ITERATION_LIMIT, 3, "deliberate phase II limit"

    monkeypatch.setattr(solver, "_run_simplex", run_simplex)
    result = solver.solve(LPProblem([1], [Constraint([1], "<=", 1)]))

    assert result.status == Status.ITERATION_LIMIT
    assert result.iterations == 5
    assert result.message == "deliberate phase II limit"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([1], [], [1]), "A, b, and senses"),
        (([float("inf")], [], []), "objective coefficients must be finite"),
    ],
)
def test_solve_validates_top_level_inputs(
    args: tuple[list[float], list[list[float]], list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solve(*args)


def test_solve_validates_explicit_sense_count() -> None:
    with pytest.raises(ValueError, match="A, b, and senses"):
        solve([1], [[1]], [1], senses=[])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (([1], [[1]], [], [], []), "A and b"),
        (([1], [], [], [[1, 2]], [1]), "constraint rows"),
    ],
)
def test_canonical_validates_remaining_dimensions(
    args: tuple[
        list[float],
        list[list[float]],
        list[float],
        list[list[float]],
        list[float],
    ],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_canonical(*args)
