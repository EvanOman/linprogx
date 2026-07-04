from __future__ import annotations

import clarabel  # noqa: F401
import pytest
import scipy  # noqa: F401

from linprogx import Solver
from linprogx.compare import compare_with_clarabel, compare_with_scipy
from linprogx.samples import SAMPLES, STANDARD_BENCHMARKS
from linprogx.types import Status

ALL_SAMPLES = (*SAMPLES, *STANDARD_BENCHMARKS)


@pytest.mark.parametrize("sample", ALL_SAMPLES, ids=[sample.name for sample in ALL_SAMPLES])
def test_sample_matches_expected_and_scipy(sample) -> None:  # type: ignore[no-untyped-def]
    result = Solver().solve(sample.problem)
    scipy_comparison = compare_with_scipy(sample.problem)
    clarabel_comparison = compare_with_clarabel(sample.problem)

    assert result.status.value == sample.expected_status
    if sample.expected_objective is not None:
        assert result.objective_value == pytest.approx(sample.expected_objective, abs=1e-7)

    assert scipy_comparison.linprogx_status == sample.expected_status
    assert scipy_comparison.solver_status == sample.expected_status
    if scipy_comparison.solver_status == Status.OPTIMAL.value:
        assert scipy_comparison.objective_delta is not None
        assert scipy_comparison.objective_delta <= 1e-7

    assert clarabel_comparison.linprogx_status == sample.expected_status
    assert clarabel_comparison.solver_status == sample.expected_status
    if clarabel_comparison.solver_status == Status.OPTIMAL.value:
        assert clarabel_comparison.objective_delta is not None
        scale = max(1.0, abs(clarabel_comparison.linprogx_objective or 0.0))
        assert clarabel_comparison.objective_delta <= max(1e-5, 1e-9 * scale)
