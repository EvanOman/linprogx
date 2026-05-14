from __future__ import annotations

import pytest

from linprogx import Solver
from linprogx.compare import assert_matches_clarabel, assert_matches_scipy
from linprogx.samples import SAMPLES

scipy = pytest.importorskip("scipy")
del scipy
clarabel = pytest.importorskip("clarabel")
del clarabel


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample.name for sample in SAMPLES])
def test_sample_matches_expected_and_scipy(sample) -> None:  # type: ignore[no-untyped-def]
    result = Solver().solve(sample.problem)

    assert result.status.value == sample.expected_status
    if sample.expected_objective is not None:
        assert result.objective_value == pytest.approx(sample.expected_objective, abs=1e-7)
    assert_matches_scipy(sample.problem)
    assert_matches_clarabel(sample.problem)
