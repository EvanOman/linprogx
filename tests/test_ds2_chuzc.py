"""DS2 component A (CHUZC): the bound-flipping dual ratio test.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

These are characterization tests for the new, gated component in
``src/linprogx/_ds2_chuzc.c``.  They do not touch the shipped dual simplex,
whose invariance under the harvest instrumentation is proven separately by
the trace-hash oracle (see ``experiments/ds2_chuzc_2026_07_26.md``).

The real validation is
``experiments/ds2_chuzc_validate.py``, which replays hundreds of pivot rows
harvested from greenbea / degen2 / 25fv47.  What lives here is the small set
of hand-built cases that pin the mechanism: does the longest-step walk step
over a boxed breakpoint, does it refuse to step over a one-sided one, and do
the Harris controls reproduce the min-ratio choice.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

ds2 = importlib.import_module("linprogx._ds2_chuzc")

LO, HI, FREE, FIXED, BASIC = 0, 1, 2, 3, 4
INF = float("inf")


def make_case(cols, sigma=1, delta=1.0):
    """cols: list of (alpha, r, bound_status, lo, hi)."""
    n = len(cols)
    alpha = np.array([c[0] for c in cols], np.float64)
    r = np.array([c[1] for c in cols], np.float64)
    bs = np.array([c[2] for c in cols], np.int8)
    lo = np.array([c[3] for c in cols], np.float64)
    hi = np.array([c[4] for c in cols], np.float64)
    pattern = np.arange(n, dtype=np.int32)
    state = ds2.State(n)
    state.build_range(lo, hi)
    return state, dict(
        alpha_row=alpha,
        pattern=pattern,
        r_ext=r,
        bound_status=bs,
        lo=lo,
        hi=hi,
        sigma=sigma,
        delta=delta,
    )


def run(kind, state, kw, **extra):
    return ds2.chuzc(
        kind,
        state,
        kw["alpha_row"],
        kw["pattern"],
        kw["r_ext"],
        kw["bound_status"],
        kw["lo"],
        kw["hi"],
        kw["sigma"],
        kw["delta"],
        **extra,
    )


# --------------------------------------------------------------------------


def test_module_reports_cycle_source():
    assert isinstance(ds2.have_tsc(), bool)


def test_empty_candidate_set_is_dual_unbounded():
    # sigma = +1 admits LO columns with alpha < 0; this one has alpha > 0.
    state, kw = make_case([(1.0, 1.0, LO, 0.0, INF)])
    for kind in ("ds2", "harris_dense", "harris_pattern"):
        assert run(kind, state, kw)["entering"] == -1


def test_basic_and_fixed_columns_are_never_chosen():
    state, kw = make_case(
        [
            (-1.0, 1.0, BASIC, 0.0, 1.0),
            (-1.0, 1.0, FIXED, 2.0, 2.0),
            (-1.0, 5.0, LO, 0.0, INF),
        ]
    )
    for kind in ("ds2", "harris_dense", "harris_pattern"):
        assert run(kind, state, kw)["entering"] == 2


def test_harris_controls_take_the_minimum_ratio():
    # ratios 3, 1, 2 -> column 1 wins; all |alpha| equal so no band effect.
    state, kw = make_case(
        [
            (-1.0, 3.0, LO, 0.0, INF),
            (-1.0, 1.0, LO, 0.0, INF),
            (-1.0, 2.0, LO, 0.0, INF),
        ]
    )
    for kind in ("harris_dense", "harris_pattern"):
        res = run(kind, state, kw)
        assert res["entering"] == 1
        assert res["flips"] == []
        assert res["theta_dual"] == pytest.approx(1.0)


def test_longest_step_flips_a_boxed_breakpoint_and_pivots_beyond_it():
    # Column 0 is the minimum-ratio breakpoint but is BOXED with a range that
    # absorbs only part of delta, so the walk steps over it (flipping it) and
    # pivots on column 1 instead.
    state, kw = make_case(
        [
            (-1.0, 1.0, LO, 0.0, 1.0),  # ratio 1, absorbs 1*1 = 1
            (-1.0, 5.0, LO, 0.0, INF),  # ratio 5, one-sided: terminates
        ],
        delta=10.0,
    )
    res = run("ds2", state, kw)
    assert res["entering"] == 1
    assert res["flips"] == [0]
    assert res["theta_dual"] == pytest.approx(5.0)
    # the incumbent stops at the first breakpoint
    assert run("harris_dense", state, kw)["entering"] == 0


def test_one_sided_breakpoint_is_never_stepped_over():
    # Same shape, but the minimum-ratio column is one-sided: the walk must
    # stop there, because flipping onto an infinite bound is meaningless.
    state, kw = make_case(
        [
            (-1.0, 1.0, LO, 0.0, INF),
            (-1.0, 5.0, LO, 0.0, 1.0),
        ],
        delta=10.0,
    )
    res = run("ds2", state, kw)
    assert res["entering"] == 0
    assert res["flips"] == []


def test_no_flip_mask_makes_a_boxed_column_behave_as_one_sided():
    # linprogx's big-M artificial boxes are finite but must not be flipped.
    state, kw = make_case(
        [
            (-1.0, 1.0, LO, 0.0, 1.0),
            (-1.0, 5.0, LO, 0.0, INF),
        ],
        delta=10.0,
    )
    assert run("ds2", state, kw)["flips"] == [0]

    state.set_no_flip(np.array([1, 0], np.uint8))
    state.build_range(kw["lo"], kw["hi"])
    res = run("ds2", state, kw)
    assert res["entering"] == 0
    assert res["flips"] == []


def test_upper_bound_columns_admit_on_the_other_sign():
    # sigma = -1 admits HI columns with alpha < 0 and LO columns with alpha > 0.
    state, kw = make_case(
        [
            (-1.0, -2.0, HI, 0.0, INF),
            (1.0, 4.0, LO, 0.0, INF),
        ],
        sigma=-1,
    )
    res = run("ds2", state, kw)
    assert res["entering"] == 0
    assert res["theta_dual"] == pytest.approx(2.0)


def test_zero_step_discards_the_flip_set():
    # The entering column is already dual infeasible (r < 0 at a lower bound),
    # so the dual step is zero and no breakpoint was really crossed.
    state, kw = make_case(
        [
            (-1.0, 0.5, LO, 0.0, 1.0),
            (-1.0, -1e-9, LO, 0.0, INF),
        ],
        delta=10.0,
    )
    res = run("ds2", state, kw)
    assert res["theta_dual"] == 0.0
    assert res["flips"] == []


def test_pivot_admission_threshold_ramps_with_lu_age():
    # |alpha| = 1e-7 is acceptable on a fresh factorisation and not after 20
    # updates, where the threshold is 1e-6.
    state, kw = make_case([(-1e-7, 1.0, LO, 0.0, INF)])
    assert run("ds2", state, kw, update_count=0)["entering"] == 0
    assert run("ds2", state, kw, update_count=25)["entering"] == -1


def test_large_alpha_backoff_prefers_a_stable_pivot_in_the_chosen_group():
    # Two columns share the minimum ratio; the test must take the larger
    # |alpha| for stability, and ties must resolve to the lowest index.
    state, kw = make_case(
        [
            (-0.5, 1.0, LO, 0.0, INF),
            (-2.0, 4.0, LO, 0.0, INF),
        ]
    )
    for kind in ("ds2", "harris_dense", "harris_pattern"):
        assert run(kind, state, kw)["entering"] == 1


def test_range_cache_does_not_change_the_decision():
    state, kw = make_case(
        [
            (-1.0, 1.0, LO, 0.0, 1.0),
            (-1.0, 5.0, LO, 0.0, INF),
        ],
        delta=10.0,
    )
    with_cache = run("ds2", state, kw)
    state.invalidate_range()
    without_cache = run("ds2", state, kw)
    assert with_cache["entering"] == without_cache["entering"]
    assert with_cache["flips"] == without_cache["flips"]
    assert with_cache["theta_dual"] == without_cache["theta_dual"]


def test_census_reports_flippability():
    state, kw = make_case(
        [
            (-1.0, 1.0, LO, 0.0, 1.0),
            (-1.0, 5.0, LO, 0.0, INF),
        ],
        delta=10.0,
    )
    state.set_census(True)
    run("ds2", state, kw)
    census = state.census()
    assert census["n_cand"] == 2
    assert census["n_flippable"] == 1
    assert census["absorb"] == pytest.approx(1.0)
    assert census["delta"] == pytest.approx(10.0)
