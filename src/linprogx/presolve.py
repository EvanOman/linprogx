"""Dependency-free presolve for equality-plus-bounds LPs.

Iterates three reductions to a fixpoint before the sparse PDHG solve:

1. Empty rows are dropped (an inconsistent empty row is left to the solver,
   which will report the infeasibility through its residuals).
2. Singleton rows ``a * x_j = b_i`` fix ``x_j`` and substitute it out.
3. Doubleton rows ``a * x_p + d * x_q = b_i`` eliminate ``x_p`` via
   ``x_p = (b_i - d * x_q) / a``, folding the substitution into every other
   row containing ``x_p``, the objective, and the bounds on ``x_q``.
Duplicate-row removal was tried and removed again: after the cascade above it
finds nothing on the Netlib benchmarks and only adds presolve time.

Doubleton elimination is what unlocks degenerate Netlib shapes such as CYCLE:
it removes the dependent-row mass that stalls PDHG's duality gap. The fill
limit matters more than the amount of reduction; eliminating high-degree
columns creates fill-in that worsens conditioning, so only low-degree columns
are eliminated.

Postsolve replays the recorded substitutions in reverse to reconstruct the
full solution vector.
"""

from __future__ import annotations

import array
import importlib
import os
import struct
from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

_RATIO_LO = 1e-4
_RATIO_HI = 1e4
_PIVOT_EPS = 1e-12
_DROP_EPS = 1e-15
_BOUND_EPS = 1e-10
_V2_MIN_REDUCTION_FRACTION = 0.08
# Second-fixpoint re-stage cost guard: only re-run the combined V2 fixpoint
# after the classic cascade when classic itself shrank the problem by at least
# this fraction of rows or columns. Classic progress is the signal that new
# fixed/forcing/empty/column-singleton opportunities were created; a problem
# classic cannot touch (e.g. the OSA dense-singleton border, where classic
# returns no reduction at all) never reaches this branch, so this guard keeps
# the expensive standalone V2 pass off those negative-control instances.
_FIXPOINT_MIN_CLASSIC_FRACTION = 0.02
# Second-fixpoint acceptance gate: keep the composed V2 reduction only when it
# removes at least this fraction of the reduced problem's rows or columns. A
# tiny second reduction does not speed the solve up and can perturb PDHG/IPM
# conditioning enough to slow it down badly (measured regressions on pds_10,
# d2q06c, and ken_* when the second pass removes < ~1.2% of the reduced shape),
# so those are discarded and the classic-only reduction is kept.
_FIXPOINT_MIN_SECOND_FRACTION = 0.02

try:
    _csparse: object = importlib.import_module("linprogx._csparse")
    _c_presolve = _csparse.presolve_eq_box  # type: ignore[attr-defined]
    _c_presolve_v2 = _csparse.presolve_v2  # type: ignore[attr-defined]
    _c_v2_candidates = _csparse.presolve_v2_candidates  # type: ignore[attr-defined]
    _c_presolve_agg = getattr(_csparse, "presolve_agg", None)  # type: ignore[attr-defined]
    _c_presolve_netagg = getattr(_csparse, "presolve_netagg", None)  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - source tree before extension build
    _csparse = None
    _c_presolve = None
    _c_presolve_v2 = None
    _c_v2_candidates = None
    _c_presolve_agg = None
    _c_presolve_netagg = None

_SSZ = struct.calcsize("n")  # sizeof(Py_ssize_t)

# Determine the array.array typecode that matches Py_ssize_t (signed, native size).
# On LP64 (Linux x86_64) sizeof(long)==8==sizeof(Py_ssize_t) so 'l' works.
# On LLP64 (Windows x64) sizeof(long)==4 but sizeof(long long)==8==sizeof(Py_ssize_t)
# so 'q' is needed.  Pick whichever typecode has the right itemsize.
_INT_TC = "l" if array.array("l").itemsize == _SSZ else "q"


def _pack_dbls(lst: list[float]) -> bytes:
    """Pack a list of floats into raw C double bytes."""
    return array.array("d", lst).tobytes()


def _unpack_ints(b: bytes) -> list[int]:
    """Decode a bytes blob of native Py_ssize_t values into a list of ints."""
    n = len(b) // _SSZ
    return list(struct.unpack(f"{n}n", b))


def _unpack_dbls(b: bytes) -> list[float]:
    """Decode a bytes blob of C doubles into a list of floats."""
    n = len(b) // 8
    return list(struct.unpack(f"{n}d", b))


@dataclass(frozen=True)
class _FixedVar:
    column: int
    value: float


@dataclass(frozen=True)
class _Doubleton:
    eliminated: int
    kept: int
    coef_eliminated: float
    coef_kept: float
    rhs: float


@dataclass(frozen=True)
class _ColumnSingleton:
    eliminated: int
    coef_eliminated: float
    rhs: float
    terms: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _Aggregation:
    """General equality-row aggregation: a (free or implied-free) column is
    substituted out of one equality row into that column's other rows, removing
    both the column and the row. The k>2 generalization of doubleton
    elimination. Postsolve mirrors ``_ColumnSingleton``: the eliminated column's
    value is recovered from the pivot equality's residual, using ``terms`` (the
    pivot row's other entries in the reduced column space at elimination time)."""

    eliminated: int
    coef_eliminated: float
    rhs: float
    terms: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _NetAggregation:
    """Equality aggregation certified by all incident rows together.

    Each incident equality supplies a valid interval for the eliminated
    column from the activity bounds of its other variables.  The explicit box
    is redundant when the intersection of those intervals lies inside it.
    Any stable incident row may then be used as the substitution pivot.
    """

    eliminated: int
    coef_eliminated: float
    rhs: float
    terms: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _DuplicateColumn:
    removed: int
    kept: int
    removed_lo: float
    removed_hi: float
    kept_lo: float
    kept_hi: float


class _LazyRecordList:
    """Lazily materializes presolve records from raw C tuples.

    Defers the dataclass construction overhead to postsolve time,
    keeping the presolve call itself fast.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: list[tuple[Any, ...]]) -> None:
        self._raw = raw

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self):  # type: ignore[override]
        for rec in self._raw:
            if rec[0] == 0:
                yield _FixedVar(rec[1], rec[2])
            elif rec[0] == 1:
                yield _Doubleton(rec[1], rec[2], rec[3], rec[4], rec[5])
            elif rec[0] == 2:
                yield _ColumnSingleton(rec[1], rec[2], rec[3], tuple(rec[4]))
            elif rec[0] == 3:
                yield _DuplicateColumn(rec[1], rec[2], rec[3], rec[4], rec[5], rec[6])
            elif rec[0] == 5:
                yield _Aggregation(rec[1], rec[2], rec[3], tuple(rec[4]))
            elif rec[0] == 6:
                yield _NetAggregation(rec[1], rec[2], rec[3], tuple(rec[4]))
            else:
                raise ValueError(f"unknown presolve record tag {rec[0]}")

    def __reversed__(self):  # type: ignore[override]
        for rec in reversed(self._raw):
            if rec[0] == 0:
                yield _FixedVar(rec[1], rec[2])
            elif rec[0] == 1:
                yield _Doubleton(rec[1], rec[2], rec[3], rec[4], rec[5])
            elif rec[0] == 2:
                yield _ColumnSingleton(rec[1], rec[2], rec[3], tuple(rec[4]))
            elif rec[0] == 3:
                yield _DuplicateColumn(rec[1], rec[2], rec[3], rec[4], rec[5], rec[6])
            elif rec[0] == 5:
                yield _Aggregation(rec[1], rec[2], rec[3], tuple(rec[4]))
            elif rec[0] == 6:
                yield _NetAggregation(rec[1], rec[2], rec[3], tuple(rec[4]))
            else:
                raise ValueError(f"unknown presolve record tag {rec[0]}")


class _ComposedRecordList:
    """Lazily concatenate and relabel records from sequential reductions."""

    __slots__ = ("_active_cols", "_first", "_second")

    def __init__(self, first: Any, second: Any, active_cols: list[int]) -> None:
        self._first = first
        self._second = second
        self._active_cols = active_cols

    def __len__(self) -> int:
        return len(self._first) + len(self._second)

    def __iter__(self):  # type: ignore[override]
        yield from self._first
        for record in self._second:
            yield _remap_record(record, self._active_cols)

    def __reversed__(self):  # type: ignore[override]
        for record in reversed(self._second):
            yield _remap_record(record, self._active_cols)
        yield from reversed(self._first)


def _empty_reduction_counts() -> dict[str, int]:
    return {
        "fixed_columns": 0,
        "empty_columns": 0,
        "forcing_rows": 0,
        "forcing_columns": 0,
        "empty_rows": 0,
        "singleton_rows": 0,
        "column_singletons": 0,
        "doubletons": 0,
        "duplicate_columns": 0,
        "aggregations": 0,
        "net_aggregations": 0,
    }


@dataclass
class PresolveResult:
    """A reduced equality-plus-bounds LP plus the recipe to undo it."""

    rows: int
    cols: int
    indptr: list[int]
    indices: list[int]
    data: list[float]
    b: list[float]
    c: list[float]
    lo: list[float]
    hi: list[float]
    objective_offset: float
    removed_rows: int
    removed_cols: int
    _records: (
        list[
            _FixedVar
            | _Doubleton
            | _ColumnSingleton
            | _DuplicateColumn
            | _Aggregation
            | _NetAggregation
        ]
        | _LazyRecordList
        | _ComposedRecordList
    )
    _active_cols: list[int]
    _original_cols: int
    _reduction_counts: dict[str, int] = field(default_factory=_empty_reduction_counts)
    _matrix: Any = field(default=None, repr=False)


def _v2_enabled() -> bool:
    return os.environ.get("LINPROGX_PRESOLVE_V2", "1") != "0"


def _agg_enabled() -> bool:
    """General equality-row aggregation inside the raw pure-Python reducer
    (``_presolve_eq_box_python`` / ``presolve_eq_box`` direct calls).

    This governs ONLY direct reducer invocations, so it stays default OFF to keep
    those paths (and the C/Python bit-equivalence tests) byte-identical to HEAD.
    The solver's aggregation ships through the composed ``_maybe_aggregate``
    re-stage, gated by ``_agg_restage_enabled`` below. Set
    ``LINPROGX_PRESOLVE_AGG=1`` to force the raw agg block on."""
    return os.environ.get("LINPROGX_PRESOLVE_AGG", "0") == "1"


def _agg_restage_enabled() -> bool:
    """Solver-path aggregation re-stage. DEFAULT ON (native, fill-gated).

    Verdict (2026-07-17): equality-row aggregation is fill-negative on the shapes
    the structural fill-gate accepts, and cuts IPM iterations (80bau3b 47->44,
    d2q06c 48->47), for a measured solve-side saving of ~14ms on 80bau3b. The
    pure-Python re-stage (~34ms) swamped that, so the re-stage stayed OFF until the
    native ``_csparse`` port (``presolve_agg``) drove the pass cost to ~2-3ms while
    reproducing the Python reference bit-for-bit. With the native pass the re-stage
    nets a wall win on every fill-gate accept (80bau3b -7.9%, d2q06c -19.7%,
    ken_07 -7.7%) and is now the ship default. The fill-gate structurally rejects
    every other board instance; ``LINPROGX_PRESOLVE_AGG=0`` restores the
    byte-identical no-re-stage behavior. See docs/HANDOFF.md."""
    return os.environ.get("LINPROGX_PRESOLVE_AGG", "1") != "0"


def _agg_fillgate_enabled() -> bool:
    """Fill-gate on the aggregation re-stage (default ON). The re-stage is kept
    only when it does not grow nnz (fill-non-positive), which structurally
    excludes fill-positive instances such as greenbea. Set
    ``LINPROGX_AGG_FILLGATE=0`` for the unconditional (always-compose) arm."""
    return os.environ.get("LINPROGX_AGG_FILLGATE", "1") != "0"


def _agg_max_fill() -> int:
    """Cap on new nonzeros created per aggregation elimination (fill guard)."""
    return int(os.environ.get("LINPROGX_AGG_MAX_FILL", "30"))


def _agg_pivot_tol() -> float:
    """Markowitz-style stability floor: the pivot coefficient must be at least
    this fraction of the largest magnitude in its equality row."""
    return float(os.environ.get("LINPROGX_AGG_PIVOT_TOL", "0.01"))


def _agg_max_nnz() -> int:
    """Size gate for the aggregation re-stage. The re-stage runs a pure-Python
    presolve pass, so it is only attempted when the reduced problem is small
    enough that the pass cost cannot regress a large fixture. Large sentinel
    fixtures (pds/osa/stocfor/pds_20) stay above this and are never re-scanned."""
    return int(os.environ.get("LINPROGX_AGG_MAX_NNZ", "50000"))


def _netagg_enabled() -> bool:
    """Large equality-network aggregation. Default OFF until certified."""
    return os.environ.get("LINPROGX_PRESOLVE_NETAGG", "1") == "1"


_NETAGG_MIN_ROWS = 10_000
_NETAGG_MIN_NNZ = 100_000
_NETAGG_MIN_NNZ_REDUCTION_FRACTION = 0.10


def _v2_native_enabled() -> bool:
    return os.environ.get("LINPROGX_PRESOLVE_V2_NATIVE", "1") != "0"


def _v2_worth_python_pass(
    candidate_rows: int,
    candidate_cols: int,
    rows: int,
    cols: int,
) -> bool:
    return candidate_rows >= _V2_MIN_REDUCTION_FRACTION * max(
        1, rows
    ) or candidate_cols >= _V2_MIN_REDUCTION_FRACTION * max(1, cols)


def _fixpoint_enabled() -> bool:
    """Second-fixpoint re-stage (default ON). Set to ``0`` for the byte-identical
    pre-change presolve behavior."""
    return os.environ.get("LINPROGX_PRESOLVE_FIXPOINT", "1") != "0"


def _fixpoint_worth_restage(
    removed_rows: int,
    removed_cols: int,
    rows: int,
    cols: int,
) -> bool:
    return removed_rows >= _FIXPOINT_MIN_CLASSIC_FRACTION * max(
        1, rows
    ) or removed_cols >= _FIXPOINT_MIN_CLASSIC_FRACTION * max(1, cols)


def _fixpoint_reduction_is_substantial(
    removed_rows: int,
    removed_cols: int,
    rows: int,
    cols: int,
) -> bool:
    return removed_rows >= _FIXPOINT_MIN_SECOND_FRACTION * max(
        1, rows
    ) or removed_cols >= _FIXPOINT_MIN_SECOND_FRACTION * max(1, cols)


def _remap_record(record: Any, active_cols: list[int]) -> Any:
    """Relabel a reduction record's column indices from the intermediate
    (classic-reduced) space back into the original column space via
    ``active_cols`` (intermediate column -> original column)."""
    if isinstance(record, _FixedVar):
        return _FixedVar(active_cols[record.column], record.value)
    if isinstance(record, _Doubleton):
        return _Doubleton(
            active_cols[record.eliminated],
            active_cols[record.kept],
            record.coef_eliminated,
            record.coef_kept,
            record.rhs,
        )
    if isinstance(record, _ColumnSingleton):
        return _ColumnSingleton(
            active_cols[record.eliminated],
            record.coef_eliminated,
            record.rhs,
            tuple((active_cols[j], coef) for j, coef in record.terms),
        )
    if isinstance(record, _DuplicateColumn):
        return _DuplicateColumn(
            active_cols[record.removed],
            active_cols[record.kept],
            record.removed_lo,
            record.removed_hi,
            record.kept_lo,
            record.kept_hi,
        )
    if isinstance(record, _Aggregation):
        return _Aggregation(
            active_cols[record.eliminated],
            record.coef_eliminated,
            record.rhs,
            tuple((active_cols[j], coef) for j, coef in record.terms),
        )
    if isinstance(record, _NetAggregation):
        return _NetAggregation(
            active_cols[record.eliminated],
            record.coef_eliminated,
            record.rhs,
            tuple((active_cols[j], coef) for j, coef in record.terms),
        )
    raise ValueError(f"unknown presolve record type {type(record)!r}")


def _compose_reductions(
    first: PresolveResult,
    second: PresolveResult,
) -> PresolveResult:
    """Compose two reductions applied in sequence (original -> first-reduced
    -> second-reduced) into a single reduction from the original space.

    ``second`` was computed on ``first``'s reduced matrix, so its records and
    active columns index that intermediate space. Postsolve replays records in
    reverse: appending ``second``'s (remapped) records after ``first``'s means
    reversed replay undoes ``second`` first, then ``first`` -- the correct order.
    """
    a1 = first._active_cols  # intermediate column -> original column
    combined_active = [a1[j] for j in second._active_cols]
    combined_records = _ComposedRecordList(first._records, second._records, a1)
    combined_counts = _empty_reduction_counts()
    for key in combined_counts:
        combined_counts[key] = first._reduction_counts.get(key, 0) + second._reduction_counts.get(
            key, 0
        )
    # Mirror ``second``'s reduced-matrix representation: the native reducer
    # returns a CSRMatrix in ``_matrix`` (components empty); the Python reducer
    # returns components with ``_matrix`` None. Either is valid downstream.
    return PresolveResult(
        rows=second.rows,
        cols=second.cols,
        indptr=second.indptr,
        indices=second.indices,
        data=second.data,
        b=second.b,
        c=second.c,
        lo=second.lo,
        hi=second.hi,
        objective_offset=first.objective_offset + second.objective_offset,
        removed_rows=first.removed_rows + second.removed_rows,
        removed_cols=first.removed_cols + second.removed_cols,
        _records=combined_records,
        _active_cols=combined_active,
        _original_cols=first._original_cols,
        _reduction_counts=combined_counts,
        _matrix=second._matrix,
    )


def _maybe_aggregate(result: PresolveResult, max_fill: int) -> PresolveResult:
    """Re-stage: run a pure-Python aggregation fixpoint on the already-reduced
    problem and compose it onto ``result`` when the extra reduction is
    substantial. The aggregation kernel lives only in the Python reducer (the C
    extension is untouched), so it is injected here as a composed second stage.

    Two guards apply. A size gate keeps the Python pass off large fixtures whose
    pass cost alone could regress them. A substantiality gate (the H1 pattern)
    discards a tiny reduction that would not pay for itself and could perturb
    downstream PDHG/IPM conditioning."""
    if result._matrix is not None:
        nnz = result._matrix.nnz
    else:
        nnz = len(result.data)
    if nnz > _agg_max_nnz():
        return result
    # Early-abort budget: aggregation stops once its cumulative fill exceeds this,
    # so a fill-positive instance bails after a handful of substitutions instead
    # of paying the whole cascade before the fill-gate rejects it. The board's
    # fill-negative accepts peak at a tiny transient excursion before diving
    # negative (measured cumulative-fill peaks: 80bau3b +20, d2q06c -2, ken_07 -2),
    # while the fill-positive rejects climb monotonically (greenbea, cre_a). A tight
    # constant cap cleanly separates them: it never triggers on the accepts (so they
    # complete bit-identically) yet aborts the rejects within a few aggregations,
    # which is what keeps the reject-path pass cost negligible. The previous
    # max(256, nnz//40) cap was far too loose -- cre_a ran ~1026 aggregations before
    # crossing 418, making the reject path expensive.
    fill_budget = 48
    # Prefer the native (_csparse) aggregation re-stage: it reproduces the
    # agg_only Python path bit-for-bit (the reduced problem and the record
    # stream) but at ~10-30x lower pass cost, which is what lets the re-stage
    # net a wall win. The Python path remains the reference and the fallback
    # when the extension is unavailable.
    if _c_presolve_agg is not None and result._matrix is not None:
        # Pass ``nnz`` so the native pass can apply the fill-gate internally and
        # return None on a fill-positive reject WITHOUT materializing the reduced
        # matrix/records it would only throw away (unless the gate is disabled).
        gate_nnz = nnz if _agg_fillgate_enabled() else -1
        raw = _c_presolve_agg(
            result._matrix,
            _pack_dbls(result.b),
            _pack_dbls(result.c),
            _pack_dbls(result.lo),
            _pack_dbls(result.hi),
            max_fill,
            fill_budget,
            gate_nnz,
        )
        second = None if raw is None else _result_from_c(raw)
    else:
        if result._matrix is not None:
            r_indptr, r_indices, r_data = result._matrix.to_components()
        else:
            r_indptr, r_indices, r_data = result.indptr, result.indices, result.data
        second = _presolve_eq_box_python(
            result.rows,
            result.cols,
            r_indptr,
            r_indices,
            r_data,
            list(result.b),
            list(result.c),
            list(result.lo),
            list(result.hi),
            max_fill=max_fill,
            agg=True,
            agg_fill_budget=fill_budget,
            agg_only=True,
        )
    if second is None or not _fixpoint_reduction_is_substantial(
        second.removed_rows, second.removed_cols, result.rows, result.cols
    ):
        return result
    # Fill-gate (SHIP DEFAULT): compose the aggregation only when it did not grow
    # nonzeros. This is a global structural threshold -- no per-problem tuning --
    # that keeps aggregation where it is fill-non-positive (80bau3b: 21798->21511)
    # and structurally excludes the fill-positive cases (greenbea: 23274->26683).
    second_nnz = second._matrix.nnz if second._matrix is not None else len(second.data)
    if _agg_fillgate_enabled() and second_nnz > nnz:
        return result
    return _compose_reductions(result, second)


def _maybe_netaggregate(result: PresolveResult, max_fill: int) -> PresolveResult:
    """Compose the native multi-row implied-bound aggregation stage.

    The up-front size gates and the final 10% nnz gate are global structural
    policy.  The native kernel separately requires every committed elimination
    to have an exact nonpositive nnz delta.
    """
    nnz = result._matrix.nnz if result._matrix is not None else len(result.data)
    if result.rows < _NETAGG_MIN_ROWS or nnz < _NETAGG_MIN_NNZ:
        return result
    if _c_presolve_netagg is None or result._matrix is None:
        return result
    raw = _c_presolve_netagg(
        result._matrix,
        _pack_dbls(result.b),
        _pack_dbls(result.c),
        _pack_dbls(result.lo),
        _pack_dbls(result.hi),
        max_fill,
        -1,
        nnz,
    )
    if raw is None:
        return result
    second = _result_from_c(raw)
    second_nnz = second._matrix.nnz if second._matrix is not None else len(second.data)
    if second_nnz > (1.0 - _NETAGG_MIN_NNZ_REDUCTION_FRACTION) * nnz:
        return result
    return _compose_reductions(result, second)


def _same_bound(lo_value: float, hi_value: float) -> bool:
    return (
        isfinite(lo_value)
        and isfinite(hi_value)
        and abs(lo_value - hi_value) <= _BOUND_EPS * max(1.0, abs(lo_value), abs(hi_value))
    )


def _near_bound(value: float, bound: float) -> bool:
    return abs(value - bound) <= _BOUND_EPS * max(1.0, abs(value), abs(bound))


def _activity_bounds(
    terms: tuple[tuple[int, float], ...] | list[tuple[int, float]],
    lo: list[float],
    hi: list[float],
) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    lower_unbounded = False
    upper_unbounded = False
    for j, coef in terms:
        if coef >= 0.0:
            if isfinite(lo[j]):
                lower += coef * lo[j]
            else:
                lower_unbounded = True
            if isfinite(hi[j]):
                upper += coef * hi[j]
            else:
                upper_unbounded = True
        else:
            if isfinite(hi[j]):
                lower += coef * hi[j]
            else:
                lower_unbounded = True
            if isfinite(lo[j]):
                upper += coef * lo[j]
            else:
                upper_unbounded = True
    return (-float("inf") if lower_unbounded else lower, float("inf") if upper_unbounded else upper)


def _column_bounds_are_redundant(
    j: int,
    coef: float,
    rhs: float,
    terms: tuple[tuple[int, float], ...],
    lo: list[float],
    hi: list[float],
) -> bool:
    if not isfinite(lo[j]) and not isfinite(hi[j]):
        return True
    rest_lo, rest_hi = _activity_bounds(terms, lo, hi)
    if coef > 0.0:
        implied_lo = -float("inf") if not isfinite(rest_hi) else (rhs - rest_hi) / coef
        implied_hi = float("inf") if not isfinite(rest_lo) else (rhs - rest_lo) / coef
    else:
        implied_lo = -float("inf") if not isfinite(rest_lo) else (rhs - rest_lo) / coef
        implied_hi = float("inf") if not isfinite(rest_hi) else (rhs - rest_hi) / coef

    if isfinite(lo[j]):
        if not isfinite(implied_lo):
            return False
        if implied_lo < lo[j] - _BOUND_EPS * max(1.0, abs(lo[j]), abs(implied_lo)):
            return False
    if isfinite(hi[j]):
        if not isfinite(implied_hi):
            return False
        if implied_hi > hi[j] + _BOUND_EPS * max(1.0, abs(hi[j]), abs(implied_hi)):
            return False
    return True


def _row_activity(
    entries: dict[int, float], lo: list[float], hi: list[float]
) -> tuple[float, float, int, int, int, float]:
    """One O(degree) pass over a whole row returning its cached activity summary:
    ``(lo_fin, hi_fin, lo_unb, hi_unb, _, row_max)`` where the row's activity
    lower bound is ``-inf`` if ``lo_unb`` else ``lo_fin`` (symmetrically upper),
    and ``row_max`` is the largest coefficient magnitude (Markowitz floor). The
    per-column implied-free test then subtracts one column's contribution in
    O(1), replacing the quadratic per-(column, row) ``_activity_bounds`` sweep."""
    lo_fin = 0.0
    hi_fin = 0.0
    lo_unb = 0
    hi_unb = 0
    row_max = 0.0
    for coef in entries.values():
        av = -coef if coef < 0.0 else coef
        if av > row_max:
            row_max = av
    for k, coef in entries.items():
        lok = lo[k]
        hik = hi[k]
        if coef >= 0.0:
            if isfinite(lok):
                lo_fin += coef * lok
            else:
                lo_unb += 1
            if isfinite(hik):
                hi_fin += coef * hik
            else:
                hi_unb += 1
        else:
            if isfinite(hik):
                lo_fin += coef * hik
            else:
                lo_unb += 1
            if isfinite(lok):
                hi_fin += coef * lok
            else:
                hi_unb += 1
    return lo_fin, hi_fin, lo_unb, hi_unb, 0, row_max


def _implied_free_from_activity(
    j: int,
    coef: float,
    rhs: float,
    lo: list[float],
    hi: list[float],
    lo_fin: float,
    hi_fin: float,
    lo_unb: int,
    hi_unb: int,
) -> bool:
    """O(1) implied-free test: the same decision as ``_column_bounds_are_redundant``
    but derived from a cached whole-row activity summary by subtracting column
    ``j``'s own contribution instead of re-summing every other column."""
    loj = lo[j]
    hij = hi[j]
    loj_fin = isfinite(loj)
    hij_fin = isfinite(hij)
    if not loj_fin and not hij_fin:
        return True
    # Column j's contribution to the row activity (mirrors _row_activity).
    if coef >= 0.0:
        j_lo_fin = coef * loj if loj_fin else 0.0
        j_lo_unb = 0 if loj_fin else 1
        j_hi_fin = coef * hij if hij_fin else 0.0
        j_hi_unb = 0 if hij_fin else 1
    else:
        j_lo_fin = coef * hij if hij_fin else 0.0
        j_lo_unb = 0 if hij_fin else 1
        j_hi_fin = coef * loj if loj_fin else 0.0
        j_hi_unb = 0 if loj_fin else 1
    rest_lo_inf = (lo_unb - j_lo_unb) > 0
    rest_hi_inf = (hi_unb - j_hi_unb) > 0
    rest_lo = lo_fin - j_lo_fin
    rest_hi = hi_fin - j_hi_fin
    if coef > 0.0:
        implied_lo_inf = rest_hi_inf
        implied_hi_inf = rest_lo_inf
        implied_lo = -float("inf") if rest_hi_inf else (rhs - rest_hi) / coef
        implied_hi = float("inf") if rest_lo_inf else (rhs - rest_lo) / coef
    else:
        implied_lo_inf = rest_lo_inf
        implied_hi_inf = rest_hi_inf
        implied_lo = -float("inf") if rest_lo_inf else (rhs - rest_lo) / coef
        implied_hi = float("inf") if rest_hi_inf else (rhs - rest_hi) / coef
    if loj_fin:
        if implied_lo_inf:
            return False
        if implied_lo < loj - _BOUND_EPS * max(1.0, abs(loj), abs(implied_lo)):
            return False
    if hij_fin:
        if implied_hi_inf:
            return False
        if implied_hi > hij + _BOUND_EPS * max(1.0, abs(hij), abs(implied_hi)):
            return False
    return True


def _choose_empty_column_value(
    j: int, c: list[float], lo: list[float], hi: list[float]
) -> float | None:
    # Equality dual variables are free, so sign-based dual fixing is sound here
    # only for columns absent from every equality row.
    if c[j] > _DROP_EPS:
        return lo[j] if isfinite(lo[j]) else None
    if c[j] < -_DROP_EPS:
        return hi[j] if isfinite(hi[j]) else None
    if (not isfinite(lo[j]) or lo[j] <= 0.0) and (not isfinite(hi[j]) or hi[j] >= 0.0):
        return 0.0
    if isfinite(lo[j]):
        return lo[j]
    if isfinite(hi[j]):
        return hi[j]
    return 0.0


def _presolve_eq_box_python(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
    *,
    max_fill: int = 5,
    agg: bool | None = None,
    agg_fill_budget: int | None = None,
    agg_only: bool = False,
) -> PresolveResult | None:
    """Pure-Python reference implementation (kept as the fallback).

    ``agg_only`` (used by the ``_maybe_aggregate`` re-stage) runs *only* the
    aggregation drain plus a single empty-row/empty-column cleanup, skipping the
    other reduction passes -- the C first stage already drove those to a fixpoint,
    so re-running the full 8-block cascade on every wave is pure overhead. This is
    the difference between a ~15ms and a ~120ms re-stage on 80bau3b.

    ``agg`` overrides the env-driven aggregation switch when not ``None`` (the
    ``_maybe_aggregate`` re-stage passes ``agg=True`` explicitly so the ship
    default does not depend on ``_agg_enabled``). ``agg_fill_budget`` caps the
    cumulative aggregation fill before the scan aborts early."""
    b = list(b)
    c = list(c)
    lo = list(lo)
    hi = list(hi)

    row_entries: list[dict[int, float]] = []
    for i in range(rows):
        entries: dict[int, float] = {}
        for offset in range(indptr[i], indptr[i + 1]):
            value = data[offset]
            if abs(value) > _DROP_EPS:
                entries[indices[offset]] = value
        row_entries.append(entries)

    col_rows: list[set[int]] = [set() for _ in range(cols)]
    for i in range(rows):
        for j in row_entries[i]:
            col_rows[j].add(i)

    records: list[
        _FixedVar
        | _Doubleton
        | _ColumnSingleton
        | _DuplicateColumn
        | _Aggregation
        | _NetAggregation
    ] = []
    removed_rows: set[int] = set()
    removed_cols: set[int] = set()
    objective_offset = 0.0
    v2 = _v2_enabled()
    agg = _agg_enabled() if agg is None else agg
    agg_max_fill = _agg_max_fill()
    agg_pivot_tol = _agg_pivot_tol()
    reduction_counts = _empty_reduction_counts()

    # Aggregation scan state (only touched when ``agg``). ``agg_worklist`` is a
    # FIFO of columns worth (re)examining -- seeded with every column, then fed
    # incrementally by the columns of rows an aggregation rewrites, so the scan
    # never re-sweeps stable columns across fixpoint waves. ``row activity cache``
    # memoizes each row's largest-magnitude coefficient for the Markowitz floor.
    agg_worklist: deque[int] = deque()
    agg_in_queue: set[int] = set()
    # Cached per-row activity summary (lo_fin, hi_fin, lo_unb, hi_unb, _, row_max);
    # built once per row and reused across all its candidate columns.
    row_act_cache: dict[int, tuple[float, float, int, int, int, float]] = {}
    agg_fill_delta = 0
    agg_aborted = False
    if agg:
        agg_worklist.extend(range(cols))
        agg_in_queue.update(range(cols))

    changed = True
    while changed:
        changed = False

        if v2 and not agg_only:
            for j in range(cols):
                if j in removed_cols or not _same_bound(lo[j], hi[j]):
                    continue
                value = lo[j]
                records.append(_FixedVar(j, value))
                objective_offset += c[j] * value
                for i in list(col_rows[j]):
                    if i in removed_rows:
                        continue
                    coef = row_entries[i].get(j)
                    if coef is None:
                        continue
                    b[i] -= coef * value
                    del row_entries[i][j]
                    col_rows[j].discard(i)
                removed_cols.add(j)
                col_rows[j].clear()
                reduction_counts["fixed_columns"] += 1
                changed = True

            for j in range(cols):
                if j in removed_cols or col_rows[j]:
                    continue
                value = _choose_empty_column_value(j, c, lo, hi)
                if value is None:
                    continue
                records.append(_FixedVar(j, value))
                objective_offset += c[j] * value
                removed_cols.add(j)
                reduction_counts["empty_columns"] += 1
                changed = True

            for i in range(rows):
                if i in removed_rows or not row_entries[i]:
                    continue
                entries: tuple[tuple[int, float], ...] = tuple(row_entries[i].items())
                lower, upper = _activity_bounds(entries, lo, hi)
                at_lower = isfinite(lower) and _near_bound(b[i], lower)
                at_upper = isfinite(upper) and _near_bound(b[i], upper)
                if not at_lower and not at_upper:
                    continue

                fixes: list[tuple[int, float]] = []
                for j, coef in entries:
                    if j in removed_cols:
                        continue
                    if at_lower:
                        value = lo[j] if coef >= 0.0 else hi[j]
                    else:
                        value = hi[j] if coef >= 0.0 else lo[j]
                    if not isfinite(value):
                        fixes = []
                        break
                    fixes.append((j, value))
                if not fixes:
                    continue

                forcing_columns = 0
                for j, value in fixes:
                    if j in removed_cols:
                        continue
                    records.append(_FixedVar(j, value))
                    objective_offset += c[j] * value
                    for other in list(col_rows[j]):
                        if other in removed_rows:
                            continue
                        coef = row_entries[other].get(j)
                        if coef is None:
                            continue
                        b[other] -= coef * value
                        del row_entries[other][j]
                        col_rows[j].discard(other)
                    removed_cols.add(j)
                    col_rows[j].clear()
                    forcing_columns += 1
                    changed = True
                reduction_counts["forcing_rows"] += 1
                reduction_counts["forcing_columns"] += forcing_columns
                if not row_entries[i]:
                    removed_rows.add(i)

        for i in range(rows):
            if i in removed_rows:
                continue
            if not row_entries[i]:
                removed_rows.add(i)
                reduction_counts["empty_rows"] += 1
                changed = True

        if agg_only:
            # Lean re-stage cleanup: fix columns that aggregation emptied. This
            # replaces the skipped v2 fixed/empty/forcing cascade -- only columns
            # left with no rows need a value chosen.
            for j in range(cols):
                if j in removed_cols or col_rows[j]:
                    continue
                value = _choose_empty_column_value(j, c, lo, hi)
                if value is None:
                    continue
                records.append(_FixedVar(j, value))
                objective_offset += c[j] * value
                removed_cols.add(j)
                reduction_counts["empty_columns"] += 1
                changed = True

        for i in () if agg_only else range(rows):
            if i in removed_rows or len(row_entries[i]) != 1:
                continue
            j, coef = next(iter(row_entries[i].items()))
            if j in removed_cols or abs(coef) < _PIVOT_EPS:
                continue
            value = b[i] / coef
            value = min(max(value, lo[j]), hi[j])
            records.append(_FixedVar(j, value))
            objective_offset += c[j] * value
            for other in list(col_rows[j]):
                if other == i or other in removed_rows:
                    continue
                entry = row_entries[other].get(j)
                if entry is not None:
                    b[other] -= entry * value
                    del row_entries[other][j]
                    col_rows[j].discard(other)
            removed_rows.add(i)
            removed_cols.add(j)
            col_rows[j].clear()
            reduction_counts["singleton_rows"] += 1
            changed = True

        if v2 and not agg_only:
            for j in range(cols):
                if j in removed_cols or len(col_rows[j]) != 1:
                    continue
                i = next(iter(col_rows[j]))
                if i in removed_rows or j not in row_entries[i] or len(row_entries[i]) <= 2:
                    continue
                coef = row_entries[i][j]
                if abs(coef) < _PIVOT_EPS:
                    continue
                terms = tuple((k, value) for k, value in row_entries[i].items() if k != j)
                if not _column_bounds_are_redundant(j, coef, b[i], terms, lo, hi):
                    continue

                records.append(_ColumnSingleton(j, coef, b[i], terms))
                objective_offset += c[j] * b[i] / coef
                cost_scale = c[j] / coef
                for k, value in terms:
                    c[k] -= cost_scale * value
                    col_rows[k].discard(i)
                removed_rows.add(i)
                removed_cols.add(j)
                col_rows[j].clear()
                reduction_counts["column_singletons"] += 1
                changed = True

        for i in () if agg_only else range(rows):
            if i in removed_rows or len(row_entries[i]) != 2:
                continue
            (jp, ap), (jq, dq) = row_entries[i].items()
            if jp in removed_cols or jq in removed_cols:
                continue

            # Eliminate the lower-degree column to limit fill-in.
            if len(col_rows[jp]) > len(col_rows[jq]):
                jp, ap, jq, dq = jq, dq, jp, ap
            if abs(ap) < _PIVOT_EPS:
                jp, ap, jq, dq = jq, dq, jp, ap
                if abs(ap) < _PIVOT_EPS:
                    continue
            ratio = abs(dq / ap)
            if ratio < _RATIO_LO or ratio > _RATIO_HI:
                jp, ap, jq, dq = jq, dq, jp, ap
                if abs(ap) < _PIVOT_EPS:
                    continue
                ratio = abs(dq / ap)
                if ratio < _RATIO_LO or ratio > _RATIO_HI:
                    continue
            if len(col_rows[jp]) - 1 > max_fill:
                continue

            # x_p = beta + alpha * x_q with the bounds of x_p mapped onto x_q.
            alpha = -dq / ap
            beta = b[i] / ap
            new_lo = lo[jq]
            new_hi = hi[jq]
            if alpha > _DROP_EPS:
                if isfinite(lo[jp]):
                    new_lo = max(new_lo, (lo[jp] - beta) / alpha)
                if isfinite(hi[jp]):
                    new_hi = min(new_hi, (hi[jp] - beta) / alpha)
            elif alpha < -_DROP_EPS:
                if isfinite(hi[jp]):
                    new_lo = max(new_lo, (hi[jp] - beta) / alpha)
                if isfinite(lo[jp]):
                    new_hi = min(new_hi, (lo[jp] - beta) / alpha)
            if new_lo > new_hi + 1e-8:
                continue
            lo[jq] = new_lo
            hi[jq] = new_hi

            records.append(_Doubleton(jp, jq, ap, dq, b[i]))
            objective_offset += c[jp] * beta
            c[jq] += c[jp] * alpha

            for other in list(col_rows[jp]):
                if other == i or other in removed_rows:
                    continue
                coef_p = row_entries[other].get(jp)
                if coef_p is None or abs(coef_p) < _DROP_EPS:
                    continue
                b[other] -= coef_p * beta
                merged = row_entries[other].get(jq, 0.0) + coef_p * alpha
                if abs(merged) < _DROP_EPS:
                    row_entries[other].pop(jq, None)
                    col_rows[jq].discard(other)
                else:
                    row_entries[other][jq] = merged
                    col_rows[jq].add(other)
                del row_entries[other][jp]
                col_rows[jp].discard(other)

            removed_rows.add(i)
            removed_cols.add(jp)
            col_rows[jp].clear()
            reduction_counts["doubletons"] += 1
            changed = True

        if agg and not agg_aborted:
            # Worklist-driven aggregation scan. Only columns whose incident rows
            # were rewritten (or the initial seed) are examined, so the scan is
            # near-linear in the number of substitutions rather than
            # O(waves x cols). ``row_act_cache`` memoizes each row's activity
            # summary so the implied-free test is O(1) per candidate column.
            row_act_cache.clear()
            while agg_worklist:
                j = agg_worklist.popleft()
                agg_in_queue.discard(j)
                if j in removed_cols:
                    continue
                # Sorted iteration (ascending row index) makes the pivot
                # tie-break and the worklist append order deterministic and
                # reproducible by the native _csparse aggregation port. Set
                # iteration order is not portable to C, so it is canonicalized
                # here; ``col_rows[j]`` is unchanged between this scan and the
                # substitution loop below, so the same order is reused there.
                rows_j = sorted(i for i in col_rows[j] if i not in removed_rows)
                if not rows_j:
                    continue
                # Choose the pivot equality row for column j that is (a) implied
                # free, (b) numerically safe (Markowitz row-relative floor), and
                # (c) within the fill guard, preferring the lowest fill.
                best: tuple[int, int, float, tuple[tuple[int, float], ...]] | None = None
                for i in rows_j:
                    row_i = row_entries[i]
                    coef = row_i.get(j)
                    if coef is None or abs(coef) < _PIVOT_EPS or len(row_i) < 2:
                        continue
                    act = row_act_cache.get(i)
                    if act is None:
                        act = _row_activity(row_i, lo, hi)
                        row_act_cache[i] = act
                    lo_fin, hi_fin, lo_unb, hi_unb, _unused, row_max = act
                    if abs(coef) < agg_pivot_tol * row_max:
                        continue
                    if not _implied_free_from_activity(
                        j, coef, b[i], lo, hi, lo_fin, hi_fin, lo_unb, hi_unb
                    ):
                        continue
                    terms = tuple((k, v) for k, v in row_i.items() if k != j)
                    # Fill guard: count nonzeros newly created across the other
                    # rows of j (conservative -- cancellations are not credited).
                    fill = 0
                    ok = True
                    for r in rows_j:
                        if r == i:
                            continue
                        rr = row_entries[r]
                        for k, _ in terms:
                            if k not in rr:
                                fill += 1
                                if fill > agg_max_fill:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if not ok:
                        continue
                    if best is None or fill < best[0]:
                        best = (fill, i, coef, terms)
                if best is None:
                    continue

                _, i, coef, terms = best
                records.append(_Aggregation(j, coef, b[i], terms))
                objective_offset += c[j] * b[i] / coef
                cost_scale = c[j] / coef
                for k, value in terms:
                    c[k] -= cost_scale * value
                # Track the net nnz change: the pivot row and column vanish, each
                # other row loses its (r, j) entry, and term substitution adds or
                # cancels fill. The running total drives the early-abort budget.
                delta = -len(row_entries[i])
                for r in rows_j:
                    if r == i or r in removed_rows:
                        continue
                    arj = row_entries[r].get(j)
                    if arj is None:
                        continue
                    mult = arj / coef
                    b[r] -= mult * b[i]
                    rr = row_entries[r]
                    for k, value in terms:
                        merged = rr.get(k, 0.0) - mult * value
                        if abs(merged) < _DROP_EPS:
                            if k in rr:
                                del rr[k]
                                col_rows[k].discard(r)
                                delta -= 1
                                # k's degree dropped: it may now be implied-free
                                # via a remaining row, so re-examine it.
                                if k not in agg_in_queue and k not in removed_cols:
                                    agg_in_queue.add(k)
                                    agg_worklist.append(k)
                        else:
                            if k not in rr:
                                col_rows[k].add(r)
                                delta += 1
                            rr[k] = merged
                    del rr[j]
                    delta -= 1
                    # Row r was rewritten: its cached activity is now stale.
                    row_act_cache.pop(r, None)
                for k, _value in terms:
                    col_rows[k].discard(i)
                row_act_cache.pop(i, None)
                removed_rows.add(i)
                removed_cols.add(j)
                col_rows[j].clear()
                reduction_counts["aggregations"] += 1
                changed = True
                agg_fill_delta += delta
                if agg_fill_budget is not None and agg_fill_delta > agg_fill_budget:
                    # Fill has grown past budget: this instance is fill-positive.
                    # Stop aggregating (the composed fill-gate will reject it).
                    agg_aborted = True
                    agg_worklist.clear()
                    agg_in_queue.clear()
                    break

        if v2 and not agg_only:
            signatures: dict[tuple[tuple[tuple[int, float], ...], float], int] = {}
            for j in range(cols):
                if (
                    j in removed_cols
                    or not col_rows[j]
                    or not isfinite(lo[j])
                    or not isfinite(hi[j])
                ):
                    continue
                signature = (
                    tuple(
                        sorted(
                            (i, row_entries[i][j])
                            for i in col_rows[j]
                            if i not in removed_rows and j in row_entries[i]
                        )
                    ),
                    c[j],
                )
                if not signature[0]:
                    continue
                kept = signatures.get(signature)
                if kept is None or kept in removed_cols:
                    signatures[signature] = j
                    continue
                if not isfinite(lo[kept]) or not isfinite(hi[kept]):
                    continue

                # Exact duplicate columns with equal objective coefficients
                # may be represented by y = x_kept + x_removed. Summed bounds
                # are exact, and postsolve can split any y in that interval
                # back into values satisfying both original boxes.
                records.append(_DuplicateColumn(j, kept, lo[j], hi[j], lo[kept], hi[kept]))
                lo[kept] += lo[j]
                hi[kept] += hi[j]
                for i in list(col_rows[j]):
                    if i in removed_rows:
                        continue
                    row_entries[i].pop(j, None)
                    col_rows[j].discard(i)
                removed_cols.add(j)
                col_rows[j].clear()
                reduction_counts["duplicate_columns"] += 1
                changed = True

    if not removed_rows and not removed_cols:
        return None

    active_rows = [i for i in range(rows) if i not in removed_rows]
    active_cols = [j for j in range(cols) if j not in removed_cols]
    new_col = {j: new_j for new_j, j in enumerate(active_cols)}

    new_indptr = [0]
    new_indices: list[int] = []
    new_data: list[float] = []
    new_b: list[float] = []
    for i in active_rows:
        for j in sorted(row_entries[i]):
            new_indices.append(new_col[j])
            new_data.append(row_entries[i][j])
        new_indptr.append(len(new_indices))
        new_b.append(b[i])

    return PresolveResult(
        rows=len(active_rows),
        cols=len(active_cols),
        indptr=new_indptr,
        indices=new_indices,
        data=new_data,
        b=new_b,
        c=[c[j] for j in active_cols],
        lo=[lo[j] for j in active_cols],
        hi=[hi[j] for j in active_cols],
        objective_offset=objective_offset,
        removed_rows=len(removed_rows),
        removed_cols=len(removed_cols),
        _records=records,
        _active_cols=active_cols,
        _original_cols=cols,
        _reduction_counts=reduction_counts,
    )


def presolve_eq_box(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
    *,
    max_fill: int = 5,
) -> PresolveResult | None:
    """Reduce an equality-plus-bounds LP given as CSR components.

    Returns ``None`` when no reduction applies, so callers can skip the
    rebuild entirely.

    Falls back to the pure-Python reference implementation.
    """
    return _presolve_eq_box_python(
        rows, cols, indptr, indices, data, b, c, lo, hi, max_fill=max_fill
    )


def _result_from_c(raw: tuple[Any, ...]) -> PresolveResult:
    if len(raw) == 12:
        (
            r_matrix,
            r_b_b,
            r_c_b,
            r_lo_b,
            r_hi_b,
            r_offset,
            r_removed_rows,
            r_removed_cols,
            r_records_raw,
            r_active_cols_b,
            r_original_cols,
            r_counts_raw,
        ) = raw
    else:
        (
            r_matrix,
            r_b_b,
            r_c_b,
            r_lo_b,
            r_hi_b,
            r_offset,
            r_removed_rows,
            r_removed_cols,
            r_records_raw,
            r_active_cols_b,
            r_original_cols,
        ) = raw
        r_counts_raw = None
    r_shape = r_matrix.shape
    reduction_counts = _empty_reduction_counts()
    if r_counts_raw is not None:
        (
            reduction_counts["fixed_columns"],
            reduction_counts["empty_columns"],
            reduction_counts["forcing_rows"],
            reduction_counts["forcing_columns"],
            reduction_counts["empty_rows"],
            reduction_counts["singleton_rows"],
            reduction_counts["column_singletons"],
            reduction_counts["doubletons"],
            reduction_counts["duplicate_columns"],
        ) = r_counts_raw
    else:
        for record in r_records_raw:
            if record[0] == 0:
                reduction_counts["singleton_rows"] += 1
            elif record[0] == 1:
                reduction_counts["doubletons"] += 1
    for record in r_records_raw:
        if record[0] == 5:
            reduction_counts["aggregations"] += 1
        elif record[0] == 6:
            reduction_counts["net_aggregations"] += 1
    return PresolveResult(
        rows=r_shape[0],
        cols=r_shape[1],
        indptr=[],
        indices=[],
        data=[],
        b=_unpack_dbls(r_b_b),
        c=_unpack_dbls(r_c_b),
        lo=_unpack_dbls(r_lo_b),
        hi=_unpack_dbls(r_hi_b),
        objective_offset=r_offset,
        removed_rows=r_removed_rows,
        removed_cols=r_removed_cols,
        _records=_LazyRecordList(r_records_raw),
        _active_cols=_unpack_ints(r_active_cols_b),
        _original_cols=r_original_cols,
        _reduction_counts=reduction_counts,
        _matrix=r_matrix,
    )


def presolve_matrix(
    matrix: Any,
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
    *,
    max_fill: int = 5,
    algorithm: str = "auto",
) -> PresolveResult | None:
    """Fast presolve accepting a CSRMatrix directly.

    When the C accelerator is available, a native opportunity scan routes
    only high-yield V2 problems through Python. Low-yield problems stay on
    the C path and avoid Python marshalling and fixpoint overhead.

    Returns a PresolveResult whose ``_matrix`` attribute holds the reduced
    CSRMatrix (avoids rebuilding from components). Falls back to the
    list-based path when the C extension is unavailable.
    """
    if _c_presolve is not None:
        packed_b = _pack_dbls(b)
        packed_c = _pack_dbls(c)
        packed_lo = _pack_dbls(lo)
        packed_hi = _pack_dbls(hi)

        def _staged(res: PresolveResult | None) -> PresolveResult | None:
            # General equality-row aggregation lives only in the Python reducer;
            # inject it as a composed re-stage over whatever the C path produced.
            # Aggregation is validated for the IPM route only: aggregated
            # shapes raise DS pivot counts and can push PDHG past its
            # iteration limit even when fill-negative, so explicit
            # simplex/dual_simplex/pdhg requests skip the re-stage.
            if res is not None and algorithm in ("ipm", "auto") and _agg_restage_enabled():
                res = _maybe_aggregate(res, max_fill)
            if res is not None and algorithm in ("pdhg", "auto") and _netagg_enabled():
                res = _maybe_netaggregate(res, max_fill)
            return res

        if _v2_enabled() and _c_v2_candidates is not None:
            candidate_rows, candidate_cols, *_ = _c_v2_candidates(
                matrix,
                packed_b,
                packed_c,
                packed_lo,
                packed_hi,
            )
            rows_val, cols_val = matrix.shape  # type: ignore[attr-defined]
            if _v2_worth_python_pass(candidate_rows, candidate_cols, rows_val, cols_val):
                if _v2_native_enabled() and _c_presolve_v2 is not None:
                    raw = _c_presolve_v2(
                        matrix,
                        packed_b,
                        packed_c,
                        packed_lo,
                        packed_hi,
                        max_fill,
                    )
                    return _staged(None if raw is None else _result_from_c(raw))
                indptr, indices, data = matrix.to_components()  # type: ignore[attr-defined]
                return _staged(
                    _presolve_eq_box_python(
                        rows_val,
                        cols_val,
                        indptr,
                        indices,
                        data,
                        b,
                        c,
                        lo,
                        hi,
                        max_fill=max_fill,
                    )
                )

        # Raw opportunity gate is closed: run the classic singleton/doubleton
        # cascade.
        raw = _c_presolve(
            matrix,
            packed_b,
            packed_c,
            packed_lo,
            packed_hi,
            max_fill,
        )
        if raw is None:
            return None
        first_result = _result_from_c(raw)
        orig_rows, orig_cols = matrix.shape  # type: ignore[attr-defined]
        # Second-fixpoint re-stage (default on): the raw gate above is scored
        # BEFORE the classic cascade, so it cannot see the fixed/forcing/empty/
        # column-singleton opportunities that cascade creates. When classic made
        # meaningful progress, run the combined V2 fixpoint on the reduced
        # problem and compose it onto the classic reduction. Staying with a
        # rebuilt reduced problem (rather than continuing the classic build in
        # place) reaches strictly the deeper fixpoint the loss census measured.
        # LINPROGX_PRESOLVE_FIXPOINT=0 restores the byte-identical classic path.
        if (
            _fixpoint_enabled()
            and _v2_enabled()
            and _fixpoint_worth_restage(
                first_result.removed_rows, first_result.removed_cols, orig_rows, orig_cols
            )
        ):
            second: PresolveResult | None
            if _v2_native_enabled() and _c_presolve_v2 is not None:
                reduced_matrix = first_result._matrix
                r1_b = _pack_dbls(first_result.b)
                r1_c = _pack_dbls(first_result.c)
                r1_lo = _pack_dbls(first_result.lo)
                r1_hi = _pack_dbls(first_result.hi)
                second_raw = _c_presolve_v2(reduced_matrix, r1_b, r1_c, r1_lo, r1_hi, max_fill)
                second = None if second_raw is None else _result_from_c(second_raw)
            else:
                r1_indptr, r1_indices, r1_data = first_result._matrix.to_components()
                second = _presolve_eq_box_python(
                    first_result.rows,
                    first_result.cols,
                    r1_indptr,
                    r1_indices,
                    r1_data,
                    list(first_result.b),
                    list(first_result.c),
                    list(first_result.lo),
                    list(first_result.hi),
                    max_fill=max_fill,
                )
            # Accept the deeper reduction only when it is substantial. A tiny
            # second reduction is not worth the pass and can slow the solver
            # down (PDHG/IPM conditioning), so fall back to the classic result.
            if second is not None and _fixpoint_reduction_is_substantial(
                second.removed_rows, second.removed_cols, first_result.rows, first_result.cols
            ):
                return _staged(_compose_reductions(first_result, second))
        return _staged(first_result)

    # Fallback: extract components and use Python path
    rows_val, cols_val = matrix.shape  # type: ignore[attr-defined]
    indptr, indices, data = matrix.to_components()  # type: ignore[attr-defined]
    return _presolve_eq_box_python(
        rows_val, cols_val, indptr, indices, data, b, c, lo, hi, max_fill=max_fill
    )


def postsolve_x(x_reduced: list[float], reduction: PresolveResult) -> list[float]:
    """Reconstruct the full solution from a reduced solution."""
    x_full = [0.0] * reduction._original_cols
    for new_j, j in enumerate(reduction._active_cols):
        x_full[j] = float(x_reduced[new_j])
    for record in reversed(reduction._records):
        if isinstance(record, _FixedVar):
            x_full[record.column] = record.value
        elif isinstance(record, _Doubleton):
            x_full[record.eliminated] = (
                record.rhs - record.coef_kept * x_full[record.kept]
            ) / record.coef_eliminated
        elif isinstance(record, (_ColumnSingleton, _Aggregation, _NetAggregation)):
            rest = sum(coef * x_full[j] for j, coef in record.terms)
            x_full[record.eliminated] = (record.rhs - rest) / record.coef_eliminated
        else:
            y = x_full[record.kept]
            kept_value = min(
                record.kept_hi,
                max(record.kept_lo, y - record.removed_lo),
            )
            removed_value = y - kept_value
            if removed_value < record.removed_lo:
                removed_value = record.removed_lo
                kept_value = y - removed_value
            elif removed_value > record.removed_hi:
                removed_value = record.removed_hi
                kept_value = y - removed_value
            x_full[record.kept] = kept_value
            x_full[record.removed] = removed_value
    return x_full
