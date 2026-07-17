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
except (ImportError, AttributeError):  # pragma: no cover - source tree before extension build
    _csparse = None
    _c_presolve = None
    _c_presolve_v2 = None
    _c_v2_candidates = None

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
            else:
                raise ValueError(f"unknown presolve record tag {rec[0]}")


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
    _records: list[_FixedVar | _Doubleton | _ColumnSingleton | _DuplicateColumn] | _LazyRecordList
    _active_cols: list[int]
    _original_cols: int
    _reduction_counts: dict[str, int] = field(default_factory=_empty_reduction_counts)
    _matrix: Any = field(default=None, repr=False)


def _v2_enabled() -> bool:
    return os.environ.get("LINPROGX_PRESOLVE_V2", "1") != "0"


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
    combined_records = list(first._records) + [_remap_record(rec, a1) for rec in second._records]
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
) -> PresolveResult | None:
    """Pure-Python reference implementation (kept as the fallback)."""
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

    records: list[_FixedVar | _Doubleton | _ColumnSingleton | _DuplicateColumn] = []
    removed_rows: set[int] = set()
    removed_cols: set[int] = set()
    objective_offset = 0.0
    v2 = _v2_enabled()
    reduction_counts = _empty_reduction_counts()

    changed = True
    while changed:
        changed = False

        if v2:
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

        for i in range(rows):
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

        if v2:
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

        for i in range(rows):
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

        if v2:
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
                    return None if raw is None else _result_from_c(raw)
                indptr, indices, data = matrix.to_components()  # type: ignore[attr-defined]
                return _presolve_eq_box_python(
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
                return _compose_reductions(first_result, second)
        return first_result

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
        elif isinstance(record, _ColumnSingleton):
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
