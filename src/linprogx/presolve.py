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
import struct
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

_RATIO_LO = 1e-4
_RATIO_HI = 1e4
_PIVOT_EPS = 1e-12
_DROP_EPS = 1e-15

try:
    _csparse: object = importlib.import_module("linprogx._csparse")
    _c_presolve = _csparse.presolve_eq_box  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - source tree before extension build
    _csparse = None
    _c_presolve = None

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


class _LazyRecordList:
    """Lazily materializes _FixedVar/_Doubleton from raw C tuples.

    Defers the dataclass construction overhead to postsolve time,
    keeping the presolve call itself fast.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: list[tuple[int, ...]]) -> None:
        self._raw = raw

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self):  # type: ignore[override]
        for rec in self._raw:
            if rec[0] == 0:
                yield _FixedVar(rec[1], rec[2])
            else:
                yield _Doubleton(rec[1], rec[2], rec[3], rec[4], rec[5])

    def __reversed__(self):  # type: ignore[override]
        for rec in reversed(self._raw):
            if rec[0] == 0:
                yield _FixedVar(rec[1], rec[2])
            else:
                yield _Doubleton(rec[1], rec[2], rec[3], rec[4], rec[5])


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
    _records: list[_FixedVar | _Doubleton] | _LazyRecordList
    _active_cols: list[int]
    _original_cols: int
    _matrix: Any = field(default=None, repr=False)


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

    records: list[_FixedVar | _Doubleton] = []
    removed_rows: set[int] = set()
    removed_cols: set[int] = set()
    objective_offset = 0.0

    changed = True
    while changed:
        changed = False

        for i in range(rows):
            if i in removed_rows:
                continue
            if not row_entries[i]:
                removed_rows.add(i)
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

    When the C accelerator is available, this avoids converting the matrix
    to Python lists and back, eliminating ~20ms of marshalling overhead.

    Returns a PresolveResult whose ``_matrix`` attribute holds the reduced
    CSRMatrix (avoids rebuilding from components). Falls back to the
    list-based path when the C extension is unavailable.
    """
    if _c_presolve is not None:
        raw = _c_presolve(
            matrix,
            _pack_dbls(b),
            _pack_dbls(c),
            _pack_dbls(lo),
            _pack_dbls(hi),
            max_fill,
        )
        if raw is None:
            return None
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
        r_shape = r_matrix.shape
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
            _matrix=r_matrix,
        )

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
        else:
            x_full[record.eliminated] = (
                record.rhs - record.coef_kept * x_full[record.kept]
            ) / record.coef_eliminated
    return x_full
