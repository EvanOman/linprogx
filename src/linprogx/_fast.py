from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

try:
    _cfast: ModuleType | None = importlib.import_module("linprogx._cfast")
except ImportError:  # pragma: no cover - exercised when building without the extension
    _cfast = None


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if _cfast is not None:
        return float(_cfast.dot(left, right))
    if len(left) != len(right):
        msg = "vectors must have the same length"
        raise ValueError(msg)
    return sum(a * b for a, b in zip(left, right, strict=True))


def pivot(tableau: list[list[float]], pivot_row: int, pivot_col: int, eps: float) -> None:
    if _cfast is not None:
        _cfast.pivot(tableau, pivot_row, pivot_col, eps)
        return

    value = tableau[pivot_row][pivot_col]
    if abs(value) <= eps:
        msg = "pivot value is too close to zero"
        raise ZeroDivisionError(msg)
    tableau[pivot_row] = [entry / value for entry in tableau[pivot_row]]
    width = len(tableau[pivot_row])
    for row_index, row in enumerate(tableau):
        if row_index == pivot_row:
            continue
        factor = row[pivot_col]
        if abs(factor) <= eps:
            row[pivot_col] = 0.0
            continue
        for col in range(width):
            row[col] -= factor * tableau[pivot_row][col]
            if abs(row[col]) <= eps:
                row[col] = 0.0
