"""Presolve-opportunity census for selected LP Suite loss instances.

This is a read-only probe over the problem remaining after linprogx's current
presolve. It estimates what a broader presolve pass could remove if every
counted reduction fired once. It does not model cascades.

Dominance test used here
------------------------
The source problems are equality-plus-box LPs. For a conservative standard
sign-based dual-fixing test, each equality is viewed as both ``<=`` and ``>=``.
A boxed column can pass the lower-bound dominance test only when ``c_j > 0``
and its coefficient has the required nonnegative sign in the ``<=`` row and
the required nonnegative sign in the negated ``>=`` row. Likewise, upper-bound
dominance requires ``c_j < 0`` with both opposite signs. For equalities, these
two sign requirements are compatible only for an all-zero column, so this probe
counts only non-fixed boxed empty columns with objective sign pointing to one
bound as dominated. That deliberately undercounts rather than inventing hope
from inequality-only rules that do not apply to free equality duals.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

INSTANCE_NAMES = (
    "greenbea",
    "cre_b",
    "cre_d",
    "pds_10",
    "pds_20",
    "woodw",
    "maros_r7",
    "80bau3b",
    "stocfor3",
    "degen3",
    "pilot87",
)

REL_TOL = 1e-12
ABS_TOL = 1e-12
TAIL_RE = re.compile(
    r"chol_setup tail: m=(?P<m>\d+) tail_start=(?P<start>\d+) "
    r"tail_len=(?P<len>\d+) prefix_flops=(?P<prefix>[0-9.eE+-]+) "
    r"tail_flops=(?P<tail>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class CSRData:
    rows: int
    cols: int
    indptr: list[int]
    indices: list[int]
    data: list[float]


@dataclass
class TailStats:
    dense_tail_len: int
    prefix_flops: float
    dense_tail_flops: float
    factor_flops_estimate: float
    source: str


@dataclass
class CensusResult:
    rows: int
    cols: int
    nnz: int
    fixed_columns: int
    column_singletons: dict[str, int]
    row_activity: dict[str, int]
    dominated_boxed_columns: dict[str, int]
    duplicate_rows: int
    duplicate_columns: int
    projected_rows: int
    projected_cols: int
    projected_nnz: int
    removable_rows: int
    removable_cols: int
    removable_row_pct: float
    removable_col_pct: float
    nnz_reduction_pct: float
    dense_tail: TailStats = field(default_factory=lambda: TailStats(0, 0.0, 0.0, 0.0, "none"))
    projected_dense_tail: TailStats = field(
        default_factory=lambda: TailStats(0, 0.0, 0.0, 0.0, "none")
    )
    dense_tail_len_delta: int = 0
    factor_flops_estimate_change_pct: float = 0.0


def finite(value: float) -> bool:
    return math.isfinite(value)


def close(a: float, b: float, *, rel: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def float_token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    if abs(value) <= ABS_TOL:
        return "0"
    quantum = REL_TOL * max(1.0, abs(value))
    return f"q:{round(value / quantum)}"


def row_entries(matrix: CSRData, row: int) -> list[tuple[int, float]]:
    return [
        (matrix.indices[k], matrix.data[k])
        for k in range(matrix.indptr[row], matrix.indptr[row + 1])
        if abs(matrix.data[k]) > ABS_TOL
    ]


def columns_from_csr(matrix: CSRData) -> list[list[tuple[int, float]]]:
    cols: list[list[tuple[int, float]]] = [[] for _ in range(matrix.cols)]
    for i in range(matrix.rows):
        for j, value in row_entries(matrix, i):
            cols[j].append((i, value))
    return cols


def boundedness_class(lo: float, hi: float) -> str:
    lo_finite = finite(lo)
    hi_finite = finite(hi)
    if lo_finite and hi_finite and close(lo, hi):
        return "fixed"
    if not lo_finite and not hi_finite:
        return "free"
    if lo_finite and not hi_finite:
        return "lower_bounded"
    if not lo_finite and hi_finite:
        return "upper_bounded"
    return "boxed"


def row_activity_bounds(
    entries: list[tuple[int, float]], lo: list[float], hi: list[float]
) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    lower_unbounded = False
    upper_unbounded = False
    for j, aij in entries:
        if aij >= 0.0:
            if finite(lo[j]):
                lower += aij * lo[j]
            else:
                lower_unbounded = True
            if finite(hi[j]):
                upper += aij * hi[j]
            else:
                upper_unbounded = True
        else:
            if finite(hi[j]):
                lower += aij * hi[j]
            else:
                lower_unbounded = True
            if finite(lo[j]):
                upper += aij * lo[j]
            else:
                upper_unbounded = True
    return (-math.inf if lower_unbounded else lower, math.inf if upper_unbounded else upper)


def duplicate_members(signatures: list[tuple[Any, ...]]) -> set[int]:
    first_seen: dict[tuple[Any, ...], int] = {}
    duplicates: set[int] = set()
    for index, signature in enumerate(signatures):
        if signature in first_seen:
            duplicates.add(index)
        else:
            first_seen[signature] = index
    return duplicates


def remove_empty_columns(
    matrix: CSRData, removed_rows: set[int], removed_cols: set[int]
) -> CSRData:
    row_map: dict[int, int] = {}
    for i in range(matrix.rows):
        if i not in removed_rows:
            row_map[i] = len(row_map)

    col_map: dict[int, int] = {}
    for j in range(matrix.cols):
        if j not in removed_cols:
            col_map[j] = len(col_map)

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for i in range(matrix.rows):
        if i in removed_rows:
            continue
        kept: list[tuple[int, float]] = []
        for j, value in row_entries(matrix, i):
            if j not in removed_cols:
                kept.append((col_map[j], value))
        for j, value in sorted(kept):
            indices.append(j)
            data.append(value)
        indptr.append(len(indices))
    return CSRData(len(row_map), len(col_map), indptr, indices, data)


def census_reduced_problem(
    matrix: CSRData, b: list[float], c: list[float], lo: list[float], hi: list[float]
) -> CensusResult:
    cols = columns_from_csr(matrix)
    fixed_cols = {j for j in range(matrix.cols) if boundedness_class(lo[j], hi[j]) == "fixed"}

    singleton_cols: dict[str, int] = {
        "free": 0,
        "lower_bounded": 0,
        "upper_bounded": 0,
        "boxed": 0,
        "fixed": 0,
    }
    free_singleton_rows: set[int] = set()
    free_singleton_cols: set[int] = set()
    for j, entries in enumerate(cols):
        if len(entries) == 1:
            cls = boundedness_class(lo[j], hi[j])
            singleton_cols[cls] += 1
            if cls == "free":
                free_singleton_cols.add(j)
                free_singleton_rows.add(entries[0][0])

    row_forcing: set[int] = set()
    row_redundant: set[int] = set()
    row_infeasible: set[int] = set()
    forcing_cols: set[int] = set()
    for i in range(matrix.rows):
        entries = row_entries(matrix, i)
        lower, upper = row_activity_bounds(entries, lo, hi)
        rhs = b[i]
        lower_ok = not finite(lower) or rhs >= lower - ABS_TOL * max(1.0, abs(rhs), abs(lower))
        upper_ok = not finite(upper) or rhs <= upper + ABS_TOL * max(1.0, abs(rhs), abs(upper))
        if not lower_ok or not upper_ok:
            row_infeasible.add(i)
            continue
        if finite(lower) and finite(upper) and close(lower, upper) and close(rhs, lower):
            row_redundant.add(i)
        elif (finite(lower) and close(rhs, lower)) or (finite(upper) and close(rhs, upper)):
            row_forcing.add(i)
            forcing_cols.update(j for j, _ in entries)

    dominated_lower: set[int] = set()
    dominated_upper: set[int] = set()
    for j, entries in enumerate(cols):
        if entries or boundedness_class(lo[j], hi[j]) != "boxed":
            continue
        if c[j] > ABS_TOL:
            dominated_lower.add(j)
        elif c[j] < -ABS_TOL:
            dominated_upper.add(j)

    row_signatures: list[tuple[Any, ...]] = []
    for i in range(matrix.rows):
        entries = tuple((j, float_token(value)) for j, value in row_entries(matrix, i))
        row_signatures.append((entries, float_token(b[i])))
    duplicate_rows = duplicate_members(row_signatures)

    col_signatures: list[tuple[Any, ...]] = []
    for j, entries in enumerate(cols):
        signature_entries = tuple((i, float_token(value)) for i, value in entries)
        col_signatures.append(
            (
                signature_entries,
                float_token(c[j]),
                float_token(lo[j]),
                float_token(hi[j]),
            )
        )
    duplicate_cols = duplicate_members(col_signatures)

    removed_rows = set()
    removed_rows.update(free_singleton_rows)
    removed_rows.update(row_forcing)
    removed_rows.update(row_redundant)
    removed_rows.update(duplicate_rows)

    removed_cols = set()
    removed_cols.update(fixed_cols)
    removed_cols.update(free_singleton_cols)
    removed_cols.update(forcing_cols)
    removed_cols.update(dominated_lower)
    removed_cols.update(dominated_upper)
    removed_cols.update(duplicate_cols)

    projected_nnz = 0
    for i in range(matrix.rows):
        if i in removed_rows:
            continue
        for j, _ in row_entries(matrix, i):
            if j not in removed_cols:
                projected_nnz += 1

    projected_rows = matrix.rows - len(removed_rows)
    projected_cols = matrix.cols - len(removed_cols)
    nnz = matrix.indptr[-1]

    return CensusResult(
        rows=matrix.rows,
        cols=matrix.cols,
        nnz=nnz,
        fixed_columns=len(fixed_cols),
        column_singletons=singleton_cols,
        row_activity={
            "forcing": len(row_forcing),
            "redundant": len(row_redundant),
            "infeasible": len(row_infeasible),
        },
        dominated_boxed_columns={
            "lower": len(dominated_lower),
            "upper": len(dominated_upper),
            "total": len(dominated_lower) + len(dominated_upper),
        },
        duplicate_rows=len(duplicate_rows),
        duplicate_columns=len(duplicate_cols),
        projected_rows=projected_rows,
        projected_cols=projected_cols,
        projected_nnz=projected_nnz,
        removable_rows=len(removed_rows),
        removable_cols=len(removed_cols),
        removable_row_pct=pct(len(removed_rows), matrix.rows),
        removable_col_pct=pct(len(removed_cols), matrix.cols),
        nnz_reduction_pct=pct(nnz - projected_nnz, nnz),
    )


def pct(part: float, whole: float) -> float:
    return 0.0 if whole == 0 else 100.0 * part / whole


@contextlib.contextmanager
def capture_stderr_fd() -> Any:
    saved = os.dup(2)
    read_fd, write_fd = os.pipe()
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield read_fd
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def parse_tail_stats(text: str) -> TailStats | None:
    matches = list(TAIL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    tail_len = int(match.group("len"))
    prefix = float(match.group("prefix"))
    tail = float(match.group("tail"))
    return TailStats(
        dense_tail_len=tail_len,
        prefix_flops=prefix,
        dense_tail_flops=tail,
        factor_flops_estimate=prefix + tail / 58.0,
        source="chol_debug",
    )


def tail_stats_from_matrix(matrix_obj: Any, rows: int) -> TailStats:
    if hasattr(matrix_obj, "supernode_sizes"):
        old_debug = os.environ.get("LINPROGX_CHOL_DEBUG")
        old_order_eval = os.environ.get("LINPROGX_ORDER_EVAL")
        os.environ["LINPROGX_CHOL_DEBUG"] = "1"
        os.environ.setdefault("LINPROGX_ORDER_EVAL", "1")
        try:
            with capture_stderr_fd() as read_fd:
                matrix_obj.supernode_sizes()
            chunks: list[bytes] = []
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            os.close(read_fd)
            parsed = parse_tail_stats(b"".join(chunks).decode("utf-8", errors="replace"))
            if parsed is not None:
                return parsed
        except Exception:
            pass
        finally:
            if old_debug is None:
                os.environ.pop("LINPROGX_CHOL_DEBUG", None)
            else:
                os.environ["LINPROGX_CHOL_DEBUG"] = old_debug
            if old_order_eval is None:
                os.environ.pop("LINPROGX_ORDER_EVAL", None)
            else:
                os.environ["LINPROGX_ORDER_EVAL"] = old_order_eval
    dense = float(rows) ** 3 / 3.0
    return TailStats(
        dense_tail_len=rows,
        prefix_flops=0.0,
        dense_tail_flops=dense,
        factor_flops_estimate=dense / 58.0,
        source="dense_m_cubed_proxy",
    )


def csr_from_components(matrix: CSRData) -> Any:
    from linprogx.sparse import csr_matrix

    return csr_matrix(matrix.rows, matrix.cols, matrix.indptr, matrix.indices, matrix.data)


def load_post_current_presolve(
    path: Path,
) -> tuple[CSRData, list[float], list[float], list[float], list[float], Any]:
    from experiments.suite_bench import load_instance
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import from_scipy_sparse

    data = load_instance(path)
    matrix_obj = from_scipy_sparse(data["A_scipy"])
    b = data["b"].tolist()
    c = data["c"].tolist()
    lo = data["lo"].tolist()
    hi = data["hi"].tolist()

    reduction = presolve_matrix(matrix_obj, b, c, lo, hi)
    if reduction is not None:
        matrix_obj = (
            reduction._matrix
            if reduction._matrix is not None
            else csr_from_components(
                CSRData(
                    reduction.rows,
                    reduction.cols,
                    reduction.indptr,
                    reduction.indices,
                    reduction.data,
                )
            )
        )
        b = reduction.b
        c = reduction.c
        lo = reduction.lo
        hi = reduction.hi

    indptr, indices, values = matrix_obj.to_components()
    matrix = CSRData(
        rows=matrix_obj.shape[0],
        cols=matrix_obj.shape[1],
        indptr=[int(value) for value in indptr],
        indices=[int(value) for value in indices],
        data=[float(value) for value in values],
    )
    return matrix, b, c, lo, hi, matrix_obj


def attach_tail_estimates(census: CensusResult, matrix: CSRData, matrix_obj: Any) -> None:
    # Tail estimates use the de-duplicated projected size already computed by
    # the census. The after estimate is intentionally approximate: it isolates
    # the factor-size effect of reducing m and does not model a new symbolic
    # ordering after column removals.
    census.dense_tail = tail_stats_from_matrix(matrix_obj, matrix.rows)
    after_rows = census.projected_rows
    dense = float(after_rows) ** 3 / 3.0
    census.projected_dense_tail = TailStats(
        dense_tail_len=min(census.dense_tail.dense_tail_len, after_rows),
        prefix_flops=0.0,
        dense_tail_flops=dense,
        factor_flops_estimate=dense / 58.0,
        source="projected_m_cubed_proxy",
    )
    census.dense_tail_len_delta = (
        census.projected_dense_tail.dense_tail_len - census.dense_tail.dense_tail_len
    )
    before = float(matrix.rows) ** 3 / 3.0 / 58.0
    after = census.projected_dense_tail.factor_flops_estimate
    census.factor_flops_estimate_change_pct = 0.0 if before == 0.0 else pct(after - before, before)


def run_census(directory: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        path = directory / f"lp_{name}.mat"
        matrix, b, c, lo, hi, matrix_obj = load_post_current_presolve(path)
        census = census_reduced_problem(matrix, b, c, lo, hi)
        attach_tail_estimates(census, matrix, matrix_obj)
        row = asdict(census)
        row["instance"] = name
        rows.append(row)
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "instance      post m      n      nnz  fixed  sing(f/l/u/b)  "
        "rows%   cols%   nnz%  dupR dupC  dom  proj m x n x nnz       tail  flop%"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        singles = row["column_singletons"]
        tail = row["dense_tail"]["dense_tail_len"]
        print(
            f"{row['instance']:<12}"
            f"{row['rows']:>7} {row['cols']:>7} {row['nnz']:>8} "
            f"{row['fixed_columns']:>6} "
            f"{singles['free']:>4}/{singles['lower_bounded']:<4}/"
            f"{singles['upper_bounded']:<4}/{singles['boxed']:<4} "
            f"{row['removable_row_pct']:>6.1f} "
            f"{row['removable_col_pct']:>6.1f} "
            f"{row['nnz_reduction_pct']:>6.1f} "
            f"{row['duplicate_rows']:>5} {row['duplicate_columns']:>4} "
            f"{row['dominated_boxed_columns']['total']:>4} "
            f"{row['projected_rows']:>6} x {row['projected_cols']:<6} x {row['projected_nnz']:<8} "
            f"{tail:>6} {row['factor_flops_estimate_change_pct']:>7.1f}"
        )


def write_json(out: Path, rows: list[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "notes": {
            "scope": "post-current-presolve equality-plus-box census",
            "cascade_modeling": "none; projected size removes counted rows/columns once",
            "duplicate_rows": "same nonzero pattern, coefficient values, and rhs after 1e-12-ish quantization",
            "duplicate_columns": "same nonzero pattern, coefficient values, objective, and bounds after 1e-12-ish quantization",
            "dominance_test": __doc__.split("Dominance test used here", 1)[1].strip(),
            "projected_factor_flops": "current tail from chol debug when available; projected change uses m^3/3 row-removal proxy divided by dense-tail speed factor 58",
        },
        "instances": rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("/tmp/lpsuite"))
    parser.add_argument("--out", type=Path, default=Path("probe_out/presolve-census.json"))
    parser.add_argument("--instances", nargs="*", default=list(INSTANCE_NAMES))
    args = parser.parse_args(argv)

    rows = run_census(args.directory, tuple(args.instances))
    write_json(args.out, rows)
    print_table(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
