# Presolve V2 Native Pass Design

## Goal

Port the high-yield Presolve V2 reduction loop from Python to `_csparse.c` while keeping the Python implementation as the bit-equality reference behind `LINPROGX_PRESOLVE_V2_NATIVE=0`.

## Interface

The C extension exposes `linprogx._csparse.presolve_v2(matrix, b, c, lo, hi, max_fill)`.
It accepts the post-V1/public CSRMatrix shape already used by `presolve_matrix` and returns the same tuple shape as the existing native presolve, with one extension: an exact reduction-count tuple.

The returned payload contains:

- reduced `CSRMatrix`
- packed reduced `b`, `c`, `lo`, `hi`
- objective offset
- removed row/column counts
- raw record stream
- active original columns
- original column count
- reduction counters

`presolve.py` decodes this through `_result_from_c`. The reduced matrix stays as a native `CSRMatrix` to avoid the production C-to-Python list round trip; tests materialize `to_components()` for bit-equality assertions.

## Record Stream

The native pass reuses Python `postsolve_x` instead of adding a C postsolve. That is the smaller sound design because the existing Python postsolve is simple, tested, and solver time is dominated by presolve plus solve, not solution reconstruction.

Record tags:

- `0`: fixed variable, `(tag, column, value)`
- `1`: doubleton row, `(tag, eliminated, kept, coef_eliminated, coef_kept, rhs)`
- `2`: column singleton, `(tag, eliminated, coef_eliminated, rhs, terms)`
- `3`: duplicate column, `(tag, removed, kept, removed_lo, removed_hi, kept_lo, kept_hi)`

`_LazyRecordList` materializes these records only when iterated, preserving the existing postsolve contract without paying dataclass construction cost on the presolve hot path.

## Routing

The existing `presolve_v2_candidates` opportunity gate remains in front of V2. High-yield problems use native V2 by default; setting `LINPROGX_PRESOLVE_V2_NATIVE=0` routes the same high-yield cases to the Python reference. `LINPROGX_PRESOLVE_V2=0` still disables V2 entirely.

Low-yield cases continue to use the older native V1 presolve path.

## Correctness Gate

Tests compare native-enabled versus `LINPROGX_PRESOLVE_V2_NATIVE=0` on all 24 LPnetlib fixtures in `/tmp/lpsuite`, plus random LPs. The assertion checks reduced CSR components, vectors, objective offset, active columns, record objects, and representative `postsolve_x` outputs exactly.
