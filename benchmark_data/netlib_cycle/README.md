# Netlib CYCLE

Source: SuiteSparse Matrix Collection, `LPnetlib/lp_cycle`

- Data URL: https://sparse.tamu.edu/mat/LPnetlib/lp_cycle.mat
- Source page: https://sparse.tamu.edu/LPnetlib/lp_cycle
- Original Netlib LP data index: https://www.netlib.org/lp/data/

Problem summary from the source notes:

- Rows: 1,903 matrix rows in the SuiteSparse form
- Variables: 3,371
- Matrix nonzeros: 21,234
- Published Netlib/MINOS objective: `-5.2263930249e+00`
- Bound types: upper and free variables
- Degeneracy note: MINOS reported 1,485 degenerate steps out of 3,156, about 47%

The `.mat` file stores `A`, `b`, `c`, lower bounds, upper bounds, and notes. The benchmark solves:

```text
min c^T x
subject to A x = b
           lo <= x <= hi
```

This benchmark is intentionally different from DFL001. It is smaller, denser, numerically rank deficient, has finite upper/free-variable structure, and has a negative optimum. It is meant to catch sparse-solver tuning that only works for DFL001.
