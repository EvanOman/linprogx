# Netlib DFL001

Source: SuiteSparse Matrix Collection, `LPnetlib/lp_dfl001`

- Data URL: https://sparse.tamu.edu/mat/LPnetlib/lp_dfl001.mat
- Source page: https://www.cise.ufl.edu/research/sparse/matrices/LPnetlib/lp_dfl001.html
- Original Netlib LP data index: https://www.netlib.org/lp/data/

Problem summary from the source notes:

- Rows: 6,071 matrix rows in the SuiteSparse form
- Variables: 12,230
- Matrix nonzeros: 35,632
- Published Netlib/CPLEX objective: `1.1266396047e+07`
- Domain: real-world airline schedule planning / fleet assignment

The `.mat` file stores `A`, `b`, `c`, lower bounds, upper bounds, and notes. The benchmark solves:

```text
min c^T x
subject to A x = b
           lo <= x <= hi
```

`linprogx` intentionally skips this instance by default. It is a dense educational tableau solver; materializing this sparse model as Python lists would benchmark memory overhead more than the algorithm.
