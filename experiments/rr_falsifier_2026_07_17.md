# Ranged-row contraction falsifier — 2026-07-17

## Verdict: KILLED

The proposed degree-2 unit-chain contraction does **not** expose the projected
work reduction on the current presolved `pds_10`.  The exact series transform
requires an interior equality row containing only the two adjacent chain arcs.
Current presolve has already eliminated every such row: the presolved `pds_10`
and `pds_20` matrices each contain **zero total-degree-2 rows**, hence zero exact
series chains and zero contractible arcs.

The resulting contracted shapes are identical to baseline.  PDHG iterations
neither rise nor drop: `pds_10` remains exactly **8,576 iterations** and
`pds_20` remains exactly **21,696 iterations**.  The required `pds_10`
`iterations * nnz` gain is 0%, below the 15% LIVE gate.  The small wall changes
are repeat-run noise on identical matrices.

This kills the proposed slack trick in its stated form before the conditioning
question.  Applying it to a path whose interior rows contain side terms is not
an exact reformulation: those terms create an additional consistency equality
as well as a range, and one bounded slack cannot encode both.

## Shapes

Shapes are `rows x columns x nonzeros`.  HiGHS figures are from the SciPy/HiGHS
1.12.0 presolve log captured during this run.

| Fixture | Baseline after linprogx presolve | Contracted | HiGHS presolved | Contracted change in `m+n+nnz` | HiGHS change in `m+n+nnz` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pds_10` | 14,438 x 47,812 x 103,230 | 14,438 x 47,812 x 103,230 | 4,092 x 32,646 x 78,216 | 0.0% | -30.5% |
| `pds_20` | 30,202 x 104,579 x 225,131 | 30,202 x 104,579 x 225,131 | 8,984 x 79,990 x 188,070 | 0.0% | -23.0% |

Structural census from the implemented exact-chain detector:

| Fixture | Total-degree-2 interior rows | Exact maximal chains | Contracted arc columns |
| --- | ---: | ---: | ---: |
| `pds_10` | 0 | 0 | 0 |
| `pds_20` | 0 | 0 | 0 |

This does not contradict the earlier count of 38,852 (`pds_10`) and 85,803
(`pds_20`) degree-2 **columns**.  Those columns are graph arcs, but their row
junctions have side terms.  For `pds_10`, 682 rows have degree two within the
degree-2-column subgraph, yet every one also contains one to three degree-3
columns; none is a two-term series equation.

## PDHG and oracle results

Both legs use the existing `SparseSolver(algorithm="pdhg", presolve=False,
eps=2e-5, max_iterations=50_000, check_interval=50_000)` on the already
presolved problem.  `contracted` is a separate solve through the produced
model and reverse map.  Since no exact chain exists, its matrix is identical to
baseline.

| Fixture | Model | Status | Iterations | Wall | Original objective | Absolute delta vs HiGHS | Relative delta vs HiGHS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pds_10` | baseline | optimal | 8,576 | 2.4934s | 26,727,095,083.27369 | 107.27369 | 4.014e-9 |
| `pds_10` | contracted | optimal | 8,576 | 2.4986s | 26,727,095,083.27369 | 107.27369 | 4.014e-9 |
| `pds_20` | baseline | optimal | 21,696 | 14.0435s | 23,821,658,161.13005 | 478.86995 | 2.010e-8 |
| `pds_20` | contracted | optimal | 21,696 | 13.9736s | 23,821,658,161.13005 | 478.86995 | 2.010e-8 |

HiGHS/SciPy oracle objectives were exactly `26,727,094,976` for `pds_10` and
`23,821,658,640` for `pds_20`.  Objective agreement is assessed in the solver's
scale-normalized `2e-5` gate; relative errors are far below it.  Baseline and
contracted objectives are bit-identical in both fixtures.

| Fixture | Model | Original equality residual | Original bound residual |
| --- | --- | ---: | ---: |
| `pds_10` | baseline | 1.323e-5 | 2.217e-12 |
| `pds_10` | contracted/reconstructed | 1.323e-5 | 2.217e-12 |
| `pds_20` | baseline | 1.806e-5 | 2.137e-11 |
| `pds_20` | contracted/reconstructed | 1.806e-5 | 2.137e-11 |

The `iterations * nnz` work proxy is unchanged:

| Fixture | Baseline | Contracted | Projected gain | Measured wall gain |
| --- | ---: | ---: | ---: | ---: |
| `pds_10` | 885,300,480 | 885,300,480 | 0.0% | -0.21% |
| `pds_20` | 4,884,442,176 | 4,884,442,176 | 0.0% | +0.50% |

`pds_20` therefore passes the flat-or-better proxy sentinel, while `pds_10`
fails the required 15% gain.

## Exact combined-bound algebra

For a genuine series path, let its arc variables be `z_1, ..., z_k`.  Interior
row `i` contains no side terms and is

```text
alpha_i z_i + beta_i z_{i+1} = b_i,
alpha_i, beta_i in {-1, +1}.
```

Choose the contracted slack/representative `s = z_1`.  Recursively write

```text
z_i = sigma_i s + delta_i,
sigma_1 = 1,
delta_1 = 0,
sigma_{i+1} = -(alpha_i / beta_i) sigma_i,
delta_{i+1} = (b_i - alpha_i delta_i) / beta_i.
```

Every `sigma_i` is exactly `-1` or `+1`.  The original arc bound
`lo_i <= z_i <= hi_i` maps to an interval on `s`:

```text
if sigma_i = +1:  lo_i - delta_i <= s <= hi_i - delta_i
if sigma_i = -1:  delta_i - hi_i <= s <= delta_i - lo_i
```

The exact combined bound is the intersection

```text
L = max_i(mapped_lo_i),
U = min_i(mapped_hi_i),
s in [L, U].
```

`L > U` proves infeasibility.  At the two endpoint rows, substitute
`z_1 = s` and `z_k = sigma_k s + delta_k`; move the constant endpoint term to
the right-hand side.  The objective transforms exactly as

```text
sum_i c_i z_i
  = (sum_i c_i sigma_i) s + sum_i c_i delta_i,
```

where the last sum is an objective offset.  The reverse map is simply
`z_i = sigma_i s + delta_i`.

The probe exhaustively verified this construction on all 64 independent sign
patterns for a three-arc path, with lower-only, upper-only, and finite bounds.
Every transformed HiGHS optimum, reconstructed equality, bound, and objective
matched the original synthetic LP.

### Why the generalized slack trick is invalid

If an interior row has side expression `q(x)`, the recurrence becomes
`z_{i+1} = sigma z_i + delta + gamma q(x)`.  Bounds then project to coupled
constraints on both `s` and `q(x)`; they are not a constant interval on one
slack.

A two-row witness is enough:

```text
u + z = 0
v - z = 0
0 <= z <= 1
```

Eliminating `z` requires both `u + v = 0` and `-1 <= u <= 0`.  Replacing these
with one ranged equality `u + v = s`, `s in [-1, 1]`, admits `(u,v)=(0,1)`,
which has no feasible original `z`.  The missing consistency equality is
exactly why a single slack cannot make arbitrary degree-2-column contraction
an eq-box shape reduction.

## Gate summary

| Gate | Result |
| --- | --- |
| `pds_10` projected PDHG work gain at least 15% | **FAIL: 0.0%** |
| `pds_10` objective/equality/bound gates | PASS |
| `pds_20` projected flat-or-better | PASS: flat |

Raw probe: `experiments/rr_falsifier_probe.py`.
