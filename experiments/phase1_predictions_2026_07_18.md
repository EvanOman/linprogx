# Dual Phase-1 derivation: tests of P1/P2/P3 (2026-07-18)

## Verdict

| prediction | verdict | decisive result |
|---|---|---|
| P1 — b-invariance | **CONFIRMED** | DuPh1 was exactly 1,655 in the baseline and all six optimal RHS perturbations. |
| P2 — relevant cost-support scaling / boxed-cost invariance | **FALSIFIED** | Randomly halving relevant cost support increased DuPh1 from 1,655 to 1,929; the support sweep was non-monotone. Adding costs only to boxed columns also moved DuPh1 to 1,808 (+9.2%). |
| P3 — constructive auxiliary start | **FALSIFIED** | The constructive mathematical claim succeeded (zero sign violations; 3,334 warm pivots), but the promised density preservation failed: 113.8 us/pivot versus 90.5 cold (+25.8%), far outside the approximately 5% band. |

**Overall: DERIVATION FALSIFIED (P2 and the density/per-pivot clause of P3 fail).**

The homogeneous auxiliary does identify a dual-feasible basis and that basis cuts the
warm-started DS pivot count into the predicted band. It does not support the claimed
cost-support law, and it does not avoid the previously measured basis-density tradeoff.

## Setup

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`
- linprogx presolved shape: 1,525 rows x 3,868 columns x 23,274 nonzeros
- HiGHS/highspy: 1.14.0, through public model/options/log/basis APIs only
- HiGHS configuration for P1/P2: presolve off, simplex solver, dual strategy,
  dual steepest-edge weight strategy 2, default scaling and tolerances,
  development log level 3
- Certificate tolerance: `eps = 2e-5`
- Runs were foreground and bounded. No network access, package installation, solver
  source inspection, per-problem tuning, C changes, or Git operations were used.
- Reproducible driver: `experiments/phase1_predictions_probe.py`; raw results and logs:
  `/tmp/phase1-predictions/results.json` and `/tmp/phase1-predictions/*.log`.

HiGHS sometimes emitted a second simplex summary for perturbation cleanup. The P1
table aggregates every summary from a run, matching its public total iteration counter.

## P1 — b-invariance

Each perturbation was `b' = b + eps * z`, with `z` drawn from a standard Gaussian
using the listed deterministic seed. All six perturbed problems remained optimal; no
runs were discarded.

| RHS | eps | seed | `||delta b||inf` | status | DuPh1 | DuPh2 | PrPh2 | total |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| original | 0 | — | 0 | optimal | **1,655** | 1,633 | 21 | 3,309 |
| perturbed | 1e-6 | 1,729 | 3.38e-6 | optimal | **1,655** | 2,213 | 16 | 3,884 |
| perturbed | 1e-6 | 2,718 | 3.73e-6 | optimal | **1,655** | 1,971 | 16 | 3,642 |
| perturbed | 1e-3 | 1,729 | 3.38e-3 | optimal | **1,655** | 2,173 | 11 | 3,839 |
| perturbed | 1e-3 | 2,718 | 3.73e-3 | optimal | **1,655** | 1,980 | 12 | 3,647 |
| perturbed | 1e-1 | 1,729 | 3.38e-1 | optimal | **1,655** | 2,095 | 17 | 3,767 |
| perturbed | 1e-1 | 2,718 | 3.73e-1 | optimal | **1,655** | 2,255 | 12 | 3,922 |

Maximum absolute DuPh1 deviation: **0 pivots (0.0%)**. DuPh2 moved freely from
1,633 to 2,255, as allowed by P1. Maximum equality residual across the runs was
`9.77e-8`.

**P1 verdict: CONFIRMED.**

## P2 — cost-support scaling

The presolved columns classify as 3,611 lower-only, 257 boxed, zero upper-only,
and zero free. Of the original 401 nonzero costs, 175 are on relevant one-sided
columns and 226 are on boxed columns.

For the support sweep, a fixed random permutation of the 175 relevant nonzero-cost
entries defined nested 75%, 50%, 25%, and 0% retained-support variants. Costs on
boxed columns were unchanged in this sweep.

| variant | relevant support | boxed support | total support | DuPh1 | DuPh2 | PrPh2 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 175 | 226 | 401 | **1,655** | 1,633 | 21 | 3,309 |
| retain 75% relevant | 131 | 226 | 357 | **2,454** | 1,736 | 19 | 4,209 |
| retain 50% relevant | 88 | 226 | 314 | **1,929** | 1,811 | 19 | 3,759 |
| retain 25% relevant | 44 | 226 | 270 | **18** | 9,353 | 900 | 10,271 |
| retain 0% relevant | 0 | 226 | 226 | **0** | 5,332 | 59 | 5,391 |

The specifically required random-half test moved DuPh1 **up by 274 pivots
(+16.6%)**, rather than dropping it substantially. Across the sweep, DuPh1 is
not monotone in relevant support: reducing support from 175 to 131 increases
DuPh1 by 48.3%, while the collapse happens only between 88 and 44.

For the boxed-only test, signed costs with magnitude on the scale of the median
nonzero relevant cost were added to every boxed column. Everything else was left
unchanged.

| variant | relevant support | boxed support | total support | DuPh1 | delta vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 175 | 226 | 401 | 1,655 | — |
| add cost to boxed columns only | 175 | 257 | 432 | 1,808 | +153 (+9.2%) |

Modified-cost objectives are intentionally not compared. All variants solved to
optimal status.

**P2 verdict: FALSIFIED.** Both the required half-support direction and the
claimed boxed-cost invariance fail; the non-monotone sweep is the stronger
rejection of a simple support-scaling law.

## P3 — constructive auxiliary

The auxiliary used the original presolved `A` and `c`, zero RHS, `[0,1]` for
the 3,611 lower-only columns, and `[0,0]` for all 257 boxed columns. There are
no upper-only or free columns in this reduction.

### Auxiliary solves

| method | status | objective | iterations | wall | `max |x|` | equality residual |
|---|---|---:|---:|---:|---:|---:|
| `scipy.optimize.linprog(method="highs")` | optimal | 0 | 1,958 | 0.150s | 0 | 0 |
| public highspy simplex, presolve off (basis retained) | optimal | 0 | 1,958 | 0.145s | 0 | 0 |

The objectives agree exactly. SciPy and highspy both use HiGHS, so this is an
API-path agreement rather than an independent implementation agreement. I also
attempted `SparseSolver(algorithm="simplex", presolve=False)` on the auxiliary;
it did not complete within the bounded 60-second reasonable-time window and was
terminated by the timeout, so it supplies no additional result.

The retained optimal basis contains 1,070 structural and 455 logical columns
(6,762 basis nonzeros). The highspy basis was valid.

### Direct dual-feasibility verification

Using only the retained column indices and sparse linear algebra, I formed `B*`,
solved

```
y = B*^-T c_B*
d = c - A^T y
```

and checked the original sign conditions at `eps = 2e-5`.

| original column class | columns | required condition | violations | maximum violation |
|---|---:|---|---:|---:|
| lower-only | 3,611 | `d_j >= -eps` | **0** | 6.09e-8 |
| upper-only | 0 | `d_j <= eps` | **0** | 0 |
| free | 0 | `|d_j| <= eps` | **0** | 0 |
| boxed | 257 | no persistent sign condition | not applicable | not applicable |

The maximum absolute reduced cost on a basic structural column was `8.14e-12`.
Thus the constructive mathematical claim is directly verified: **zero violations**.

### Warm start

The environment-gated hook accepted `B*` with no singular repairs, no identity
fallback, and no imported bound-status vector. Both runs used Dantzig leaving,
matching the stated 4,399-pivot cold reference.

| run | status | pivots | wall | us/pivot | FTRAN mean density | BTRAN mean density | reduced objective |
|---|---|---:|---:|---:|---:|---:|---:|
| cold native crash | optimal | 4,399 | 0.398s | 90.5 | 0.2408 | 0.4279 | -72,557,668.26492292 |
| from auxiliary `B*` | optimal | **3,334** | 0.379s | **113.8** | 0.3224 | 0.4949 | -72,557,668.26492676 |

The warm result agrees with the cold reduced objective to `5.3e-14` relative.
After applying the presolve objective offset of 2,420.135077, the reconstructed
original objectives are -72,555,248.12984590 cold and -72,555,248.12984978 warm;
the maximum original equality residuals are `1.77e-7` and `4.77e-7`.

The warm Phase-2 count is inside the predicted 3,300–3,600 interval and is 195
pivots (5.5%) below the 3,529-pivot foreign-transfer reference. If the helper
auxiliary solve is charged as pivot work, its 1,958 iterations plus the 3,334
warm pivots total 5,292; the contract's direct cold/foreign comparison is reported
above using warm DS pivots alone.

The density/per-pivot prediction does not survive:

- warm us/pivot is **25.8% above the measured cold control** (113.8 vs 90.5);
- it is **22.5% above the stated 92.9 us/pivot cold reference**;
- FTRAN mean density rises 33.9% and BTRAN mean density rises 15.7%; and
- wall improves only 4.7% despite a 24.2% pivot reduction.

**P3 verdict: FALSIFIED.** The dual-feasibility and pivot-count subclaims are
confirmed, but the explicit “without foreign-basis densification / within ~5%”
performance claim fails decisively. The auxiliary basis exhibits the same
trade-against mechanism as the foreign Phase-1 basis.

## Follow-up — dense-regime kernels on the B* trajectory

### Question and method

The cold greenbea trajectory had previously killed the env-gated dense-U FTRAN
path, but the B* start is dense from pivot 1. This follow-up asks whether that
different kernel regime changes the result.

The retained P3 basis was reused exactly. Seven repetitions were run for every
arm, with the four B* configurations interleaved and their starting order rotated
between repetitions. Separate cold/B* solve-slice runs were alternated. All runs
were foreground and bounded; no source was read or changed, and no network,
package installation, per-problem tuning, or Git operation was used.

- Driver: `experiments/phase1_dense_regime_followup.py`
- Raw results: `/tmp/phase1-predictions/dense_regime_followup.json`
- Warm-start hook: `LINPROGX_DS_WARM_START=1`
- Dense FTRAN arm: `LINPROGX_DS_FT_DENSE_U=1`
- Scheduler arm: `LINPROGX_CHOL_SCHED=0`
- Attribution runs: `LINPROGX_DS_SOLVE_SLICE=1`
- Each table entry below is the median of seven runs.

### Kernel and scheduler matrix

The default B* result reproduces P3: 3,334 pivots at 113.4 us/pivot and 0.378s.
All arms follow exactly the same 3,334-pivot trajectory.

| B* arm | dense-U | CHOL scheduler | pivots | DS wall | us/pivot | delta vs B* default | auxiliary + DS |
|---|---|---|---:|---:|---:|---:|---:|
| default | off | default | 3,334 | 0.3780s | 113.4 | — | 0.5231s |
| dense-U | on | default | 3,334 | 0.3743s | 112.3 | -0.98% | 0.5194s |
| scheduler 0 | off | 0 | 3,334 | 0.3778s | 113.3 | -0.05% | 0.5229s |
| dense-U + scheduler 0 | on | 0 | 3,334 | **0.3739s** | **112.1** | **-1.10%** | **0.5189s** |

The per-repeat paired median ratio for dense-U alone is 0.9869 (-1.31%),
consistent with the ratio-of-medians result above. Setting `CHOL_SCHED=0` is
flat without dense-U (-0.05%) and changes the dense-U result by only -0.13%.
There is no meaningful scheduler interaction.

Dense-U therefore **does pay on the B* regime, but only by about 1%**. It does
not restore the cold per-pivot rate and does not constitute a material break in
the pivots-times-density conservation.

### Where the B* density cost lands

The slice controls keep all candidate knobs off. Their measured walls are
0.394s cold (4,399 pivots, 89.5 us/pivot) and 0.378s from B* (3,334 pivots,
113.5 us/pivot). Phase times below are normalized per pivot.

| additive phase bucket | cold us/pivot | B* us/pivot | absolute growth | relative growth |
|---|---:|---:|---:|---:|
| pivot row | 17.18 | **24.57** | **+7.39** | +43.0% |
| FTRAN phase | 17.52 | 21.75 | +4.23 | +24.2% |
| BTRAN/rho phase | 17.99 | 20.78 | +2.79 | +15.5% |
| ratio test | 14.50 | 17.44 | +2.94 | +20.3% |
| reduced-cost update | 9.04 | 11.90 | +2.86 | +31.7% |
| LU update | 5.55 | 7.08 | +1.53 | +27.5% |
| refactorization | 5.07 | 7.03 | +1.96 | +38.7% |

The exact nested triangular-solve timers give a complementary view. These
timers sit inside the broader FTRAN and BTRAN/rho phase buckets and must not be
added to the table above.

| isolated solve body | cold us/pivot | B* us/pivot | absolute growth | relative growth |
|---|---:|---:|---:|---:|
| FTRAN body | 19.46 | 22.24 | +2.78 | +14.3% |
| BTRAN body | 11.23 | 14.94 | **+3.70** | **+33.0%** |
| combined solve bodies | 30.70 | 37.16 | +6.46 | +21.0% |

The densification is not confined to one solve kernel. **Pivot-row scatter is
the largest single phase increase**, while the isolated **BTRAN body grows more
than FTRAN** in both absolute and relative terms. Ratio testing, reduced-cost
updates, LU updates, and refactorization also become more expensive. This is a
trajectory-wide density tax, which explains why accelerating FTRAN alone has
only a roughly 1% end-to-end ceiling here.

### Auxiliary cost and honest pipeline wall

The auxiliary was rebuilt seven times through the public highspy path. Every
run was optimal at objective 0 in 1,958 iterations and returned exactly the
retained P3 basis.

| auxiliary timing | median |
|---|---:|
| solver `run()` only | 0.1425s |
| model setup + solve + basis extraction (charged pipeline cost) | **0.1451s** |

The best arm is dense-U plus scheduler 0 at 0.3739s for B*-DS alone, but the
honest pipeline is **0.1451 + 0.3739 = 0.5189s**.

- versus the stated 0.409–0.420s cold baseline: **23.6–26.9% slower**;
- versus the same-pass local cold median of 0.3971s: **30.7% slower**; and
- versus HiGHS at 0.24s: **2.16x slower** (+116%).

Even ignoring auxiliary construction, the best B*-DS median is only 4.2ms
(1.1%) below the reproduced 0.3780s B* baseline. It sits at the lower edge of
the same approximately 0.38–0.40s regime, not in a new performance regime.

### Correctness gates and verdict

Every cold, B*, dense-U, scheduler, and solve-slice arm was optimal. Every B*
arm returned reduced objective `-72557668.26492676`, maximum original equality
residual `4.77e-7`, and maximum bound violation `2.88e-12`. The objective,
status, residual, and bound gates therefore pass at `2e-5` in every arm.

**Dense-regime follow-up verdict: KILLED.** Dense-U has a real but approximately
1% effect on the B* trajectory; scheduler 0 is inert. No arm materially breaks
the approximately 0.38–0.40s pivots-times-density conservation, and charging
the auxiliary makes the best total pipeline 0.5189s—worse than cold and more
than twice HiGHS.
