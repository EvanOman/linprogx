# Close-six W2-B — exact-DSE full-cost attribution on 25fv47 / degen2 / greenbeb

**Date:** 2026-07-31 · **Worktree:** `linprogx-close6-w2b-dse-cost`, branch
`close6/w2b-dse-cost`, based at `fc2f86e` · **No Modal spend.**

## Verdict — THE JOB IS KILLED

No measured exact-DSE composition meets any of the three funding gates on
current code. The best certifying arm falls short by **1.84x** on 25fv47,
**1.32x** on degen2 and **1.24x** on greenbeb, and greenbeb's best arm
additionally **regresses the greenbea control to 1.184x**. Every number below
is an end-to-end whole-cell measurement of the production `SparseSolver.solve`
path at HEAD, not a projection from a solve-slice.

| target | gate | best certifying arm | measured | sign test | shortfall | pivot ratio | per-pivot ratio | greenbea control on same arm |
| --- | ---: | --- | ---: | :---: | ---: | ---: | ---: | ---: |
| lp_25fv47 | 0.275 | `dse` (exact DSE, `leaving_rule=5`) | **0.506** | 11/11, p=0.0005 | 1.84x | 0.376 | 1.346 | n/a (DS route) |
| lp_degen2 | 0.268 | `dse_twophase` | **0.353** | 11/11, p=0.0005 | 1.32x | 0.432 | 0.816 | n/a (DS route) |
| lp_greenbeb | 0.705 | `ds2_pair_refac250` | **0.875** | 11/11, p=0.0005 | 1.24x | 0.964 | 0.907 | **1.184 — regressed** |

The DSE gains are real, large and statistically unambiguous — 25fv47 halves and
degen2 drops to 0.35 — and they are still not enough. `0.506` against a `0.275`
gate is a 3.53x cell becoming ~1.79x: a much smaller loss, still a loss.

## Artifacts

| path | contents |
| --- | --- |
| `/tmp/linprogx-close6/wave2/w2b/dse_cost.json` | **required deliverable.** 30 records (instance x arm): pivots, CPU median, refactorizations, FTRAN/BTRAN slice, DS phase timing, status, objective, residual, whole-cell ratio, sign test, gate arithmetic, decision table, opportunity ceiling |
| `/tmp/linprogx-close6/wave2/w2b/w2b_timed_rounds.jsonl` | 330 raw timed observations (11 rounds x 30 instance-arm cells) |
| `/tmp/linprogx-close6/wave2/w2b/w2b_instrument.jsonl` | 30 raw instrumentation records incl. pivot-trace digests |
| `/tmp/linprogx-close6/wave2/w2b/floor.json` | non-solve fixed-cost floor per instance |
| `experiments/w2b_dse_cost.py` | arm table, whole-cell timing worker, route-replicating instrumentation worker |
| `experiments/w2b_drive.py` | paired round driver |
| `experiments/w2b_analyze.py` | medians, exact sign tests, gate arithmetic, decision table |
| `experiments/w2b_floor.py` | opportunity-ceiling probe |

## Method

**Cell.** Each timed sample is the production `SparseSolver(algorithm="auto",
max_iterations=50_000, eps=2e-5, check_interval=50_000).solve(...)` whole cell —
the exact shape `experiments/suite_bench.py:57-70` times, including problem
construction, presolve, route, solve, postsolve, objective and residual.

**Metric.** `time.process_time` (CPU), because the host is shared and was under
load average 19-22 on 12 cores throughout. `OMP/OPENBLAS/MKL_NUM_THREADS=1` in
every arm so parallel BLAS cannot be charged to CPU time asymmetrically.

**Pairing.** 11 rounds. Each round is one fresh subprocess per instance that
runs every arm of that instance back to back, so all arms of a round see the
same host load; arm order rotates with the round index so no arm keeps a
warm-cache advantage. Verdict statistic = per-arm median CPU seconds + exact
one-sided paired binomial sign test against the shipped arm (11/11 => p=0.0005).

**Instrumentation** is a separate un-timed route-replicating run per arm
(`presolve_matrix(matrix, b, c, lo, hi, ...)` -> aggregation gate -> DS/DS2, per
`src/linprogx/sparse.py:171-460`), which returns the raw C result dict:
pivots, refactorizations, `phase_us`, `solve_slice_us`, `pivot_trace_hash`.

**Baseline reproduction (verified first, before any arm ran):** 25fv47 6,948
pivots obj 5501.845888 res 4.5e-12; degen2 1,453 obj -1435.178 res 4.3e-15;
greenbeb 4,320 obj -4302260.261207 res 1.1e-11; greenbea 2,424 obj
-72555248.129846 res 1.5e-08. All four exact against the ledger, and re-verified
byte-identical after every source patch.

## Experimental patches (all env-gated, all default OFF)

Production defaults are untouched: with no environment set the route, pivot
counts, objectives and residuals at HEAD are byte-identical (re-checked after
each rebuild).

| flag | file | purpose |
| --- | --- | --- |
| `LINPROGX_W2B_DS_RULE` (default 1) | `sparse.py:24-32` | override the stall-shortcut leaving rule so exact DSE (5) can be measured end-to-end on the cells the aggregation gate declines |
| `LINPROGX_W2B_FORCE_DS2` (default 0) | `sparse.py:35-42`, applied `sparse.py:~292` | route the shortcut through the DS2 composition even when aggregation declines |
| `LINPROGX_W2B_FORCE_AGG` (default 0) | `presolve.py:~652` | bypass the 20%/5% exchange rate to measure the declined aggregation |
| `LINPROGX_DS2_TRACE_HASH` (default 0) | `_ds2_core.c:155-161, ~1120, ~1400` | FNV-1a digest over the realised DS2 pivot sequence, mirroring the shipped DS oracle at `_csparse.c:16063-16069` — DS2 had no pivot-sequence oracle before this |
| `LINPROGX_DS2_SOLVE_SLICE` (default 0) | `_ds2_core.c:162-186, +4 accumulation sites` | FTRAN/BTRAN counts and time for the DS2 route (counters live on the LUContext and reset at every refactorization) |

`just ci` passes with these applied: 707 passed, 7 skipped, coverage **99.40%**
(floor 98%) — the Wave-0 baseline, preserved by two added tests that assert
each Python hook is off by default and does what it claims when set
(`tests/test_sparse.py`, `tests/test_presolve_coverage.py`).

**Default-route regression check, re-run after every rebuild:** with no
environment set, all four instances reproduce their ledger pivots, objectives,
residuals and backend exactly.

## Arm results

Ratios are candidate/shipped whole-cell CPU medians over 11 paired rounds.
`sign` is wins/trials for candidate < shipped. All arms are deterministic:
identical pivot counts in every round, and bit-identical objectives and pivot
digests across independent replicates.

### lp_25fv47 — gate 0.275, shipped 6,948 pivots @ 0.9083 s CPU

| arm | pivots | CPU median | ratio | sign | p | pivot ratio | per-pivot ratio | status |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | --- |
| shipped (Dantzig+churn) | 6,948 | 0.9083 | 1.000 | — | — | 1.000 | 1.000 | optimal |
| **dse** | 2,614 | 0.4599 | **0.506** | 11/11 | 0.0005 | 0.376 | 1.346 | optimal |
| dse_logical | 2,614 | 0.4638 | 0.511 | 11/11 | 0.0005 | 0.376 | 1.357 | optimal |
| dse_churn | 2,733 | 0.4682 | 0.515 | 11/11 | 0.0005 | 0.393 | 1.311 | optimal |
| ds2_agg_pair | 3,038 | 0.5317 | 0.585 | 11/11 | 0.0005 | 0.437 | 1.339 | optimal |
| ds2 | 2,879 | 0.8049 | 0.886 | 9/11 | 0.0327 | 0.414 | 2.139 | optimal |
| ds2_agg | 3,257 | 0.8240 | 0.907 | 10/11 | 0.0059 | 0.469 | 1.935 | optimal |
| ds2_pair | 3,202 | 0.8633 | 0.950 | 8/11 | 0.1133 | 0.461 | 2.062 | optimal |
| dse_twophase | (34, IPM) | 1.0652 | 1.173 | 4/11 | 0.8867 | — | — | **DS returns `dual_infeasible`** |

`dse_twophase` does not certify on 25fv47: the dual simplex ends
`dual_infeasible` after 2,871 pivots and the whole cell falls through to the
IPM, which solves it in 34 iterations at 1.173x. It is not a DSE composition
result and is excluded from the decision table.

### lp_degen2 — gate 0.268, shipped 1,453 pivots @ 0.1110 s CPU

| arm | pivots | CPU median | ratio | sign | p | pivot ratio | per-pivot ratio |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| shipped | 1,453 | 0.1110 | 1.000 | — | — | 1.000 | 1.000 |
| **dse_twophase** | 628 | 0.0392 | **0.353** | 11/11 | 0.0005 | 0.432 | 0.816 |
| **ds2_pair** | 669 | 0.0392 | **0.353** | 11/11 | 0.0005 | 0.460 | 0.767 |
| ds2 | 662 | 0.0423 | 0.381 | 11/11 | 0.0005 | 0.456 | 0.837 |
| dse | 653 | 0.0431 | 0.389 | 11/11 | 0.0005 | 0.449 | 0.865 |
| dse_churn | 653 | 0.0438 | 0.394 | 11/11 | 0.0005 | 0.449 | 0.878 |
| dse_logical | 653 | 0.0439 | 0.396 | 11/11 | 0.0005 | 0.449 | 0.880 |
| ds2_agg_pair | 1,453 | 0.1130 | 1.019 | 3/11 | 0.9673 | 1.000 | 1.019 |
| ds2_agg | 1,453 | 0.1147 | 1.033 | 3/11 | 0.9673 | 1.000 | 1.033 |

The two `*_agg` arms are unchanged from shipped because **degen2 has no
aggregation to force**: `aggressive_aggregate_for_ds2` returns `candidate is
result`, i.e. `_maybe_aggregate` finds nothing, so the forced-gate flag has
nothing to admit and the route stays on the dual simplex (identical 1,453
pivots, identical digest `15487449636237099328`). This answers W1-B §5's open
question for degen2: the 20%/5% exchange rate is **not** what declines it. For
25fv47 the aggregation does exist and the gate is what declines it —
`ds2_agg` reaches DS2 on a 3,257-pivot trajectory.

### lp_greenbeb — gate 0.705, shipped 4,320 pivots @ 0.9637 s CPU (DS2 route)

| arm | pivots | CPU median | ratio | sign | p | pivot ratio | per-pivot ratio |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| shipped (DS2 + aggregation) | 4,320 | 0.9637 | 1.000 | — | — | 1.000 | 1.000 |
| **ds2_pair_refac250** | 4,166 | 0.8431 | **0.875** | 11/11 | 0.0005 | 0.964 | 0.907 |
| ds2_refac60 | 4,590 | 1.0226 | 1.061 | 1/11 | 0.9995 | 1.062 | 0.999 |
| ds2_pair | 4,867 | 1.0264 | 1.065 | 0/11 | 1.0000 | 1.127 | 0.945 |
| ds2_churn | 4,930 | 1.0569 | 1.097 | 0/11 | 1.0000 | 1.141 | 0.961 |
| ds2_refac250 | 4,746 | 1.0835 | 1.124 | 0/11 | 1.0000 | 1.099 | 1.023 |

### lp_greenbea — control, shipped 2,424 pivots @ 0.3939 s CPU

| arm | pivots | CPU median | ratio | sign | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| shipped | 2,424 | 0.3939 | 1.000 | — | — | 0.986 certified win, preserved |
| ds2_refac250 | 2,424 | 0.3814 | 0.968 | 8/11 | 0.1133 | **not a change** — identical pivot count and digest `17695645300895107540`; the 125-pivot interval never fires on greenbea (28 refactorizations come from other triggers), so this is noise, not a win |
| ds2_pair_refac250 | 2,811 | 0.4665 | 1.184 | 0/11 | 1.0000 | **regression** — greenbeb's best arm |
| ds2_refac60 | 2,967 | 0.4609 | 1.170 | 0/11 | 1.0000 | regression |
| ds2_pair | 2,841 | 0.4877 | 1.238 | 0/11 | 1.0000 | regression |
| ds2_churn | 3,790 | 0.7605 | 1.931 | 0/11 | 1.0000 | regression |

The control is intact under production defaults and every arm that helps
greenbeb hurts greenbea.

## Causal decomposition — where exact DSE's cost goes

`whole-cell ratio = pivot ratio x per-pivot ratio`, both measured.

**The cost is FTRAN volume, and it is instance-dependent.** From
`solve_slice_us` (counts are exact and deterministic):

| instance | arm | FTRAN/pivot | BTRAN/pivot | refactorizations |
| --- | --- | ---: | ---: | ---: |
| 25fv47 | shipped (Dantzig) | **1.17** | 1.01 | 41 |
| 25fv47 | dse | **2.24** | 1.28 | 16 |
| degen2 | shipped (Dantzig) | 1.25 | 1.01 | 10 |
| degen2 | dse | 2.38 | 1.64 | 4 |
| greenbeb | shipped (DS2, DSE) | 2.09 | 1.01 | 50 |
| greenbea | shipped (DS2, DSE) | 2.13 | 1.01 | 28 |

Exact DSE roughly **doubles FTRAN calls per pivot** (+1.07 on 25fv47, +1.13 on
degen2) — the extra `lu_ftran(lu, rho, dse_tau)` in `pricing_update` — and adds
BTRAN traffic as well. This reproduces the greenbea mechanism finding
(`ds2_chuzr_2026_07_26.md:308-322`) on current code and on new instances.

But the *net* per-pivot ratio is not the same sign on both targets:

- **25fv47: 1.346.** DSE's extra solves are not repaid. Pivot ratio 0.376 is
  excellent — better than the historical 2,600-3,000 expectation and below
  HiGHS's own 3,033 — yet 0.376 x 1.346 = 0.506.
- **degen2: 0.816-0.880.** Here DSE is *cheaper* per pivot despite doubling
  FTRANs, because the shorter trajectory needs **4 refactorizations instead of
  10**, so the LU stays sparse and each solve is cheaper. The refactorization
  interaction the brief asked about is real, and it is favourable on degen2 —
  just not by enough.

**greenbeb is not a trajectory cell and its arms confirm it.** Every DS2 arm
except `ds2_pair_refac250` *raises* pivots (4,320 -> 4,590/4,867/4,930/4,746).
The one arm that helps buys 0.875 from a 0.964 pivot ratio and a 0.907 per-pivot
ratio — a genuine 12.5% win, 1.24x short of the gate, and it costs greenbea
18.4%.

## What each target would still need

Holding the other factor at its measured best:

| target | gate | best per-pivot ratio | pivots required at that per-pivot cost | best pivots achieved | HiGHS pivots† |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25fv47 | 0.275 | 1.311 (`dse_churn`) | **1,458** | 2,614 | 3,033 |
| degen2 | 0.268 | 0.767 (`ds2_pair`) | **508** | 628 | ~536 |
| greenbeb | 0.705 | 0.907 (`ds2_pair_refac250`) | **3,357** | 4,166 | ~4,900 |

† HiGHS pivot counts are not measured here. 3,033 for 25fv47 is the recorded
figure (`twophase_rule_matrix_2026_07_26.md`); degen2 and greenbeb are derived
from the Wave-1 synthesis pivot ratios (2.71x, 0.88x) applied to the HEAD
counts, so treat them as approximate context, not evidence.

25fv47 would have to run at **less than half of HiGHS's own pivot count while
paying 1.31x per pivot** — the requirement is not a tuning gap, it is a
different algorithm. Alternatively, at its best measured 2,614 pivots it would
need a per-pivot ratio of 0.731, i.e. DSE would have to become 45% *cheaper*
per pivot than Dantzig while doing twice the FTRAN work.

degen2 is the closest: 0.353 measured against 0.268, needing either 508 pivots
(vs 628 achieved) or a 0.620 per-pivot ratio (vs 0.816). That is a ~24% gap on
either axis, from an arm stack that already composes DSE + two-phase + BFRT +
DS2 scaling + perturbation. No unexercised knob in the tree is worth 24%.

**Opportunity ceiling is not the binding constraint.** The non-solve fixed cost
(problem construction, presolve, postsolve, residual) is only 1.3% of the
25fv47 cell, 4.0% of degen2, 3.1% of greenbeb, 6.7% of greenbea
(`floor.json`). The gates are attackable in principle; the measured arms simply
do not reach them.

## Decision / certificate preservation — the fused two-RHS FTRAN is NOT decision-preserving

This is the check the brief demanded be shown by digest, not narrative. DS2 had
no pivot-sequence oracle, so one was added (`LINPROGX_DS2_TRACE_HASH`,
default off).

| instance | arm | pivots | pivot-trace digest |
| --- | --- | ---: | --- |
| greenbeb | shipped (unfused) | 4,320 | `17459436523699290653` |
| greenbeb | `LINPROGX_DS2_DSE_PAIR=1` | 4,867 | `9723296439091328361` |
| greenbea | shipped (unfused) | 2,424 | `17695645300895107540` |
| greenbea | `LINPROGX_DS2_DSE_PAIR=1` | 2,841 | `10880670726935228400` |
| 25fv47 | `ds2` (unfused) | 2,879 | `5263590802680406244` |
| 25fv47 | `ds2_pair` | 3,202 | `12044966397164627569` |
| degen2 | `ds2` (unfused) | 662 | `17038016673644594655` |
| degen2 | `ds2_pair` | 669 | `11673470582677883935` |

The fused variant changes the pivot sequence on **every** instance. It is a
different trajectory, not a cheaper way to walk the same one — the shared eta
pass changes floating-point accumulation order, which changes CHUZR/CHUZC
decisions. Consequences:

1. Its FTRAN **count** is unchanged (2.08-2.10 per pivot on greenbeb both ways;
   `lu_ftran_pair` counts as two calls by construction,
   `_csparse.c:12357`). Its only mechanical saving is one eta traversal
   instead of two.
2. Its apparent wins and losses are trajectory lotteries: -3.6% pivots on
   greenbeb with `refac=250`, but +12.7% on greenbeb alone, +17.2% on greenbea,
   +11.2% on 25fv47-DS2. This is exactly why it was killed by the greenbeb
   control in `greenbea_goal_v2_2026_07_29.md:138-141` and why its 0.9077 on
   greenbea was never bankable.
3. Every arm is nonetheless **deterministic**: repeated instrumentation runs
   reproduce identical digests, identical pivot counts and bit-identical
   objectives (`-4302260.261206586845`, `5501.845888286745`).

**Certificate preservation.** All certifying arms return `optimal` with
objectives matching the ledger in original units and residuals far inside
eps=2e-5: 25fv47 5501.845888 (res <= 9.1e-12), degen2 -1435.178 (<= 7.1e-15),
greenbeb -4302260.261207 (<= 1.8e-11), greenbea -72555248.129846 (<= 4.4e-08).
The one exception is 25fv47 `dse_twophase`, whose dual simplex returns
`dual_infeasible`; the cell then certifies via the IPM at a different objective
(5501.847509, still within eps but a different solver's answer).

## Secondary findings

1. **`LINPROGX_DS_PHASE1` is inert without `LINPROGX_DS_LOGICAL_FORM=1`**
   (`_csparse.c:14766`: `if (logical_form && ds_phase1_on())`). Any past
   two-phase measurement that set only `DS_PHASE1` measured nothing. The
   `dse_logical` arm is the control that isolates this: logical form alone
   produces a byte-identical trajectory to `dse` on both instances
   (25fv47 2,614 / `17591434483853778631`; degen2 653 /
   `16585645486273647114`), so the entire `dse_twophase` delta is Phase 1.
2. **`LINPROGX_DS2_REFAC=250` is a no-op on greenbea** — identical pivots and
   digest to shipped. greenbea's 28 refactorizations are all triggered by
   growth/fill conditions, never by the 125-pivot interval. Any cadence result
   quoted on greenbea alone is measuring noise.
3. **DS phase timing is not usable as a verdict statistic under load.** The
   `dse` and `dse_logical` arms have provably identical pivot sequences, yet
   their `phase_us` splits differ by up to 50% (e.g. 25fv47 `btran_rho`
   35.6 vs 53.0 us/pivot) because the C profiler uses wall clock. `phase_us`
   is recorded in the artifact for structure only; all verdicts here rest on
   CPU-time medians and exact FTRAN/BTRAN counts.
4. **degen2's exact-DSE trajectory is 653 pivots** and 628 with Phase 1,
   confirming the historical 653/632 figures on current code. 25fv47's is
   **2,614**, confirming the historical 2,613-2,614.

## Reopening conditions

This kill is on *composition arithmetic*, not on exact DSE as a mechanism —
DSE is by far the largest single trajectory lever measured on these two cells
and it is not funded here only because the gates are 0.275/0.268.

- **25fv47** reopens only on a mechanism worth a pivot ratio <= 0.205 at
  unchanged per-pivot cost (<= 1,426 pivots), or on halving DSE's FTRAN volume
  per pivot rather than resharing it. The fused pair is not that mechanism (it
  does not reduce the count).
- **degen2** reopens on any ~24% improvement to either factor, measured on the
  full `dse_twophase` / `ds2_pair` stack rather than against shipped Dantzig.
- **greenbeb** stays closed on its own DS2 arms: the best is 1.24x short and
  regresses the greenbea control by 18.4%. A greenbeb candidate must come from
  outside the DS2 arm set, and must be re-certified against greenbea.
- The **fused two-RHS FTRAN** should not be re-measured as a per-pivot
  optimisation again. It is a trajectory perturbation with an unchanged FTRAN
  count; only a genuinely decision-preserving sharing scheme (identical digest)
  would be a per-pivot mechanism.
