# LS-A — level-scheduled DS BTRAN/FTRAN (2026-07-19)

## Verdict: KILLED

S1 passed strongly: greenbea's triangular-factor DAGs are wide enough to
expose latency-hiding parallelism.  The static schedules average 47-100 nodes
per level, and their critical paths are only 1.0-2.1% of the 1,525 factor
nodes (0.46-0.98% of off-diagonal factor nnz).  Live U' remains wide during
Forrest-Tomlin intervals, averaging 24-26 nodes per exact level.

The implementation is nevertheless **KILLED** by the stated two-attempt rule:

1. Refactor-amortized L/L^T levels preserved results byte-for-byte but made
   the combined BTRAN+FTRAN slice **21.08% slower cold** and **22.85% slower
   from B***.  L has only about one off-diagonal per row on these factors, so
   schedule and interleave overhead dominates the short accumulator chains.
2. Incrementally patched live-U' FTRAN levels produced an invalid dependency
   order under repeated FT slot recycling.  Cold returned
   `dual_unbounded_boxed` at pivot 174 and B* at pivot 581 instead of reaching
   the established certificate-backed optima at 4,399 and 3,334 pivots.

There is therefore no valid active arm for the median-of-nine S3 performance
gate, routed-fixture regression gate, or SHIP-CANDIDATE verdict.

## Gates fixed before measurement

- S1 early kill: mean level width below 4, or critical path above 40% of
  factor nnz.
- S2 identity: preserve each target's subtraction order and require byte-
  identical output; if reassociation proved unavoidable, fall back to equal
  pivot counts, relative objective agreement within `1e-9`, and residuals
  within the fixed `eps=2e-5` certificate gate.
- S3: at least 25% improvement in the combined nested BTRAN+FTRAN slice, at
  least 10% cold end-to-end, no routed/public fixture regression above 1%,
  full pytest green, and byte-identical knob-off behavior.
- Campaign kill: stop after an inadequate S1 structure or two prototype
  attempts that miss S3.

## S1 — dependency-DAG census

### Method

`LINPROGX_DS_LEVEL_CENSUS=1` instruments the existing factor without changing
solve arithmetic.  After every refactorization it constructs source levels
for four triangular orientations:

- `LF`: L forward solve used by FTRAN;
- `LB`: L^T back solve used by BTRAN;
- `UF`: live U' back solve used by FTRAN;
- `UB`: live U'^T forward solve used by BTRAN.

After every accepted FT update it recomputes the exact live-U' levels for the
census only.  The per-update computation deliberately is not a prototype: its
cost is diagnostic overhead and is excluded from all A/B measurements.  It
measures critical-path drift and the fraction of node level labels that differ
from the refactor-time schedule.

The two subprocess-isolated trajectories use the campaign configuration
(`leaving_rule=1`, `expand=1`, `bfrt=0`, `tol=1e-8`).  Cold uses the native
crash.  B* reuses the retained 1,525-column P3 basis from the existing local
`/tmp/phase1-predictions/results.json`; no auxiliary solver was rerun.

Correctness reproduced the established trajectories:

| start | status | pivots | reduced objective | max equality residual | valid factor contexts | FT update samples |
|---|---|---:|---:|---:|---:|---:|
| cold | optimal | 4,399 | -72,557,668.26492292 | 1.03e-7 | 34 | 4,391 |
| B* | optimal | 3,334 | -72,557,668.26492676 | 4.77e-7 | 30 | 3,327 |

`valid factor contexts` is the initial factor plus each ordinary
refactorization represented in the solve.  Final certificate-only factors,
which do not build the DS transposes, are not included.

### Refactor-time structure

`CP/nodes` is mean level count divided by 1,525. `CP/nnz` divides by mean
off-diagonal nnz in that triangular factor.  `mean width` is 1,525 divided by
mean level count.

| start | DAG | levels mean [min,max] | mean width | off-diagonal nnz mean | CP/nodes | CP/nnz |
|---|---|---:|---:|---:|---:|---:|
| cold | LF | 15.24 [1,28] | 100.10 | 1,547.79 | 1.00% | 0.98% |
| cold | LB | 15.24 [1,28] | 100.10 | 1,547.79 | 1.00% | 0.98% |
| cold | UF | 24.00 [8,37] | 63.54 | 5,259.21 | 1.57% | 0.46% |
| cold | UB | 24.00 [8,37] | 63.54 | 5,259.21 | 1.57% | 0.46% |
| B* | LF | 17.90 [10,32] | 85.20 | 1,983.57 | 1.17% | 0.90% |
| B* | LB | 17.90 [10,32] | 85.20 | 1,983.57 | 1.17% | 0.90% |
| B* | UF | 32.43 [19,50] | 47.02 | 5,461.20 | 2.13% | 0.59% |
| B* | UB | 32.43 [19,50] | 47.02 | 5,461.20 | 2.13% | 0.59% |

The early-kill boundaries are nowhere close: the narrowest mean is 47.02,
not below 4, and the largest critical-path/nnz ratio is 0.98%, not above 40%.

### Level-width distributions

Each cell is the number of levels, aggregated across all valid factor
contexts, whose width falls in the named bin.  FTRAN and BTRAN orientations
have the same longest-path length but different source-level width
distributions.

| start / DAG | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold LF | 173 | 127 | 57 | 48 | 35 | 16 | 62 |
| cold LB | 10 | 21 | 20 | 43 | 67 | 107 | 250 |
| cold UF | 51 | 58 | 69 | 104 | 175 | 199 | 160 |
| cold UB | 73 | 96 | 107 | 105 | 97 | 162 | 176 |
| B* LF | 176 | 120 | 80 | 42 | 32 | 27 | 60 |
| B* LB | 20 | 27 | 36 | 42 | 83 | 98 | 231 |
| B* UF | 49 | 64 | 91 | 128 | 297 | 249 | 95 |
| B* UB | 97 | 133 | 151 | 133 | 126 | 179 | 154 |

The skew matters: 55-58% of LF source levels contain fewer than four nodes,
despite LF's very high arithmetic mean width.  LB is much broader (91-94% of
levels have width at least four), and 76-88% of U/U^T levels have width at
least four.  The aggregate mean alone therefore overstates the opportunity in
the sparse L forward pass; attempt 1 exposes that cost.

### Stability across each FT interval

| start | exact live-U' levels mean [min,max] | implied mean width | live edges mean | UF node labels changed vs refactor | UB node labels changed vs refactor |
|---|---:|---:|---:|---:|---:|
| cold | 57.68 [8,154] | 26.44 | 6,755.11 | 34.18% | 13.53% |
| B* | 63.71 [18,132] | 23.94 | 6,596.03 | 40.37% | 13.23% |

The structure remains wide, but a refactor-time U' schedule is not directly
reusable: one third to two fifths of FTRAN-side labels move during the
interval.  BTRAN labels are more stable at about 13%, making patching
plausible, but still not optional.

### Dominant variant selected

The greenbea density history crosses the shipped 25% adaptive threshold after
the first few calls.  The dominant path is consequently the dense-staged
`lu_ft_ftran` / `lu_ft_btran` execution reached through the hyper-sparse API,
not the fresh-factor Gilbert-Peierls-only path.  The slice counters label these
calls `*_sparse` because they time the API wrapper, even when that wrapper
selects dense FT staging internally.

## S2 — prototypes

### Attempt 1: immutable L/L^T schedules

At refactorization, LF and LB nodes were packed by level.  Four independent
row accumulators were stepped round-robin.  Within every target row the entry
order remained the baseline order, so independent rows were reordered without
reassociating any row's floating-point sum.

One alternating, subprocess-isolated falsifier pair was enough to reject the
mechanism before spending on nine repetitions:

| start | arm | BTRAN slice | FTRAN slice | combined slice | wall |
|---|---|---:|---:|---:|---:|
| cold | off | 71.070 ms | 118.845 ms | 189.915 ms | 0.561829 s |
| cold | on | 78.846 ms | 151.095 ms | 229.941 ms | 0.607301 s |
| cold | delta | **+10.94%** | **+27.14%** | **+21.08%** | **+8.09%** |
| B* | off | 68.776 ms | 101.561 ms | 170.337 ms | 0.530244 s |
| B* | on | 79.629 ms | 129.634 ms | 209.263 ms | 0.572592 s |
| B* | delta | **+15.78%** | **+27.64%** | **+22.85%** | **+7.99%** |

**Identity achieved: byte-identical.** Across both arms, cold remained 4,399
pivots with objective bits `-0x1.14c91910f47f4p+26` and identical reduced-x
SHA-256; B* remained 3,334 pivots with objective bits
`-0x1.14c91910f48f6p+26` and identical reduced-x SHA-256.  Both statuses were
`optimal`.

The failure mechanism is structural rather than noisy: L has roughly
1,548-1,984 off-diagonal entries over 1,525 rows.  Most accumulator chains are
too short to amortize level lookup, row conversion, and round-robin control.

### Attempt 2: patched conservative live-U' FTRAN schedule

The second attempt maintained source levels incrementally.  An FT replacement
deletes old dependencies, resets the replaced slot after it is cycled to the
logical end, adds the new spike edges, and propagates only level increases.
FTRAN used four interleaved row gathers, with contributions ordered newest
live spike first and then surviving static columns in reverse order, matching
the intended baseline order for each target.

The correctness smoke test falsified the patch invariant:

| start | off result | active result | failure pivot |
|---|---|---|---:|
| cold | optimal, 4,399 pivots | `dual_unbounded_boxed` | 174 |
| B* | optimal, 3,334 pivots | `dual_unbounded_boxed` | 581 |

The active wall and slice values are invalid performance evidence because the
solver exits thousands of pivots early.  Repeated FT slot recycling creates
live versioned dependencies that the monotone conservative patch does not
fully represent.  This attempt fails before either byte identity or the
fallback trajectory-identity gate can be considered.

## S3 — A/B and validation disposition

The specified two-attempt kill rule fires before the formal alternating
median-of-nine run.  Reporting a nine-run speedup for attempt 2 would reward an
incorrect early exit; running the public regression set with that active knob
would not create a ship candidate.  Accordingly:

| S3 gate | result |
|---|---|
| combined slice improves at least 25% | **fail**: attempt 1 regresses 21-23%; attempt 2 invalid |
| cold wall improves at least 10% | **fail**: attempt 1 regresses 8.09%; attempt 2 invalid |
| active arm preserves certificates/trajectory | **fail** on attempt 2 |
| no routed/public fixture regresses over 1% | not run; no valid active candidate |
| byte-identical knob-off | **pass** on cold and B* hashes, objectives, pivots, status |
| focused solver tests | **60 passed** |
| full coverage-gated pytest | **522 passed, 7 skipped**, 89.16% coverage |
| experiment Ruff checks | pass |

All tests ran with the experimental knobs off.  `pip-audit` was not invoked
because its advisory lookup is network-capable and the campaign forbids all
network access.

## S4 — flip arithmetic with K4

There is no valid LS-A saving to stack.  K4 remains the only live component:

- Campaign-normalized K4 projection: **86.5 us/pivot cold**, versus the
  required 54 us/pivot; the remaining gap is **32.5 us/pivot**.
- Applying K4's measured 7.21% cold wall reduction directly to the dossier's
  90.5 us/pivot reference gives **84.0 us/pivot**.  This is still 30.0
  us/pivot above target.
- Counterfactual attempt-1 stack (not a candidate):
  `90.5 * (1 - 0.0721) * (1 + 0.08094) = 90.8 us/pivot`.  Its overhead erases
  K4's gain and leaves a 36.8 us/pivot gap.
- B*: K4 projects **106.7 us/pivot**, about **0.356 s** for the DS solve;
  charging the unchanged 0.145 s auxiliary gives **0.501 s**.  That remains
  far above the 65-72 us/pivot DS band and the separate sub-0.05 s auxiliary
  target.

## Reproduction and artifacts

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation

# S1, both trajectories; aggregate + raw per-context records
PYTHONPATH=$PWD taskset -c 2 .venv/bin/python experiments/lsa_level_census_run.py
# /tmp/lsa-census/summary.json

# Falsifier A/B driver (the retained raw one-pair runs)
PYTHONPATH=$PWD taskset -c 2 .venv/bin/python experiments/lsa_ab_probe.py --reps 1
# /tmp/lsa-ab/smoke.json and /tmp/lsa-ab/smoke2.json

UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 just test-cov
```

Code and drivers:

- `src/linprogx/_csparse.c`: `LINPROGX_DS_LEVEL_CENSUS=1` census and
  `LINPROGX_DS_LEVELSOLVE=1` failed prototype gate;
- `experiments/lsa_level_census_probe.py`;
- `experiments/lsa_level_census_run.py`;
- `experiments/lsa_ab_probe.py`.

No network access, solver-source inspection, per-problem tuning, Git
operation, or background execution was used.

**Final verdict: KILLED.** The DAG has ample theoretical width, but the stable
L portion is too sparse to pay for scheduling, and the attempted cheap U'
patch is not a correct representation of the versioned FT dependency graph.
