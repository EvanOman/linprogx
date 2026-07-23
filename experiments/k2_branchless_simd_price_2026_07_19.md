# K2 — branchless SIMD PRICE / pivot-row scan (2026-07-19)

## Verdict

**KILLED.** The branchless AVX2 leaving-row PRICE reduction is real and safe,
but it is not the dossier's 24.8% pivot-row CSR-scatter bucket. It clears the
25% micro-kernel threshold on both cold and B* trajectories, yet saves only
about 2–4% end to end. The attempted AVX2 gather/scatter rewrite of the actual
pivot-row bucket violated the trajectory and failed the certificate, which is
an explicit K2 kill condition.

The safe experiment remains environment-gated by
`LINPROGX_DS_PRICE_AVX2=1`. With the knob absent, the historical scalar loop is
unchanged. The unsafe pivot-row gather path was removed after falsification.

## Contract and kill criterion

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`
- Presolved shape: 1,525 rows x 3,868 columns x 23,274 nonzeros
- Cold and retained P3 B* starts both tested
- Dossier Dantzig path (`leaving_rule=1`) and code-map exact-DSE path
  (`leaving_rule=5`) both tested
- Fixed certificate tolerance: `eps = 2e-5`
- Seven interleaved repetitions per arm, with rotated starting order
- Kill if the attacked scan is less than 25% faster or if any trajectory
  deviates

No network access, Git operation, per-problem tuning, or external solver-source
inspection was used. B* was read from the already-retained offline artifact
`/tmp/phase1-predictions/results.json`; no auxiliary solver was rerun.

## Implementation

The safe AVX2 loop operates on four basis rows at a time:

- contiguous loads of `x_B` and DSE weights;
- AVX2 gathers of the basic columns' lower and upper bounds;
- masked lower/upper violation comparisons;
- branchless sigma selection;
- vector score evaluation for Dantzig or exact DSE; and
- four-lane maxima followed by a deterministic scalar merge.

Each lane retains the first strictly larger score it sees. The final merge
breaks equal scores by the lowest basis-row index, exactly matching the scalar
ascending scan. `LINPROGX_DS_PRICE_VERIFY=1` independently recomputes the
scalar result on every call and raises on a position or sigma mismatch.

The project phase profiler never ticks bucket 1 on the ordinary path, so its
reported `leaving_scan` is always zero and bucket 2 combines PRICE with BTRAN.
The probe therefore adds an environment-gated nested timer
(`LINPROGX_DS_PRICE_SLICE=1`) around PRICE alone.

## Safe PRICE results

The primary numbers below are ratio-of-paired-run medians. Absolute kernel
times are medians across seven runs.

| rule / trajectory | scalar PRICE us/pivot | AVX2 PRICE us/pivot | PRICE reduction | paired end-to-end reduction | pivots |
|---|---:|---:|---:|---:|---:|
| Dantzig cold | 8.846 | 5.475 | **38.1%** | **4.31%** | 4,399 |
| Dantzig B* | 7.840 | 5.559 | **29.8%** | **2.33%** | 3,334 |
| exact DSE cold | 10.838 | 6.021 | **44.4%** | **2.20%** | 4,675 |
| exact DSE B* | 8.971 | 6.156 | **33.1%** | **1.86%** | 2,154 |

The Dantzig cold median wall was 0.5418s scalar and 0.5219s AVX2; B* was
0.5205s scalar and 0.5057s AVX2. These instrumented walls are slower than the
dossier controls, so the paired ratios and per-pivot slice differences are the
appropriate attribution.

The actual pivot-row phase was not accelerated by this safe path: its median
was 22.903 vs 22.541 us/pivot cold and 32.492 vs 31.927 us/pivot from B*, only
about 1.6–1.7% incidental run-to-run movement and far below K2's 25% gate.

## Trajectory and certificate gates

Across scalar and safe AVX2 arms:

- every pivot-pair trajectory hash matched for cold and B*, under both rules;
- every final basis and bound-status SHA-256 matched its scalar control;
- the verification mode checked **14,566 PRICE calls with zero mismatches**;
- every run returned `optimal` with the same pivot count and reduced objective;
- Dantzig cold: objective `-72,557,668.26492292`, original equality residual
  `1.77e-7`, bound violation `3.86e-12`;
- Dantzig B*: objective `-72,557,668.26492676`, original equality residual
  `4.77e-7`, bound violation `2.88e-12`.

All residual and bound gates are comfortably below `eps = 2e-5`.

## Actual pivot-row AVX2 falsifier

I also replaced the CSR pivot-row accumulation under the same gate with AVX2
loads/gathers and scalar-order scatters. Within each row, columns and pattern
insertion remained in source order; a second version explicitly fell back to
scalar accumulation for repeated columns inside a four-entry block.

Both versions failed immediately and identically on the cold warmup:

| result | value |
|---|---:|
| status | `numerical_error` |
| original equality residual | `2.021e9` |
| original bound violation | `2.086e10` |

This is a decisive trajectory/certificate failure. The experiment did not time
or project the invalid kernel, and the unsafe code was removed.

## Flip projection

Use the measured safe PRICE savings with the dossier's uncontaminated baseline
rates, rather than the slower instrumented walls.

### Cold path

```
PRICE saving       = 8.846 - 5.475 = 3.371 us/pivot
projected rate     = 90.5 - 3.371 = 87.129 us/pivot
projected DS wall  = 4,399 * 87.129 us = 0.3833s
required flip rate = about 54 us/pivot
remaining gap      = 33.1 us/pivot
```

The safe AVX2 PRICE selection supplies only 9.2% of the required 36.5
us/pivot reduction.

### B* path

```
PRICE saving             = 7.840 - 5.559 = 2.281 us/pivot
projected dense rate     = 113.4 - 2.281 = 111.119 us/pivot
projected B* DS wall     = 3,334 * 111.119 us = 0.3705s
plus retained auxiliary  = 0.3705 + 0.1451 = 0.5156s
required dense rate      = about 65-72 us/pivot
required auxiliary       = below 0.05s
```

The kernel leaves the dense trajectory 39.1 us/pivot above even the looser
72-us target, before the auxiliary-cost miss.

## Validation

- Prescribed initial build:
  `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation`
- Prescribed reinstall after every C edit:
  `UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation`
- Focused dual tests: **37 passed**
- Offline project gate: Ruff lint and format, ty, Bandit, and coverage tests
  all passed
- Full tests: **522 passed, 7 skipped**, coverage **89.16%** (floor 85%)
- `pip-audit` was deliberately not run because it accesses an external
  vulnerability index and the campaign forbids all network access

Reproducer: `experiments/k2_branchless_price_probe.py`

Raw measurements: `experiments/k2_branchless_price_results_2026_07_19.json`

**Final K2 verdict: KILLED.** The exact branchless PRICE sub-kernel is useful
but too small; the actual 24.8% pivot-row rewrite fails the mandatory
trajectory/certificate gate.
