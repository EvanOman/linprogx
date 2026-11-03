# K5 scan+update fusion — MEASURED AT LAST, AND DEAD (2026-07-25)

**Verdict: KILLED. 0.05% of wall.** K5 was the campaign's largest never-measured
angle. It is now measured properly, bit-identically, on a clean A/B, and it is
worth essentially nothing.

## History

`experiments/k1_census_2026_07_19.md`, the kernel campaign's master census,
ranked K5 as its **#3 of 12** angles:

> **K5 (scan+update fusion)** — PRICE (18.9%) + rcost (10.0%) touch the same
> dense alpha vectors twice; fusion attacks memory traffic on ~29% of wall.

`experiments/kernel_campaign_angles_2026_07_19.md` specified it and set the bar:

> Fuse into one pass (update-then-scan) touching the data once. Trajectory
> identical required. **Kill if <20% on the combined slice.**

`docs/HANDOFF.md` (KERNEL CAMPAIGN CLOSED, 2026-07-20) records what happened:

> ABANDONED (opencode zombies killed after 24h hung at build; angles' residual
> value negligible against the proven floor): **K5 fusion**, K6 prefetch, K10
> threaded PRICE, K11 fp32-compare.

So it died of worker-infrastructure failure and was then written off against a
floor that governs a different slice. That unexamined status is what motivated
this whole endgame wave.

## What fusion actually is here

The dependency structure forbids the naive reading. `pivot_row` must FULLY build
alpha before the ratio test can find its argmin, and `rcost_update` needs theta
FROM the ratio test — so a same-pivot three-way fusion is impossible. The real,
available fusion is inside 4g, which walked the **same `alpha_pattern` support
twice, back to back**:

```c
for (ki_ ...) { ... r_ext[j] += theta_d * sigma_d * alpha_j; }   /* pass 1 */
for (ki_ ...) { alpha_scratch[...] = 0.0; alpha_touched[...] = 0; } /* pass 2 */
```

Fused into one pass by clearing unconditionally at the top of the body, before
the filters' `continue`s — which preserves exactly the old clear set, while the
per-column `r_ext +=` is order-independent.

## Bit-identity

Verified with the vector-level trace oracle, not just objective+iterations:

| | vectors | digest |
|---|---:|---|
| fusion OFF | 6,016 | `679168a4baad36d6` |
| fusion ON | 6,016 | `679168a4baad36d6` |

4,399 iterations, objective `-72555248.12984592`, residual 1.769e-07.

## Result

Alternating within-process A/B, 21 pairs, **clean** (worst control drift 0.67%):

| phase | B/A median |
|---|---:|
| rcost_update (treatment) | 0.9953 |
| btran_rho | 1.0025 |
| ftran_col | 1.0059 |
| pivot_row | 1.0046 |
| TOTAL | 1.0014 |

**−0.47% on `rcost_update` = 0.05% of wall.** Against K5's own predeclared bar
(">=20% on the combined slice") it fails by more than two orders of magnitude.

## Why an entire removed traversal is free

The second pass re-touched `alpha_scratch` and `alpha_touched` entries that the
first pass had *just* accessed — they are L1-resident, and the loop is a
sequential walk of `alpha_pattern` with well-predicted branches. On an
out-of-order core that pass is nearly free. Removing "half the memory traffic"
on paper removed almost no time, because the traffic was already cached.

## The wider lesson, now on its fourth confirmation

Work-count models have failed to predict cycles four times this session:

| model | predicted | measured |
|---|---|---|
| branchless pattern collection | fewer mispredicts | **worse** (15.2 vs 9.95 cyc/elem) |
| permutation-boundary streams | 10.1% of wall | different distribution; 1.01% where predicted |
| static U′ compaction | 30.07% of visits removed | **zero** |
| **K5 fusion** | **~29% of wall attacked** | **0.05%** |

**Counts of work do not track time in this solver's hot loops.** What does:
dependent-gather latency (spike elements 3.27 cyc vs static 0.80, a 4.09× gap)
and genuinely non-pipelined arithmetic (the unconditional `vdivpd` in Harris
pass 1, whose removal was worth 2.22% — the one win of the wave).

## The honest irony

This wave's reopening thesis was "K5 was abandoned unmeasured, so the pivot-row
pipeline has unexamined dead work." **The thesis was right and K5 was wrong.**
The pipeline did contain real dead work — 88.08% of Harris blocks computing a
divide whose result is discarded — but it was nothing like what K5 proposed, and
K5's own mechanism is worthless. Reopening the angle was correct; inheriting its
hypothesis would not have been.
