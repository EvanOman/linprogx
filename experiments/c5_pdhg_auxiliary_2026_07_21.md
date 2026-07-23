# C5 — PDHG-approximated auxiliary (2026-07-21)

## Verdict: KILLED

Bounded sparse PDHG on the homogeneous Phase-1 auxiliary does not produce a
basis guess good enough to unlock the K7/K9 pipeline. The best arm is the
500-iteration budget: PDHG + G2 crossover + certificate-backed warm dual
simplex takes **0.6841s raw median** on this host, versus a **0.5693s** local
cold median. That is **20.2% slower**, not the required >=10% improvement.

Normalizing by the same-pass cold ratio to the dossier's 0.370s baseline gives
**0.4446s** end to end. This misses C5's 0.333s live gate by 0.1116s (33.5%)
and misses the board-flip target of 0.3034s (-18%) by 0.1412s (46.5%). The
pre-registered kill criterion fires.

## Falsifier stated up front

C5 is LIVE only if some globally bounded PDHG budget crosses the P3 auxiliary
iterate to a deterministic warm basis, finishes the main solve with its normal
certificate at `eps=2e-5`, and has median charged pipeline wall <=0.333s against
the 0.370s dossier baseline. All PDHG, crossover, and warm-DS time is charged.
The broader greenbea flip needs <=0.3034s (-18%).

The budget is the only swept mechanism variable. The existing PDHG check/restart
cadence is fixed at 2,048 iterations for every arm, avoiding a confounded sweep.

## Setup and mechanism

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- Presolved shape: 1,525 rows x 3,868 columns x 23,274 nonzeros.
- Auxiliary: `min c'x`, `Ax=0`; the 3,611 lower-only columns use `[0,1]` and
  the 257 boxed columns are fixed at `[0,0]` (the exact P3/K7 auxiliary).
- Approximation: existing native `solve_eq_box_pdhg`, hard-capped at budgets
  `{1,2,5,10,20,50,100,200,500,1000,2000,5000}`.
- Crossover: the existing G2 deterministic iterate-prioritized Bixby crash,
  unchanged. It admits a stable structural singleton only when its pivot is at
  least half the column maximum, then fills uncovered rows with artificials.
- Main solve: existing diagnostic basis/status injection, Dantzig dual simplex,
  and the normal optimal exit certificate. No solver implementation changed.
- Timing: seven cold controls and five complete repetitions per budget,
  foreground. The table reports medians.
- Raw JSON: `/tmp/c5-pdhg-auxiliary/results.json`.
- Driver: `experiments/c5_pdhg_auxiliary_probe.py`.

No network access was used; all `uv` invocations additionally set
`UV_OFFLINE=1`. No solver source was read, no solver source was changed, no
per-problem rule was introduced, and no Git operation was used.

## Budget sweep

The final column normalizes each raw pipeline by `0.370 / 0.569334 = 0.649882`
to the dossier host. `support` counts auxiliary iterate entries greater than
`1e-8`. Every row was optimal and certificate-clean in all five repetitions.

| PDHG budget | support | PDHG max primal resid | warm DS pivots | PDHG raw | crossover raw | DS raw | pipeline raw | pipeline normalized |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 1.00 | 4,861 | 0.0049 | 0.0739 | 0.8794 | 0.9517 | 0.6185 |
| 2 | 115 | 1.00 | 4,852 | 0.0077 | 0.0728 | 0.6247 | 0.7041 | 0.4576 |
| 5 | 168 | 1.00 | 4,988 | 0.0228 | 0.0742 | 0.6722 | 0.7732 | 0.5025 |
| 10 | 403 | 1.00 | 4,988 | 0.0314 | 0.0701 | 0.6635 | 0.7672 | 0.4986 |
| 20 | 59 | 1.00 | 4,759 | 0.0123 | 0.0718 | 0.6314 | 0.7154 | 0.4649 |
| 50 | 465 | 1.00 | 4,852 | 0.0352 | 0.0716 | 0.6297 | 0.7381 | 0.4797 |
| 100 | 289 | 1.00 | 4,870 | 0.0402 | 0.0746 | 0.6482 | 0.7595 | 0.4936 |
| 200 | 73 | 2.20e-2 | 4,950 | 0.0352 | 0.0768 | 0.6378 | 0.7532 | 0.4895 |
| **500** | **958** | **4.72e-3** | **4,341** | **0.0528** | **0.0749** | **0.5570** | **0.6841** | **0.4446** |
| 1,000 | 1,344 | 1.75e-3 | 4,785 | 0.0969 | 0.0753 | 0.5999 | 0.7711 | 0.5011 |
| 2,000 | 1,119 | 1.70e-3 | 4,950 | 0.1790 | 0.0728 | 0.6478 | 0.8996 | 0.5847 |
| 5,000 | 1,095 | 1.32e-3 | 4,938 | 0.4381 | 0.0725 | 0.6953 | 1.1900 | 0.7734 |

The best single repetition is also the 500-iteration arm: 0.6604s raw,
0.4292s normalized. Timing luck therefore cannot reach the 0.333s kill gate.

## Why it dies

The approximate support does not recover K7's auxiliary-basis quality. Every
G2 crash has 1,372 structural and 153 artificial basic columns. K7's exact
native auxiliary basis had 1,464 structural and only 61 artificials and needed
2,399 warm pivots. C5's best approximate basis needs **4,341 pivots**, only 58
(1.3%) fewer than the 4,399-pivot cold start.

This makes the basis-quality failure independently decisive. On the dossier
host, the best arm's DS stage alone normalizes to **0.3620s**. Even granting
free PDHG and free crossover, that is only 2.2% below 0.370s and already above
both the 0.333s C5 gate and 0.3034s flip target. The actually charged normalized
PDHG and crossover costs add 0.0343s and 0.0487s respectively.

More PDHG work does not repair the guess: at 1,000-5,000 iterations warm pivots
rise to 4,785-4,950, while approximation cost grows. At 5,000 iterations PDHG
alone costs 0.4381s raw before crossover or certification.

## Correctness and determinism gates

All 60 warm pipelines returned `optimal` and passed the fixed `eps=2e-5`
certificate checks. The best arm has:

- original objective `-72555248.12984593` versus cold
  `-72555248.12984590` (relative error `4.11e-16`);
- maximum original equality residual `5.78e-8`;
- maximum original bound violation `4.95e-13`; and
- warm hook used, imported statuses used, zero singular repairs, and no
  identity fallback.

Within every budget, all five repetitions matched PDHG-iterate, basis, and
bound-status hashes, DS pivots, and final status. The mechanism is deterministic.

## Build and validation

The required local build completed from the offline cache:

```text
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
```

The driver passes focused Ruff lint, Ruff format check, and Python bytecode
compilation. The full foreground experiment completed and wrote the raw result.

**Verdict: KILLED** — the best certificate-backed end-to-end pipeline is 20.2%
slower than the same-pass cold control, and host-normalizes to 0.4446s versus
the 0.333s live gate and 0.3034s flip target.
