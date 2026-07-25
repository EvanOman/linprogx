# Hypersparse route override — KILLED (2026-07-25)

**Verdict: KILLED. Forcing the hyper-sparse Gilbert-Peierls path is 51% SLOWER**
(568.6 ms vs ~377 ms). The adaptive route is correctly tuned; the dense-staged
FT bodies are the right choice for greenbea despite genuinely hypersparse
intermediate phases.

## The hypothesis and why it was reasonable

The C-2 phase-2 kill left a measured pointer. Inside `lu_ft_btran`:

| phase | cycles | share of run |
|---|---:|---:|
| U′ᵀ forward solve | 53,336,721 | 4.21% |
| eta application | 18,478,282 | 1.46% |
| L^T back solve | 43,797,461 | 3.46% |

Two loops appeared to be grinding over mostly-zero data:

- Only **26.9 of m=1,525 rows** (1.76%) survive the `zj == 0.0` test in the U′ᵀ
  phase, yet the loop scans all m to find them — 17.3M of that phase's 53.3M
  cycles is scan overhead rather than element work.
- Only **17.78%** of L^T rows carry a nonzero, yet the back solve runs **all m
  rows with no zero-skip at all**.

Meanwhile the adaptive route (`s_cnt > 8 && s_nnz * 4 > m * s_cnt`) selects the
dense-staged bodies from the **output** solution density (59–94% dense), not
from this intermediate sparsity — fill arrives late. And the solver already
contains hyper-sparse GP paths. So: is the threshold simply mis-set for
greenbea?

## Measurement

`LINPROGX_DS_FORCE_GP=1` bypasses the adaptive branch at both call sites and
forces the GP path.

| | baseline (dense-staged) | forced GP |
|---|---:|---:|
| wall median | ~377 ms | **568.6 ms** |
| iterations | 4,399 | 4,402 |
| objective | `-72555248.12984592` | `-72555248.12984592` |
| residual | 1.769e-07 | 1.769e-07 |

**+51% slower.** Also note it is *not* bit-identical — 4,402 vs 4,399 pivots, a
trajectory change from the different accumulation order, though the certified
objective and residual are unchanged.

## Why

The repo already states the mechanism, in the route's own comment:

> when the measured average solution density is high (dense-ish instances like
> woodw, m~1k, sol ~30% of m), the **branchy virtual-adjacency DFS costs more
> than one dense sweep**; use the dense staging path there.

The GP reach computation over the *virtual* U′ adjacency — spike columns from
the packed FT store, static columns liveness-filtered — costs more than the
dense scan it replaces. Sparsity of the intermediate vector is real, but
*discovering* which entries are live is the expensive part, and the dense sweep
avoids that discovery entirely.

This is the same shape as the recorded K3 result (dense sweeps of sparse storage
75–87× slower) inverted: here the sparse traversal loses to the dense one. Both
say the same thing — **the traversal strategy is already at its optimum for this
instance, in both directions.**

## Consequence

The solve-interior lane is now closed from three directions in one session:

1. **C-2 phase 1** — removing 30.07% of static element visits and every static
   predicate: no measurable gain.
2. **C-2 phase 2** — spike-chain de-virtualization: latency hypothesis confirmed
   (4.09× per element) but the whole U′ slice is 2.64% of the run, so it cannot
   fund the gap.
3. **This** — routing the hypersparse phases to the existing sparse path: 51%
   slower.

The remaining L^T cost (43.8M cycles, 3.46% of run, over 17.78%-nonzero rows)
cannot be recovered by the existing GP machinery, and a bespoke sparse L^T would
need exactly the reach computation that makes GP lose here.

`LINPROGX_DS_FORCE_GP` is left as a diagnostic, default off.
