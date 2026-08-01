# C-2 phase 2 (spike-chain de-virtualization) — KILLED on funding (2026-07-25)

**Verdict: KILLED.** The latency hypothesis is **confirmed** — spike-chain
elements cost **4.09×** what static elements cost — but the entire U′ traversal
in BTRAN is only **2.64% of the solve run**. Erasing all of it, chain included,
cannot fund the 13.5% still required.

## The correctly-framed question

`uprime_compaction_falsifier_2026_07_25.md` killed phase 1 and, in doing so,
established that work-counts do not predict cycles here. It explicitly
reframed phase 2:

> Its case must NOT rest on the 16.09% spike waste or on removing predicates —
> by this result those are worth little. It can only pay if it removes
> **dependent-gather latency**.

So phase 2 needed a latency measurement, not a work-count. This is it.

## Measurement

rdtsc brackets at **row granularity** (never per element, so overhead is ~0.2%
of the run) separating the two U′ access patterns inside `lu_ft_btran`:

- **static**: a contiguous indexed walk of `ut_indices` / `ut_values`.
- **spike**: a linked list where every step costs `ft_spk_col[e] →
  ft_spk_slot[sid] → ft_col_spike[slot] → ft_spk_created[sid] →
  lu_ft_row_live(...)` — four serially dependent loads before one axpy.

| path | cycles | element visits | cyc/elem |
|---|---:|---:|---:|
| static | 4,004,586 | 5,006,589 | **0.80** |
| spike | 29,519,879 | 9,025,064 | **3.27** |
| | | | **4.09× ratio** |

*Caveat, stated because it matters:* the cycle counters bracket **BTRAN only**,
while the element counters accumulate from both solve bodies. The per-element
figures are therefore understated and the **4.09× ratio is the robust part**.
The absolute cycle totals are BTRAN-only and exact.

## Why this kills it

Two independent conclusions:

1. **The latency hypothesis was right.** Spike elements really do cost ~4× static
   elements, and static elements at 0.80 cyc/elem are effectively free — which
   independently re-confirms the phase-1 kill from a second direction.
2. **The slice is too small to matter.** Total BTRAN U′ inner work is
   `4,004,586 + 29,519,879 = 33,524,465` cycles against a whole-DS-run total of
   **1,266,898,752** cycles measured by the same instrument — **2.64%**.

De-virtualizing the spike chain would, at an optimistic ceiling, recover the
gap between spike and static rates: `9,025,064 × (3.27 − 0.80) ≈ 22.3M cycles`,
i.e. **1.76% of the run**, before charging any of the maintenance the explicit
structure requires. Against the 13.5% still needed, and given that phase 1's
maintenance machinery already measured at zero net benefit, this is not
fundable.

**C-2 is closed at both phases.**

## Where BTRAN's cost actually is — the pointer for the next wave

This is the useful residue. `btran_rho` is **17.05% of wall**, but its U′
traversal is only 2.64% of the run. The cost is therefore in the rest of the
body, and the census exposes a strong candidate:

`rows_seen = 68,774` counts row iterations that survive the `zj == 0.0` test,
across 2,557 BTRAN calls at m = 1,525 — that is **26.9 active rows per BTRAN,
1.76% of m**. The U′ᵀ phase is **hypersparse**, yet:

- the U′ᵀ loop scans all m rows every call to find those ~27, and
- the `L^T` back solve that follows runs over **all m rows with no zero-skip at
  all**.

Meanwhile the adaptive route (`s_nnz * 4 > m * s_cnt`) selects the dense-staged
body from the **output** density (59–94% dense), not from this intermediate
sparsity — fill arrives in the later phase. A hybrid (sparse U′ᵀ traversal via
the existing GP reach machinery, dense `L^T`) is therefore a distinct,
unexplored mechanism, and unlike C-2 it targets the part of BTRAN that actually
holds the time.

This report does **not** claim it. It is a measured pointer, and it must be
funded by its own direct cycle measurement — the lesson of the three failed
work-count models this session.

## Reproduction

```bash
PYTHONPATH=. LINPROGX_DS_UPRIME_CENSUS=1 \
  uv run python experiments/wholewall_census.py --repeats 1 --no-instrument
```

Bit-identical throughout: 4,399 iterations, objective `-72555248.12984592`.
Instrumentation env-gated; production behaviour unchanged.
