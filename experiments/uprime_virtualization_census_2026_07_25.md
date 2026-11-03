# U′ virtualization census — the C-2 funding measurement (2026-07-25)

**Verdict: C-2 IS FUNDED. 21.08% of all U′ element visits in the solve interiors
are dead, and the surviving 78.92% pay the same predicates.** This is the largest
quantified opportunity remaining on greenbea and it sits in the one slice the
composition audit said the missing time must come from.

## Why this slice, and why now

Worker B's composition audit (`endgame_composition_audit_2026_07_25.md`) returned
`COMPOSITION_KILL` with a specific diagnosis:

> Slice monoculture under the authority constraint. Every authority-preserving
> mechanism in the repo attacks the same OWNER slice... The missing 15.323 ms
> exists only in BTRAN/FTRAN, and **no solve-slice member is both exact and
> charged.**

The certified endgame units (−4.89% on-host) mined the OWNER slice, confirming
B's point: that seam is now largely worked out. The triangular-solve interiors
(`btran_rho` 17.05% + `ftran_col` 20.59% = **37.64% of wall**) are the remaining
mass.

Worker D (`endgame_fresh_classes_2026_07_25.md`) proposed **C-2: de-virtualize
the live U′**. The FT state represents U′ *virtually* — the static U filtered at
**solve time** by liveness predicates, plus packed spike columns reached through
a **linked list**. Reading the bodies:

```c
/* lu_ft_btran, static row scatter */
if (ctx->ft_del_stamp[j] < 0) {
    for (p in ut_indptr[j]..) {
        int32_t l = ctx->ut_indices[p];
        if (l == j) continue;
        if (ctx->ft_col_spike[l] >= 0) continue;   /* static col dead */
        z[l] -= ctx->ut_values[p] * zj;
    }
}
/* ... then the spike chain: a serial dependent-load linked list */
for (e = ctx->ft_rowhead[j]; e >= 0; e = ctx->ft_spk_nextrow[e]) {
    int32_t sid  = ctx->ft_spk_col[e];
    int32_t slot = ctx->ft_spk_slot[sid];
    if (ctx->ft_col_spike[slot] != sid) continue;
    if (!lu_ft_row_live(ctx, j, ctx->ft_spk_created[sid])) continue;
    z[slot] -= ctx->ft_spk_val[e] * zj;
}
```

Each spike entry costs **four chained gathers plus a liveness call** before one
axpy. That is the true content of K1's "11 instructions per element at IPC 0.30".

## Measurement

`LINPROGX_DS_UPRIME_CENSUS=1` counts element visits and applications in both
solve bodies. Pure counts — **load-invariant and exact**, no timing, so the
result is unaffected by the contended box.

greenbea, all 4,399 pivots, bit-identical (objective `-72555248.12984592`,
residual 1.769e-07):

| stream | visited | applied | wasted |
|---|---:|---:|---:|
| static U′ elements | 5,006,589 | 3,501,037 | **30.07%** |
| spike-chain entries | 9,025,064 | 7,572,875 | **16.09%** |
| **all U′ visits** | **14,031,653** | 11,073,912 | **21.08%** |

Separately, 44,746 of 68,774 BTRAN row-scatters (65.06%) are skipped wholesale
by `ft_del_stamp[j] >= 0`. That branch is *already* efficient — it is not part
of the opportunity and is excluded from the waste figure above.

## The opportunity is larger than 21%

Two distinct costs are removable, and only the first is captured by the 21.08%:

1. **The 2,957,741 wasted visits** — loaded, chain-chased, branched, discarded.
2. **The predicate cost on the 11,073,912 surviving visits.** Every applied
   element also pays `ft_col_spike[l] >= 0`, or the `ft_spk_col → ft_spk_slot →
   ft_col_spike → ft_spk_created → lu_ft_row_live` chain. An explicitly
   maintained sparse U′/U′ᵀ collapses each inner body to an unconditional
   `x[idx[p]] -= val[p] * xj` over already-live entries.

A naive visit-proportional projection gives `0.2108 × 37.64% = 7.94% of wall`
from (1) alone. (2) is not separately quantified here and is plausibly larger,
because it removes dependent-gather latency from the IPC-0.30 slice rather than
just skipping elements. **This report does not claim a combined number** — the
element-visit model was already shown to be a poor cycle proxy in
`perm_boundary_census_2026_07_25.md`, where it predicted 10.1% and delivered a
different distribution entirely. The honest claim is: the mechanism is funded
and worth building; its size must be measured, not projected.

## How this differs from the recorded kills

- **K3 (dense-mode SIMD bodies, KILLED at 75–87× slower)** performed *exactly
  this synchronization* — "keep U′ synchronized after every committed FT update
  by deleting the old row, replacing the spike column, and appending a dense eta
  row" — but materialized U′ **densely** at 1,525², which K3's own diagnosis
  calls "orders of magnitude of zero multiplies". **The synchronization was
  never the failure; the density was.** A *sparse* synchronized U′ has never
  been built.
- **LS-B (chain interleaving, KILLED)** attacked store/load proximity on the
  scatter streams and never removed a predicate load.
- **LS-A (level scheduling, KILLED)** changed traversal *order*, which forced it
  to re-derive live U′ levels and died on FT slot recycling. C-2 changes the
  *representation of liveness* and keeps order fixed, so it never touches the
  dependency graph that killed LS-A.
- **C1 (factor representation, KILLED at 10.7% slower)** benchmarked alternative
  storage formats on SciPy SuperLU factors with no Forrest-Tomlin machinery at
  all, and explicitly declined to touch the production traversal.

## Risk and cost

This is an **AGENTS.md high-risk area** (memory ownership, pointer lifetimes,
buffer indexing in the C extension; FT update logic). It requires
characterization tests before behaviour changes, and the maintained structure
must be updated in place at every FT commit (delete row, replace spike column,
append eta) — the update already walks exactly those entries to build the spike,
so the maintenance is O(entries touched), not O(nnz).

Bit-identity is achievable in principle: same values, same traversal order, same
arithmetic — only the liveness filtering moves from solve time to update time.
That must be verified, not assumed.

## Reproduction

```bash
cd /home/evan/dev/linprogx-harris-census
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
PYTHONPATH=. LINPROGX_DS_UPRIME_CENSUS=1 \
  uv run python experiments/wholewall_census.py --repeats 1 --no-instrument
```

No network, no solver source read, no per-problem tuning, eps=2e-5 untouched.
Census is env-gated; production behaviour unchanged.
