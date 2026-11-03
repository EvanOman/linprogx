# C-2 phase 1: static U′ compaction — KILLED on measurement (2026-07-25)

**Verdict: KILLED.** Eliminating **30.07%** of static U′ element visits and
**every** per-element liveness predicate on that path produced **no measurable
whole-wall improvement** (−0.97% on `btran_rho` against 24.78% control drift).
Bit-identical and correct — just worthless.

## What was built

`experiments/uprime_virtualization_census_2026_07_25.md` measured that 21.08% of
all U′ element visits are loaded, chased and discarded. Phase 1 attacked the
static half: an eagerly maintained, order-preserving compacted copy of the live
static U′ (columns, for FTRAN) and U′ᵀ (rows, for BTRAN).

- Rebuilt by scanning the **original list in index order** and copying only live
  entries, so every floating-point accumulation order is preserved.
- Maintenance is eager and cheap: one FT commit marks exactly one slot `t`
  (`ft_col_spike[t] = sid; ft_del_stamp[t] = upd;`), and the affected sets are
  read straight off the transpose — the columns containing row `t` are
  `ut_indptr[t]..`, the rows containing column `t` are `u_indptr[t]..`. So a
  commit rebuilds O(nnz of one row + one column), roughly 11 short lists per
  pivot against ~3,190 element visits per pivot.
- Solve loops become unconditional: `z[utc_idx[p]] -= utc_val[p] * zj`.

## Correctness: verified bit-identical at vector level

A new trace-hash oracle (`LINPROGX_DS_TRACE_HASH=1`) folds the raw bits of every
BTRAN and FTRAN output vector into an FNV-1a digest — a far stronger gate than
objective + iteration count, which a reordering can preserve while perturbing
intermediates.

| configuration | vectors | digest |
|---|---:|---|
| compaction OFF | 6,016 | `679168a4baad36d6` |
| compaction ON | 6,016 | `679168a4baad36d6` |

4,399 iterations, objective `-72555248.12984592`, residual 1.769e-07 in both.

The census confirms the work really was removed:

| | before | after |
|---|---:|---:|
| static visits | 5,006,589 | **3,501,037** |
| static wasted | 30.07% | **0.00000%** |
| all U′ visits wasted | 21.08% | 11.59% |

## The measurement

Alternating within-process A/B, 21 pairs. **First attempt was invalid**: the gate
cached its `getenv` in a static on first call, so both arms ran the same
configuration and the apparent −10% was pure noise — visible because
`lu_update` and `refactor`, which this change never touches, moved 8–14% too.
After making the gate refresh per solve (as the other units do):

| phase | B/A median |
|---|---:|
| btran_rho (treatment) | 0.9903 |
| ftran_col | 1.0131 |
| TOTAL | 1.0495 |

**−0.97% on `btran_rho`; worst control drift 24.78%.** The box was heavily
contended, so the honest statement is: no effect is detectable, and it is
certainly not the ~1.75% the visit-count model projected.

## Why it fails, and what that implies for C-2 phase 2

The removed predicate is a single `ft_col_spike[l]` load-and-compare on
cache-resident data with a well-predicted branch — near-free on an out-of-order
core. Meanwhile the compacted structures add two arrays of the same size as U,
so the memory footprint grows in exactly the slice K1 measured as latency-bound.

**This is the third time this session an element/predicate-count model has failed
to predict cycles**, after the branchless pattern scan (15.2 vs 9.95
cyc/element — *worse*) and the permutation-boundary streams (predicted 10.1%,
delivered a completely different distribution). The consistent lesson: in these
solve interiors, **counts of work do not track time; dependent-gather latency
does.**

That materially reframes C-2 phase 2 (de-virtualizing the spike chain). Its case
must NOT rest on the 16.09% spike waste or on removing predicates — by this
result those are worth little. It can only pay if it removes **dependent-gather
latency**: the `ft_spk_col[e] → ft_spk_slot[sid] → ft_col_spike[slot] →
ft_spk_created[sid] → lu_ft_row_live(...)` chain of four serially dependent
loads before each axpy, which is the plausible cause of BTRAN's IPC 0.30. A
phase-2 probe should therefore be designed and funded on a **latency/chain-depth**
argument with a direct cycle measurement, not on a work-count argument.

## Disposition

Left in the tree behind `LINPROGX_DS_UPRIME_COMPACT=1`, **default off**. It is
exact and cheap to re-measure on a quiet box should the noise picture change,
but it is not claimed and not shipped.
