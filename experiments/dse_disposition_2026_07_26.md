# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

# DSE: the better rule that the board will not let us ship

## The rule comparison, all nine simplex-routed cells

Pivot counts on the shipped solver, presolved public route, all certificate-backed
optimal at eps=2e-5. `leaving_rule=1` (Dantzig, ships) vs `leaving_rule=5` (exact
Forrest-Goldfarb DSE, present and switched off).

| cell | Dantzig/HiGHS | DSE/HiGHS | better |
|---|---:|---:|---|
| 25fv47 | 2.737 | **0.862** | DSE |
| agg2 | 0.513 | **0.489** | DSE |
| agg3 | 0.483 | **0.462** | DSE |
| degen2 | 2.695 | **1.216** | DSE |
| fffff800 | **0.814** | 0.877 | Dantzig |
| **greenbea** | **1.551** | 1.648 | **Dantzig** |
| greenbeb | 1.819 | **1.149** | DSE |
| israel | 0.975 | **0.504** | DSE |
| tuff | 1.270 | **1.052** | DSE |

| statistic | Dantzig | DSE |
|---|---:|---:|
| mean vs HiGHS | 1.429 | **0.918** |
| median | 1.270 | **0.877** |
| **worst case** | 2.737 | **1.648** |
| stdev | 0.804 | **0.374** |
| cells beaten (of 9) | 2 | **7** |

**On average, DSE beats HiGHS on trajectory.** It has half the variance and a
worst case 1.66x better. By every aggregate measure it is the better rule.

## Why it cannot ship as the default

**The 24-cell board contains exactly one simplex-routed cell: greenbea.** Every
other cell routes to the IPM or PDHG, where `leaving_rule` is never read. So
switching the default to DSE would:

- improve **seven** simplex cells, **none of which are on the board**;
- regress **the one cell that is**.

On greenbea DSE is worse on both factors at once: more pivots (4,675 vs 4,399)
*and* ~1.76x the per-pivot cost (253.5 vs 143.8 us), for roughly 1,100 ms against
616 ms. That per-pivot gap has been proven irreducible — four separate attacks on
`pricing_update` were killed, and even a *free* `pricing_update` leaves greenbea
at **1.470x** Dantzig.

Churn narrows the *rule* gap on greenbea to almost nothing — Dantzig+churn 4,283
vs DSE+churn 4,342, **1.4%**, down from 6.3% — but cannot touch the per-pivot
cost, which is what actually decides the cell.

## The uncomfortable conclusion

**The board is selecting the worse rule on the strength of a single lucky
sample.**

greenbea is the cell where Dantzig lands an anomalously good draw
(`why_greenbea_resists_2026_07_26.md`: Dantzig's sensitivity to the right-hand
side is **5x** DSE's, and greenbea is its good day). A benchmark with one
simplex cell, which happens to be that cell, will always prefer the volatile rule.

This is exactly the sampling artefact `docs/BOARD-V2.md` was written to fix.

## Disposition

1. **DSE stays gated and default OFF.** Shipping it now would regress the board.
   This is not a statement that Dantzig is better; it is a statement about what
   the current board measures.
2. **DSE is the headline board-v2 result.** On the proposed 32-cell board with
   nine simplex cells, DSE takes the class from **1.843x to 1.115x** of HiGHS and
   puts 25fv47 **below** HiGHS outright.
3. **The original category question is answered.** The campaign asked whether
   greenbea represents a whole category we cannot solve. It does not: **the
   category is solvable, DSE solves it, and greenbea is not representative of it.**
4. **Wall, not just pivots.** Independently measured on CPU time with a sign-test
   gate: 25fv47 **2.28x CPU win** for DSE, greenbeb **parity** (1.039,
   inconclusive), greenbea a clear loss. DSE pays off in wall wherever it cuts
   pivots hard.

## What would change the answer

A **global** trigger selecting the rule from observed solver state — not from
instance identity — could capture both. None has been measured, and the bar is
high: any trigger fitted to make greenbea pick Dantzig is per-problem tuning
wearing a disguise. Note also that a switch cannot recover pivots already spent,
so an adaptive rule is not simply the per-cell minimum of the table above.
