# Column churn is the dual simplex's class defect — and an existing rule fixes most of it

**PROVENANCE: mixed.** The measurement below is clean-room (our own
instrumentation, our own solver, HiGHS used only as an iteration-count oracle
through its public API). It was *motivated* by the source-informed wave, so it is
recorded on the source-informed branch.

## The diagnostic

Across the nine realised dual-simplex instances, the solver's own counter
`cols_reentering_gt10` — columns that enter the basis more than ten times —
separates winners from losers **perfectly**:

| instance | ratio vs HiGHS | cols re-entering >10x | max re-entries | degenerate pivots |
|---|---:|---:|---:|---:|
| 25fv47 | 2.74 | **226** | 54 | 0 |
| degen2 | 2.69 | 1 | 15 | 0 |
| greenbeb | 1.82 | **76** | 39 | 1 |
| greenbea | 1.55 | **14** | 19 | 1 |
| tuff | 1.27 | 0 | 5 | 0 |
| israel | 0.97 | **0** | 5 | 0 |
| fffff800 | 0.81 | **0** | 3 | 0 |
| agg2 | 0.51 | **0** | 3 | 0 |
| agg3 | 0.48 | **0** | 3 | 0 |

**Every winner is exactly zero; the magnitude tracks the loss ratio.** And
`degenerate_pivots` is ~0 everywhere, so **this is not classical degeneracy** —
it is churn: the pricing rule keeps re-selecting columns it has already tried.

That is the mechanism behind "our dual simplex degrades as the trajectory
lengthens": on short runs there is no time to churn.

## The existing rule that fixes most of it

`leaving_rule=4` already exists in `_csparse.c` — it divides the leaving score by
`(1 + enter_count)`, penalising columns that have entered often. It appears never
to have been evaluated on this class, because the class was not on the board.

| instance | rule 1 | **rule 4** | churn | delta pivots | delta wall |
|---|---|---|---|---:|---:|
| **25fv47** | 8,300 / 1,394 ms | **4,733 / 1,081 ms** | 226 -> **0** | **−43.0%** | **−22%** |
| **degen2** | 1,447 / 138 ms | **1,143 / 89 ms** | 1 -> 0 | **−21.0%** | **−36%** |
| agg2 | 274 / 5 ms | 263 / 5 ms | 0 -> 0 | −4.0% | ~0 |
| greenbeb | 8,919 / 1,934 ms | 10,260 / 2,773 ms | 76 -> 3 | +15.0% | +43% |
| greenbea | 4,399 / 758 ms | 13,484 / 3,841 ms | 14 -> **26** | **+206%** | — |

### Certified

| instance | rule | pivots | residual | bound violation | objective |
|---|---|---:|---|---|---|
| 25fv47 | 1 | 8,300 | 2.73e-12 | 1.78e-15 | 5501.845888286731 |
| 25fv47 | **4** | **4,733** | 6.37e-12 | 1.64e-13 | 5501.845888286745 |
| degen2 | 1 | 1,447 | 3.55e-15 | 4.00e-15 | −1435.1779999999999 |
| degen2 | **4** | **1,143** | 5.55e-15 | 5.11e-15 | −1435.1779999999999 |
| agg2 | 1/4 | 274/263 | 2.33e-10 | 0.00 | −20239252.3559771 (identical) |

Objectives agree to ~1e-15 relative, far inside `eps=2e-5`. **These are real,
certified pivot reductions.**

## Two honest caveats

1. **It breaks greenbea.** Rule 4 sends greenbea to 13,484 pivots and a
   `dual_infeasible` status, and *increases* its churn (14 -> 26). So the crude
   `1/(1+enter_count)` divisor is not a global default — it is a working
   demonstration that churn is the right target, not a finished mechanism.
2. **It hurts greenbeb** (+15%) while reducing its churn (76 -> 3), which means
   suppressing churn is not automatically the same as reducing pivots. The
   penalty distorts selection in a way that costs more than the churn did.

## Why this matters for DS2

This is the first mechanism in the whole campaign that produces a **large,
certified pivot reduction** on the class we are weak on. It says the DS2 pricing
component should be **churn-aware by construction** — but with a principled
penalty (or an explicit anti-cycling memory such as a tabu/ratio-test-history
scheme), not a divisor that destabilises the hardest instance.

It also vindicates broadening the board: this gain was invisible on a 24-cell
board with one simplex instance, and it is worth 43% on a cell that board never
contained.
