# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

# The class result: DSE takes the simplex class from 1.84x to 1.12x of HiGHS,
# and churn removes greenbea's status as the exception

Two independent lines collided productively today.

## The two findings

1. **DS2-CHUZR (delegated) overturned a ledger verdict.** The campaign's
   recorded "exact DSE measured worse" was taken **on greenbea, on the cold
   big-M path**. Swept across all nine simplex-routed cells on the *shipped*
   solver, exact DSE (`leaving_rule=5`, already present, off by default) wins
   the trajectory on **seven of nine** — and on 25fv47 it reaches **2,614
   pivots against HiGHS's 3,033, beating HiGHS outright**.
2. **The churn penalty** (this session, from the `cols_reentering_gt10`
   diagnostic) is the one mechanism that helps **greenbea specifically**.

greenbea was DSE's only significant loss. Churn was greenbea's only fix. So
DSE+churn was the cell that had to be measured, and it required lifting the
penalty out of the Dantzig-only branch of `ds_price_avx2` so it composes with
whatever score the rule produced.

## Measured — all nine cells, all certified optimal at eps=2e-5

| arm | 25fv47 | agg2 | agg3 | degen2 | fffff800 | greenbea | greenbeb | israel | tuff | CLASS | vs HiGHS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dantzig (ships)** | 8300 | 274 | 272 | 1447 | 345 | 4399 | 8919 | 234 | 221 | 24,411 | **1.843x** |
| **DSE** | 2614 | 261 | 260 | 653 | 372 | 4675 | 5633 | 121 | 183 | **14,772** | **1.115x** |
| **DSE + churn a=2.0 d=5** | 2733 | 261 | 260 | 653 | 372 | **4342** | 5980 | 121 | 183 | 14,905 | 1.126x |
| DSE + churn a=1.0 d=5 | 2783 | 261 | 260 | 653 | 372 | 4475 | 5823 | 121 | 183 | 14,931 | 1.127x |
| DSE + churn a=0.5 d=5 | 2785 | 261 | 260 | 653 | 372 | 4346 | 6089 | 121 | 183 | 15,070 | 1.138x |
| DSE + churn a=1.0 d=2 | 2965 | 261 | 260 | 621 | 367 | **4170** | **FAIL** | 121 | 183 | — | disqualified |
| HiGHS | 3033 | 534 | 563 | 537 | 424 | 2836 | 4902 | 240 | 174 | 13,243 | 1.000x |

**The class moves from 1.843x to 1.115x on a rule that already ships.**

### What churn does to greenbea under DSE

**4,675 -> 4,342.** That is below the shipped Dantzig baseline of 4,399, and it
means **greenbea stops being DSE's exception.** The anomaly that made the
ledger record "DSE is worse" is removed by a global mechanism.

### An unplanned confirmation the two findings are the same defect

`DSE + churn a=0.5 d=10` is **bit-for-bit identical to pure DSE on all nine
cells.** A deadband of 10 penalises nothing, which means **under DSE no column
re-enters more than 10 times.** DSE already suppresses the churn that the
penalty was invented to punish. The diagnostic and the rule are attacking the
same defect; DSE attacks it better.

## Dantzig + churn, for completeness (separate sweep, same nine cells + 2)

Best certified global setting **a=2.0, deadband=5: class -7.49% pivots**, with
25fv47 -16.3%, greenbea -2.6%, greenbeb -4.1%, degen2 +0.4%, and the five short
cells at **exactly +0.0%** — the deadband doing precisely its job. Best single
greenbea value is a=1.0/d=5 at **4,245 (-3.5%)**, but its class figure is only
-1.77%; preferring it *because* it suits greenbea would be per-problem tuning,
so **a=2.0/d=5 is the honest global pick**.

`lp_cycle` — `dual_infeasible` on the shipped path — becomes **optimal** under
five of seven churn settings. That is a robustness gain, not a speed one, and
it is currently unexploited.

## The honest wall picture, which is NOT yet a board win

DSE costs **+109.70 us/pivot** on greenbea (143.80 -> 253.50), and the DS2-CHUZR
bucket census locates essentially all of it in **`pricing_update`: 0.87 ->
48.75 us**, the extra `lu_ftran(lu, rho, dse_tau)` the Forrest-Goldfarb
recurrence needs. So:

| instance | wall-optimal arm today | why |
|---|---|---|
| 25fv47, degen2, israel, tuff | **DSE** | pivot win is 2-3x; easily pays the +110 us |
| greenbea, greenbeb (m=2,392) | **Dantzig + churn** | pivot win is only 3-7%; does not pay |

**For the board of record, greenbea's cell is still decided by Dantzig+churn**
(4,283 pivots x 143.8 us ~ 616 ms against DSE+churn's 4,342 x 253.5 ~ 1,101 ms).
The class win is a board-v2 result, not a greenbea result.

## The lever this creates — and it is a real one

> **DSE wins the trajectory on 7 of 9 cells and is blocked from being a wall win
> by exactly one thing: an extra FTRAN per pivot in `pricing_update`.**

That is far more specific than anything the campaign has had on the simplex
class. It is a single named kernel with a measured cost (47.88 us/pivot) and a
known consumer. Prior levers died because the thing they attacked was 3-7% of a
pivot; this one is **19%** of a DSE pivot and gates a 1.84x -> 1.12x class move.

Candidate attacks, none yet measured: reuse of the BTRAN'd `rho` already
computed for the pivot row; exploiting hypersparsity in `dse_tau` (the FTRAN
right-hand side is a single unit vector); a Devex/DSE hybrid that pays the exact
recurrence only on a subset of pivots.

## Status

- **Nothing shipped.** `LINPROGX_DS_CHURN_DSE` defaults OFF, as does
  `LINPROGX_DS_CHURN_DANTZIG`; `leaving_rule` still defaults to Dantzig.
- **Shipped path proven untouched** after restructuring the AVX2 pricing kernel:
  greenbea trace digest `679168a4baad36d6`, 6,016 vectors, 4,399 pivots —
  bit-identical to the pre-change baseline.
- **Board remains 23W-0P-1L.**
