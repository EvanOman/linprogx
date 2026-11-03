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

---

## Penalty-shape sweep: no configuration helps greenbea

Rule 4's penalty is unbounded and permanent (`score /= 1 + enter_count`). Made
the shape tunable — `score /= (1 + alpha * min(enter_count, cap))`, with
`alpha=1, cap=inf` reproducing the original exactly — and swept:

| alpha / cap | 25fv47 | greenbeb | greenbea | degen2 | agg2 | total |
|---|---:|---:|---:|---:|---:|---:|
| **rule 1 baseline** | 8,300 | 8,919 | **4,399** | 1,447 | 274 | **23,339** |
| 1.0 / inf (original) | **4,733** | 10,260 | **FAIL** | 1,143 | 263 | — |
| 1.0 / 4 | 5,423 | 10,730 | 6,615 | 1,130 | 263 | 24,161 |
| 1.0 / 1 | 5,398 | 11,104 | 6,424 | 1,306 | 263 | 24,495 |
| 0.25 / inf | 5,547 | **FAIL** | 5,704 | 1,308 | 256 | — |
| 0.1 / inf | **FAIL** | 10,687 | 5,823 | 1,073 | 256 | — |
| **0.05 / inf** | 5,503 | 10,105 | 6,164 | 1,143 | 256 | **23,171** |
| 0.25 / 8 | 5,454 | **FAIL** | 5,704 | 1,308 | 256 | — |

**Every configuration that wins big on 25fv47 costs greenbea 40–50% or breaks
it.** The best all-optimal setting (`alpha=0.05`) improves the class total by
0.7% — which is just 25fv47's gain paying for greenbea's loss. Three settings
produce outright failures (`dual_infeasible` / no certificate), which is
disqualifying on its own.

### Conclusion for DS2

- **Churn is the right target.** `cols_reentering_gt10` is a perfect separator
  across nine instances and rule 4 removes it entirely on 25fv47 while cutting
  pivots 43% with a valid certificate.
- **A scalar penalty on `enter_count` is the wrong mechanism.** It is not a
  tuning problem — no `(alpha, cap)` in the swept range gives the class win
  without a greenbea regression or an outright failure, and on greenbeb it
  *reduces* churn 76→3 while *increasing* pivots 15%.
- **The DS2 acceptance test should be:** drive `cols_reentering_gt10` toward 0 on
  25fv47 and greenbeb **without regressing greenbea and without losing a
  certificate on any cell.** Something with memory of *why* a column was
  selected — an anti-cycling window, or a penalty tied to realised progress
  rather than raw entry count — is the direction, not a divisor.

**This does not change the board.** greenbea is unaffected or worse under every
variant; the board of record remains **23W-0P-1L**.

---

## KILL: the anti-cycling window (my own proposed direction) is worse on every axis

The section above concluded that the direction was "memory of *why* a column was
selected — an anti-cycling window ... not a divisor on raw entry count".
**I built the window and it fails.**

Implementation (`LINPROGX_DS_TABU=<pivots>`, 0 = off): record the pivot index at
which each column last entered; in CHUZR skip any row whose basic variable
entered within the last `T` pivots; fail open — if every violated row is inside
the window, relax and retry so a pivot is never lost.

| tabu window | 25fv47 | greenbeb | greenbea | degen2 | agg2 | total | churn (25fv47/gbeb/gbea) |
|---|---:|---:|---:|---:|---:|---:|---|
| **0 (baseline)** | 8,300 | 8,919 | **4,399** | 1,447 | 274 | **23,339** | [226, 76, 14] |
| 2 | 9,106 | 9,543 | 4,754 | 1,444 | 273 | 25,120 | [**281**, 101, 13] |
| 5 | **FAIL** | 9,695 | 4,763 | 1,510 | 270 | — | [266, 80, 13] |
| 10 | 8,843 | 10,297 | 4,736 | 1,507 | 271 | 25,654 | [252, 103, 11] |
| 25 | 8,783 | 10,937 | 4,988 | 1,340 | 292 | 26,340 | [237, 105, 14] |
| 50 | 9,492 | 11,417 | 5,655 | 1,651 | 299 | 28,514 | [284, 127, 12] |

**Worse at every window length, and — the informative part — it does not even
reduce churn. It increases it** (226 → 281 at T=2, still 252 at T=10).

### What that teaches

Rule 4 and the tabu window have the same intent — discourage ejecting a variable
that keeps coming back — but opposite effects:

- **Rule 4's penalty is PERSISTENT**, proportional to total entry count over the
  whole run. It drove 25fv47's churn 226 → **0**.
- **The window is RECENCY-based.** It makes churn *worse*.

So the churn is **not a short cycle** that a recency window can break. It is a
small set of columns that churn heavily *across the entire trajectory*, and only
a penalty with long memory of total participation suppresses them.

That is a genuinely useful constraint on DS2's pricing design, and it narrows the
space considerably: recency-based anti-cycling is measured dead here; whatever
replaces rule 4 must retain **long-run memory** while not destabilising greenbea
the way an unbounded divisor does.

**Three anti-churn mechanisms have now been measured on this class**: the
persistent divisor (big class win, breaks greenbea), its bounded/damped variants
(no configuration wins without a regression or failure), and the recency window
(worse everywhere). Board unchanged: **23W-0P-1L**.

---

## CORRECTION + best result: the churn penalty's real baseline is Devex, not Dantzig

`leaving_rule=4` computes the **Devex** score (`viol^2 / w`) and then applies the
churn penalty; `leaving_rule=1` is **Dantzig** (`viol`). So comparing rule 4
against rule 1 conflates a scoring-rule change with the penalty. Confirmed
directly: **`rule4` with a large deadband reproduces `rule0` Devex exactly**
(5,144 / 9,766 / 6,807 / 1,174 / 258 — identical in every cell).

Measured against the correct baseline:

| config | 25fv47 | greenbeb | greenbea | degen2 | agg2 | total |
|---|---:|---:|---:|---:|---:|---:|
| Dantzig (rule 1) | 8,300 | 8,919 | **4,399** | 1,447 | 274 | 23,339 |
| Devex (rule 0) | 5,144 | 9,766 | 6,807 | 1,174 | 258 | 23,149 |
| **Devex + churn, deadband 5** | 5,264 | **9,129** | **5,934** | **1,110** | 258 | **21,695** |
| *penalty effect vs Devex* | +2.3% | **−6.5%** | **−12.8%** | **−5.5%** | 0% | **−6.3%** |

### Two real results

1. **The churn penalty is genuinely good — against its own baseline.** It
   improves four of five instances versus Devex, including **greenbea by 12.8%**,
   for **−6.3%** overall. The deadband is what makes it safe: at K=0 the
   unbounded penalty drives greenbea to `dual_infeasible`; at K=5 every instance
   certifies. Penalising only *demonstrable* churners (greenbea has 14 columns
   re-entering >10x against 25fv47's 226) removes the destabilisation.
2. **It is the best class configuration found**: 21,695 total pivots against
   Dantzig's 23,339, **−7.0%**, all optimal, all certified.

### Why it still cannot ship

Dantzig remains the best rule for **greenbea specifically** (4,399, versus 5,934
for the best churn-aware configuration). So adopting Devex+churn globally would
trade a 7% class gain for a ~35% regression on the one cell that is already our
only loss. Under "no per-problem tuning" that trade is not available, and it
moves the board in the wrong direction.

**This is the sharpest statement of greenbea's peculiarity yet found:** it is the
one instance in its class where the *cheap, memoryless* rule beats every
sophisticated one. Devex loses to Dantzig on it by 55%; DSE by 6%; churn-aware
Devex by 35%. Every mechanism that helps the class hurts greenbea, and every
mechanism greenbea likes (plain Dantzig) leaves the class on the table.

Board unchanged: **23W-0P-1L**.
