# Is greenbea an outlier, or a category? — ANSWER: a category (2026-07-26)

**greenbea is NOT an outlier. It belongs to a real, systematic weakness: every
instance that routes to linprogx's DUAL SIMPLEX loses trajectory to HiGHS. The
category is narrow — 5 of ~30 sampled — which is why a 24-instance board exposes
it exactly once.**

## Method

Fetched 22 additional LPnetlib instances (46 total) and compared **iteration
counts**, not wall time. Wall clock on this shared box is unusable for the
question: a single-shot local run reported greenbea at 927 ms where a proper
median-of-9 measures 377 ms. Iteration counts are deterministic and
load-invariant. linprogx uses its public auto route; HiGHS is measured through
`highspy` (`simplex_iteration_count`).

## The result: the split is perfectly clean along the ROUTE

| route | instance | lx iters | hx iters | ratio | one-sided | col nnz |
|---|---|---:|---:|---:|---:|---:|
| **simplex** | 25fv47 | 8,300 | 3,033 | **2.74** | 100.0% | 5.71 |
| **simplex** | degen2 | 1,447 | 537 | **2.69** | 100.0% | 5.55 |
| **simplex** | greenbeb | 8,919 | 4,910 | **1.82** | 92.7% | 5.55 |
| **simplex** | greenbea | 4,399 | 2,836 | **1.55** | 93.0% | 5.55 |
| **simplex** | sierra | 725 | 914 | **0.79** | **25.6%** | 2.93 |
| ipm | bnl2 | 63 | 1,293 | 0.05 | 100.0% | 3.34 |
| ipm | czprob | 38 | 716 | 0.05 | 93.6% | 3.01 |
| ipm | ganges | 16 | 528 | 0.03 | 76.7% | 4.07 |
| ipm | stocfor2 | 21 | 1,668 | 0.01 | 100.0% | 3.07 |
| ipm | pilot | 65 | 5,216 | 0.01 | 74.4% | 9.13 |
| ipm | degen3 | — | — | 0.01 | 100.0% | 9.77 |
| ipm | truss | — | — | 0.00 | 100.0% | 3.16 |

…and ~20 more IPM-routed cells, all in the 0.00–0.35 range.

**Median iteration ratio across the sample: 0.01–0.05. Every one of the five
simplex-routed cells is 0.79–2.74.** greenbea is not even the worst — 25fv47
(2.74x) and degen2 (2.69x) are worse.

## Three findings

### 1. The weakness is the dual simplex, not greenbea

linprogx's **IPM is exceptional** — 20x to 500x fewer iterations than HiGHS's
simplex on every IPM-routed instance. That is the real story behind the 23 board
wins. Its **dual simplex is 1.5–2.7x behind** HiGHS's on trajectory. The board
hides this because only one of its 24 cells routes to the simplex.

### 2. Within the category, the losses track the one-sided fraction

`sierra` routes to the simplex and **wins** (0.79x) — and it is the only one with
a low one-sided fraction (**25.6%**). The four losers all sit at **92.7–100%**
one-sided with column nnz ~5.5–5.7.

That is precisely the **big-M signature**. linprogx invents an artificial bound
at `M = 1e5 x scale` for every column whose reduced cost points at an infinite
bound; HiGHS never does, because its Phase-1 bound swap makes every column boxed.
Instances with few one-sided columns barely pay the penalty; instances with ~all
of them pay it everywhere.

### 3. The Phase-1 mechanism DOES help the category — just not greenbea

Testing the in-place Phase 1 + BFRT built during the source-informed wave:

| instance | baseline | phase1 alone | **phase1 + BFRT** | HiGHS |
|---|---:|---:|---:|---:|
| 25fv47 | 8,300 | 28,738 | **7,592 (−8.5%)** | 3,033 |
| degen2 | 1,447 | 1,574 | **1,115 (−23%)** | 537 |
| sierra | 725 | 726 | **647 (−11%)** | 914 |
| greenbea | 4,399 | 5,124 | 4,675 (worse) | 2,836 |
| greenbeb | 8,919 | 10,728 | 9,570 (worse) | 4,910 |

**Three of five improve, one by 23%.** And Phase 1 *alone* is catastrophic
(25fv47: 28,738) while Phase 1 *with* BFRT wins — confirming the S3 prediction
that they are a single change, because boxed columns are exactly what makes a
bound-flipping ratio test effective.

**greenbea and greenbeb are the resistant sub-case even within their own
category.** Whatever defeats them is narrower than the category weakness.

## What this changes

- The dual simplex work is **not** greenbea-specific. It is a real component
  deficit worth 1.5–2.7x on a class of problems, currently masked by instance
  selection.
- There is a **funded, working mechanism** (Phase 1 + BFRT) that improves most
  of that class and is already implemented behind gates.
- Shipping it requires per-instance evidence and a routing decision, since it
  **regresses the two greenbea cells** — and no per-problem tuning is permitted,
  so it would need a global predicate or a genuine fix for the resistant case.

## Caveats

Iteration counts are exact; the wall-time consequences are not measured here and
this box cannot measure them. Any ship decision needs Modal paired runs. The
sample is ~30 of the ~100 LPnetlib instances, biased toward moderate sizes.
`lp_nesm` failed to download and was skipped.

Artifacts: `/tmp/category_iters.jsonl`, `/tmp/cat2.jsonl`,
`experiments/category_iters.py`, `experiments/category_bench.py`.

---

## CORRECTION (same day): finding 3 was wrong — Phase 1 does NOT help the category

Finding 3 above credited "Phase 1 + BFRT" with improving 25fv47 (−8.5%), degen2
(−23%) and sierra (−11%). **That comparison was against baseline WITHOUT BFRT, so
it attributed BFRT's gains to Phase 1.** The fair comparison isolates each
mechanism. Four arms, pivots and wall (ms):

| instance | base | bfrt | ph1 | ph1+bfrt | best by WALL |
|---|---|---|---|---|---|
| greenbea | 4399 / **802** | 4298 / 970 | 5124 / 906 | 4675 / 1056 | **base** |
| greenbeb | 8919 / **1702** | 8916 / 2735 | 10728 / 2474 | 9570 / 3602 | **base** |
| 25fv47 | 8300 / **1475** | 7454 / 2070 | 28738 / 5112 | 7592 / 1980 | **base** |
| degen2 | 1447 / 167 | 1435 / 195 | 1574 / **131** | 1115 / 154 | ph1 |
| sierra | 725 / 27 | 645 / 24 | 726 / **17** | 647 / 52 | ph1 |

**Corrected conclusions:**

- **Phase 1's own contribution is negative or neutral** on every instance except
  degen2: greenbea +8.8%, greenbeb +7.3%, 25fv47 +1.9%, sierra +0.3%, degen2
  −22.3% (measured against base+bfrt, holding BFRT constant).
- **BFRT is the mechanism with real pivot value** — 25fv47 8,300→7,454 (−10.2%),
  sierra 725→645 (−11.0%), greenbea 4,399→4,298 (−2.3%) — but it costs more wall
  than it saves on every instance measured.
- **Baseline wins on wall for all three larger instances.** The two apparent
  Phase-1 wall wins are on the smallest instances in the sample (167 ms and
  27 ms), where local noise dominates and no conclusion is warranted.

So there is **no shippable mechanism here**, and my previous claim of one was an
artefact of an unfair baseline. This is the second correction to this document
and both were mine.

## What survives, and it is the valuable part

The **category finding itself is unaffected**, because it rests on iteration
ratios against HiGHS, not on our own mechanism arms:

1. **greenbea is a category, not an outlier.** All five simplex-routed cells sit
   at 0.79–2.74x HiGHS's iteration count; all ~25 IPM-routed cells sit at
   0.00–0.35x. greenbea (1.55x) is not even the worst — 25fv47 (2.74x) and
   degen2 (2.69x) are.
2. **linprogx's IPM is exceptional and its dual simplex is the weakness.** That
   asymmetry is the real story behind the 23 board wins, and the board masks the
   weakness because only 1 of its 24 cells routes to the simplex.
3. **Within the category, losses track the one-sided fraction** — the big-M
   signature. sierra (26.5% one-sided) wins; the four cells at 93–98% all lose.

What does **not** follow is that we currently know how to fix it. Every mechanism
transplanted from HiGHS — Phase 1, BFRT, DSE, weight floor, re-selection,
tie-breaks, row-cost perturbation, logical-basis start — has now been measured on
this category, and none is a net win on wall.
