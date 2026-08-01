# Dantzig + churn penalty: the first greenbea improvement of the campaign

**Result: greenbea 4,399 -> 4,283 pivots (-2.6%), certified optimal.** Alongside
25fv47 -16.3% and greenbeb -4.1%. This is the first mechanism in the entire
campaign to reduce greenbea's pivot count.

## How it was found

1. The churn diagnostic: `cols_reentering_gt10` separates the nine
   simplex-routed instances perfectly, and `degenerate_pivots` is ~0 everywhere,
   so the class defect is **column churn**, not degeneracy.
2. `leaving_rule=4` (the existing churn penalty) turned out to be **Devex +
   penalty**, not Dantzig + penalty — confirmed because a large deadband
   reproduces pure Devex exactly. Against *that* baseline the penalty improves
   four of five instances, **including greenbea by 12.8%**.
3. That is the key observation: **the penalty helps greenbea; Devex hurts it.**
   Dantzig is greenbea's best rule, so the untested combination was
   **Dantzig + churn penalty**.

## The mechanism

`score = violation / (1 + alpha * max(0, min(enter_count, cap) - deadband))`

- The **deadband** is what makes it safe. With no deadband the unbounded penalty
  drives greenbea to `dual_infeasible`. Penalising only *demonstrable* churners
  (greenbea has 14 columns re-entering >10x; 25fv47 has 226) leaves ordinary
  columns untouched. It is a global, self-targeting rule — **no per-problem
  tuning**.
- The penalty changes for exactly **one basis position per pivot** (the one
  receiving the entering column), so it is maintained incrementally and folded
  into the AVX2 pricing kernel as a single vector divide. **This mattered
  enormously**: the first implementation forced the scalar scan and lost the
  SIMD pricing unit (~11% of wall), which ate the entire pivot win — 25fv47's
  16.3% fewer pivots netted to 0.8% of wall. Vectorised, the win survives.

## Measured (alpha=2.0, deadband=5)

| instance | pivots | delta | status | objective |
|---|---|---:|---|---|
| **greenbea** | 4,399 -> **4,283** | **−2.6%** | optimal | `-72555248.12984596` |
| **25fv47** | 8,300 -> **6,948** | **−16.3%** | optimal | `5501.845888286745` |
| **greenbeb** | 8,919 -> **8,553** | **−4.1%** | optimal | `-4302260.261206588` |
| degen2 | 1,447 -> 1,453 | +0.4% | optimal | `-1435.178` |
| agg2 | 274 -> 274 | 0% | optimal | unchanged |
| **class total** | 23,339 -> **21,511** | **−7.8%** | all optimal | |

All certified in ORIGINAL units: greenbea residual 1.77e-07, bound violation
2.88e-12; 25fv47 residual 4.55e-12. Objectives agree with baseline to ~1e-15
relative, far inside `eps=2e-5`.

There is also a trade-off frontier — `alpha=1.0, deadband=5` gives greenbea
**4,245 (−3.5%)**, its best value, with a smaller class gain (−1.9%).

## Wall time: suggestive, not established

Alternating within-process A/B on greenbea, 15 pairs, trajectory-aware:
**TOTAL B/A = 0.9539 (−4.6%)**. But every phase moved together (0.91–0.96),
which is expected when the pivot count drops, and the worst phase drift was
8.66%. **So the wall result is directionally favourable but inside the noise on
this box.** The pivot counts are load-invariant and are what should be believed
today.

## Honest status

- **Real and certified**: a −2.6% pivot reduction on greenbea and −7.8% on the
  class, with no instance regressing more than +0.4%.
- **Not established**: the wall consequence, and therefore the board effect. A
  quiet-box or Modal envab run is required before any board claim.
- **Not a flip**: even if −2.6% carried fully to wall, greenbea moves ~1.156 ->
  ~1.126, still above 1.0.

**Board remains 23W-0P-1L.** But this is the first forward motion on greenbea
since the certified −4.89% kernel ship, and it came from broadening the board —
the churn diagnostic is invisible on a suite containing one simplex instance.
