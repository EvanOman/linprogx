# greenbea research plan — synthesis of the four-thread ideation fan-out (2026-07-18)

Inputs: four independent ideation threads (codex gpt-5.5 standard + contrarian,
claude-opus, GLM-5.2), all working from experiments/greenbea_dossier_2026_07_18.md
with no cross-contamination; all network-audited clean. Idea files:
greenbea_ideas_{gpt5,codex-contrarian,claude-opus,glm-5.2}_2026_07_18.md.

## Convergence map

| Class | Proposing threads | Bets | Attacks |
|---|---|---|---|
| A. Block/rank-k dual pivoting (shadow panels, minor pivots) | glm, gpt5, contrarian | gpt5, contrarian | pivots x us/pivot (amortization) |
| B. Precision (fp32 body w/ fp64 certificate; low-precision scout) | opus, gpt5, contrarian | opus | us/pivot (trajectory-preserving) |
| C. Active-set reduction (IPM-predicted fixing; sifting; dormant pricing) | glm, gpt5, contrarian | glm | route/problem size |
| D. Behavioral identification of the 1,090-pivot machinery | glm, opus, gpt5 | — | measurement (enables pivot trim) |
| E. Locality/SIMD representation (contiguity reorder, tiling) | glm, opus, contrarian | — | us/pivot (stack multiplier) |
| F. Schur/bordered-block basis factorization | glm, opus | — | us/pivot |
| G. Early dual-face completion | gpt5 | — | route |

## The tension structure

A, B, C are the primaries — three different theories of where the -41% lives:
- A: the trajectory is fine; amortize its linear algebra across k pivots.
- B: the trajectory is fine; the arithmetic is 2x too expensive for a 2e-5-eps
  problem with 59-94%-dense solve vectors.
- C: the trajectory is too long because the problem is too big; 83.2% of
  variables are decorative at optimum — solve the small problem instead.
They are not mutually exclusive (opus's stacking thesis: B x native-pivot-trim
multiplies to a flip), but each claims flip-or-near-flip ceiling alone, and
their falsifiers are independent. D is cheap measurement that de-risks the
pivot side for everyone. E/F are held as stack multipliers pending a primary.

## Adjudications (orchestrator)

1. TIMING DISCREPANCY resolved for C: opus's "route-change is wall-dead
   (~0.68s to good primal)" contradicts the PRIMARY SOURCE — G2's k-sweep
   (greenbea_warmstart_2026_07_18.md) measured partial IPM at 0.117s (k=50)
   and 0.128s (k=60). C's probe must re-measure this first, but the plan
   proceeds on the primary source. If 0.68s reproduces, C dies immediately.
2. Contrarian's exclusion of dual Phase-1/Phase-2 (falsified by the Maros-
   style and phase-boundary transfer experiments) is ACCEPTED — G is also
   held (single-thread, route-adjacent to closed axes).
3. All ceiling claims are author-side projections; probes carry the numbers.

## Funded probes (fan-out, all falsifier-first, all audited)

- P-A (block dual simplex shadow-panel): implement a p=4 shadow-panel
  measurement — at each pivot, record how many of the next p Dantzig
  candidates survive intervening pivots and what batched BTRAN/pivot-row
  work would cost vs sequential. KILL unless survival x batching projects
  >=2.8x on the attacked wall pool (~64.7%: pivot-row + BTRAN + FTRAN).
- P-B (precision): simulate fp32 rounding inside the fp64 DS (round solve
  vectors/pricing to fp32 each step) to test trajectory preservation and
  backward-error margins vs eps=2e-5; separately time an fp32-container
  BTRAN/pivot-row microbenchmark for the real bandwidth gain. KILL if the
  simulated trajectory diverges (different final basis without cheap
  recovery) or projected gain <20%.
- P-C (active-set prediction): partial IPM (k~40-60, re-measure cost),
  threshold-predict the active set from the iterate, fix, solve the reduced
  LP cold with the existing DS, postsolve + certify in original space at
  eps=2e-5. KILL if end-to-end projected wall >=0.30s local or the
  certificate fails (mis-predicted set -> repair loop counts against wall).
- P-D (behavioral tomography): design + run perturbation/response probes
  against HiGHS-as-black-box (runtime API only) and our DS to localize the
  1,090-pivot mechanism: cost perturbation response, pricing staleness
  simulation, pivot-trace divergence point analysis. Deliverable is a
  mechanism hypothesis with evidence, not a speedup.

Sequencing: all four in parallel (disjoint worktrees). If B and any pivot
trim both land, opus's stack arithmetic (~0.215s local = flip) is the ship
path; if C lands alone it flips outright; if A lands it flips outright.
