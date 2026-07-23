# greenbea P-C falsifier — active-set prediction via partial IPM + reduced-LP certification

Probe contract: `experiments/greenbea_research_plan_2026_07_18.md` (P-C).
Mechanism under test: `experiments/greenbea_ideas_glm-5.2_2026_07_18.md`
idea 1, with the sifting restoration borrowed from
`experiments/greenbea_ideas_gpt5_2026_07_18.md` idea 2. Primary timing
source the plan proceeds on: `experiments/greenbea_warmstart_2026_07_18.md`
(G2's k-sweep, 0.117 s at k=50, 0.128 s at k=60).

## Verdict: KILLED at STEP 0

The plan's STEP-0 adjudication is binary: proceed on G2's 0.117-0.128 s, or
KILL immediately if ~0.68 s reproduces (opus's "route-change is wall-dead"
claim). Direct re-measurement on the current build gives:

| k | IPM wall (median of 3) | iters | mu | G2's recorded wall |
|---:|---:|---:|---:|---:|
| 40 | **0.088 s** | 39 | 7.426e-01 | 0.088 s |
| 50 | **0.327 s** | 49 | 5.947e-03 | 0.117 s |
| 60 | **0.630 s** | 58 | 3.013e-09 | 0.128 s |

k=60 reproduces opus's ~0.68 s (within 8 %) and falsifies G2's 0.128 s.
k=50 is 2.8x G2's figure. k=40 is the only point that still matches G2, and
at mu=0.74 its active-set prediction is unusable (see the sweep below). The
contract's KILL trigger fires at k=60; the rest of this document records
why the mechanism is dead on every other axis too, so the kill is
over-determined rather than resting on a single timing point.

### Why G2's 0.117 s is not reproducible

The warmstart probe sets `LINPROGX_IPM_CROSSOVER_SLICE=1`, which its report
describes as "a diagnostic-only C gate that suppresses IPM certificate
cleanup for a deliberately truncated run." That gate string is **absent
from the current C source** (`rg LINPROGX_IPM_CROSSOVER_SLICE
src/linprogx/_csparse.c` returns no matches); the probe's `main()` still
requires it, but the C side ignores the env var. Whatever shortcut that
gate represented in G2's build no longer exists.

The `LINPROGX_IPM_SLICE=1` per-phase breakdown localizes the cost explosion
that the (now-no-op) crossover gate used to hide:

| k | wall | refactor | triangular_solves | matvecs_residuals | setup_order | symbolic | **other** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.088 s | 0.052 s (59 %) | 0.011 s (12 %) | 0.006 s (6 %) | 0.013 s (15 %) | 0.001 s | **0.006 s (7 %)** |
| 50 | 0.327 s | 0.073 s (22 %) | 0.014 s (4 %) | 0.017 s (5 %) | 0.011 s (3 %) | 0.001 s | **0.213 s (65 %)** |
| 60 | 0.630 s | 0.078 s (12 %) | 0.017 s (3 %) | 0.022 s (3 %) | 0.011 s (2 %) | 0.001 s | **0.504 s (80 %)** |

The `other` bucket — the IPM main-loop endgame work (Newton solve, ratio
tests, complementarity, MCC/gamma correction) — explodes 36x from k=40 to
k=60. This is exactly the iter-40-58 dual-certificate-floor region the
stall report (`greenbea_ipm_stall_2026_07_18.md`) documents: the primal
converges but the dual sign certificate floors near 1.8e-6, and the
endgame iterations do expensive work that does not progress the
certificate. G2's 0.117 s was measured against a build that shortcut this
region; the current build does the full work, and the cost is irreducible
without re-opening the (closed) IPM-remedy axis.

The refactor and triangular-solve costs are stable across k (0.052-0.078 s
and 0.011-0.017 s respectively); the endgame is not a factorization cost,
it is an iteration-cost problem.

## Ground-truth active set (and a falsification of the 83.2 % premise)

Cold DS on the presolved problem with `LINPROGX_DS_EXPORT_BASIS=1`:
4,399 pivots, 0.401 s, status `optimal`, objective
`-72555248.129846` (matches the dossier's certified reference). The
`bound_status` export (codes: 0=LO, 1=HI, 2=FREE, 3=FIXED, 4=BASIC)
over the 3,868 structural columns gives:

| status | count | share |
|:---|---:|---:|
| LO (nonbasic at lower bound) | 2,196 | 56.8 % |
| HI (nonbasic at upper bound) | 185 | 4.8 % |
| FIXED (lo==hi) | 0 | 0 % |
| FREE | 0 | 0 % |
| BASIC | 1,487 | 38.4 % |

The probe-relevant active set — columns that should be FIXED at a bound —
is the nonbasic-at-bound set: **2,381 columns (61.6 %)**, not the dossier's
83.2 % (3,219). The dossier's figure is a miscount: it is close to the
79.1 % (3,061) of columns whose *value* lands within
`1e-6 * (1 + |bound|)` of a bound, a count that includes 681 BASIC columns
sitting degenerately at a bound. But basic columns cannot be fixed without
collapsing the basis; the fixable set is the nonbasic-at-bound 61.6 %.

This falsifies the glm idea's ceiling arithmetic directly. The idea assumed
a ~649-column reduced LP (83.2 % of 3,868 fixed). The true reduced LP has
~1,487 free columns (38.4 % of 3,868), and that is the *best case* when
prediction is perfect. With the prediction quality measured below, the
free set is 1,027-1,500 columns.

## Global rule form (no per-problem tuning)

Both sweep parameters are global scalars; no greenbea-specific constants
appear anywhere in the probe.

- **Extraction rule:** run IPM with iteration cap `k` (global; swept
  `{40, 50, 60}`). Equivalently, since mu is monotone in k on this
  fixture, "extract when mu < X globally" with X in
  `{7.4e-1, 5.9e-3, 3.0e-9}` — but the k form is the cleaner global lever
  because mu is problem-scaled.
- **Prediction rule:** for each column `j`, predict active-at-bound when
  `|x_j - nearest_bound_j| < tau * (1 + |nearest_bound_j|)`, with `tau`
  global (swept `{1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1}`). The predicted
  fix target is the nearest finite bound. Free columns (no finite bound)
  are never predicted active by construction (`d_nearest = inf`).

## Prediction quality sweep (precision/recall vs nonbasic-at-bound GT)

| k | tau | pred_active | TP | FP | FN | precision | recall | F1 | IPM wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1e-4 | 5 | 5 | 0 | 2,376 | 1.000 | 0.002 | 0.004 | 0.087 s |
| 40 | 1e-3 | 31 | 31 | 0 | 2,350 | 1.000 | 0.013 | 0.026 | 0.087 s |
| 40 | 1e-2 | 127 | 112 | 15 | 2,269 | 0.882 | 0.047 | 0.089 | 0.087 s |
| 40 | 3e-2 | 277 | 230 | 47 | 2,151 | 0.830 | 0.097 | 0.173 | 0.087 s |
| 40 | 1e-1 | 514 | 405 | 109 | 1,976 | 0.788 | 0.170 | 0.280 | 0.087 s |
| 40 | 3e-1 | 956 | 767 | 189 | 1,614 | 0.802 | 0.322 | 0.460 | 0.087 s |
| 50 | 1e-4 | 195 | 166 | 29 | 2,215 | 0.851 | 0.070 | 0.129 | 0.324 s |
| 50 | 1e-3 | 559 | 445 | 114 | 1,936 | 0.796 | 0.187 | 0.303 | 0.324 s |
| 50 | 1e-2 | 1,515 | 1,181 | 334 | 1,200 | 0.780 | 0.496 | 0.606 | 0.324 s |
| 50 | 3e-2 | 2,005 | 1,524 | 481 | 857 | 0.760 | 0.640 | 0.695 | 0.324 s |
| 50 | 1e-1 | 2,368 | 1,785 | 583 | 596 | 0.754 | 0.750 | 0.752 | 0.324 s |
| 50 | 3e-1 | 2,556 | 1,929 | 627 | 452 | 0.755 | 0.810 | 0.781 | 0.324 s |
| 60 | 1e-4 | 2,811 | 2,140 | 671 | 241 | 0.761 | 0.899 | 0.824 | 0.630 s |
| 60 | 1e-3 | 2,813 | 2,140 | 673 | 241 | 0.761 | 0.899 | 0.824 | 0.630 s |
| 60 | 1e-2 | 2,813 | 2,140 | 673 | 241 | 0.761 | 0.899 | 0.824 | 0.630 s |
| 60 | 3e-2 | 2,819 | 2,141 | 678 | 240 | 0.759 | 0.899 | 0.823 | 0.630 s |
| 60 | 1e-1 | 2,824 | 2,143 | 681 | 238 | 0.759 | 0.900 | 0.823 | 0.630 s |
| 60 | 3e-1 | 2,841 | 2,151 | 690 | 230 | 0.757 | 0.903 | 0.824 | 0.630 s |

Reading the table:

- **k=40 (mu=0.74) is dead on arrival.** Recall tops out at 0.32 even at
  tau=3e-1; the iterate is too far from the optimum to identify the active
  set. The 0.087 s IPM cost is the only one matching G2, but the
  prediction is useless.
- **k=50 (mu=5.9e-3) plateaus at precision ~0.75 / recall ~0.81.** The
  precision ceiling means ~580-630 false positives at any recall worth
  having. Each false positive is a column fixed at the wrong bound (or
  fixed when it should be free/basic), and each one must be repaired.
- **k=60 (mu=3e-9) is the best prediction point and still floors at
  precision ~0.76.** The ~680 false positives are exactly the columns
  whose IPM iterate is near a bound but whose optimal status is BASIC
  (the 681 degenerate-basic columns identified in the ground-truth
  reconciliation). The IPM cannot distinguish "at a bound because
  optimal-nonbasic" from "at a bound because degenerate-basic" — both
  look identical in the primal iterate, and this is a structural limit
  of primal-iterate-based prediction, not a threshold-tuning problem.

The precision ceiling (~0.76 at high recall) is the load-bearing
falsification of the mechanism class, independent of timing: even with a
free IPM, the primal iterate at mu=3e-9 misclassifies ~680 columns, and
the sifting restoration must unfix each of them.

## Reduced-LP cold solve (best combos) — certification unreachable

For the two best prediction points (k=50 and k=60, at tau=1e-1 and
tau=3e-1), the reduced LP was built by fixing predicted-active columns at
their nearest bound (modifying bounds only, keeping `A x = b` intact) and
solved COLD with the existing DS path (no warm basis, `leaving_rule=1`,
`bfrt=0`, `tol=1e-8`):

| k | tau | n_fixed | n_free | reduced status | red_pivots | red_wall | reduced objective |
|---:|---:|---:|---:|:---|---:|---:|---:|
| 50 | 1e-1 | 2,368 | 1,500 | `dual_unbounded_boxed` | 3,287 | 0.305 s | -72,548,847.47 |
| 50 | 3e-1 | 2,556 | 1,312 | `dual_unbounded_boxed` | 2,286 | 0.210 s | -51,240,252.42 |
| 60 | 1e-1 | 2,824 | 1,044 | `dual_unbounded_boxed` | 1,603 | 0.129 s | -66,285,232.90 |
| 60 | 3e-1 | 2,841 | 1,027 | `dual_unbounded_boxed` | 1,145 | 0.087 s | -68,640,514.25 |

Every reduced LP returns `dual_unbounded_boxed` — the dual is unbounded
with boxed variables, i.e. the reduced primal is infeasible. The
predicted-active set over-fixes: columns fixed at the wrong bound (false
positives, plus true positives fixed at the wrong-side bound) make
`A_free x_free = b - A_fixed x_fixed` inconsistent. The KKT repair loop
is designed to handle this by unfixing violators, but:

1. The reduced-LP solve alone (0.087-0.305 s) already consumes the entire
   0.30 s kill budget at every combo, before any repair work.
2. The ~580-690 false positives each require a repair re-solve. Even
   batched (unfix all violators per round, per the gpt5 sifting cousin),
   the number of rounds and the per-round re-solve cost (which grows as
   the free set shrinks toward the true 1,487) cannot fit under 0.30 s.
3. The objectives are wrong by 0.4-21.3 M, far outside the `eps=2e-5`
   relative gate — the reduced solve does not even land near the optimum
   before repair.

The reduced-LP size (1,027-1,500 free columns) confirms the 83.2 %
falsification: even at the most aggressive prediction (k=60, tau=3e-1),
the reduced LP is 1.6-2.3x larger than the glm idea's ~649-column
assumption, and the per-solve cost reflects it.

## Best-combo breakdown vs the kill bar

Best prediction point: **k=60, tau=1e-1** (precision 0.759, recall 0.900,
F1 0.823 — the highest F1 in the sweep).

| stage | wall | notes |
|:---|---:|:---|
| partial IPM (k=60) | 0.630 s | already 2.1x the 0.30 s kill bar |
| predict (tau=1e-1) | ~0.001 s | 2,824 fixed, 1,044 free, 681 FP |
| reduced LP cold solve | 0.129 s | `dual_unbounded_boxed` — no certificate |
| repairs (projection) | >= 0.13 s | >= 1 re-solve of >= 1,044-col LP, batched |
| certify in original space | ~0.002 s | postsolve + residual check |
| **end-to-end (projected)** | **>= 0.89 s** | **3.0x the 0.30 s kill bar** |

The kill criterion is `end-to-end projected wall >= 0.30 s local OR
certification unreachable`. Both clauses fire: the projected wall is
>= 0.89 s, and certification is unreachable (every reduced LP is
infeasible, and the repair burden is structurally too large to recover).

For completeness, the k=50 combo (the point G2's premise was built on):

| stage | wall | notes |
|:---|---:|:---|
| partial IPM (k=50) | 0.327 s | 2.8x G2's 0.117 s |
| predict (tau=1e-1) | ~0.001 s | 2,368 fixed, 1,500 free, 583 FP |
| reduced LP cold solve | 0.305 s | `dual_unbounded_boxed` |
| **subtotal (no repairs)** | **0.633 s** | **2.1x the 0.30 s kill bar** |

## Falsifications, summarized

1. **STEP-0 timing (primary kill).** k=60 = 0.630 s reproduces opus's
   ~0.68 s and falsifies G2's 0.128 s. The `LINPROGX_IPM_CROSSOVER_SLICE`
   gate G2 relied on is absent from the current C source; G2's 0.117 s
   measured a shortcut path through the iter-40-58 endgame that no longer
   exists. The `other` IPM bucket explodes 0.006 -> 0.213 -> 0.504 s
   across k=40/50/60 — this is the irreducible dual-certificate-floor
   endgame work the (closed) IPM-remedy axis already documented.
2. **End-to-end kill.** Best-combo projected wall >= 0.89 s, 3x the
   0.30 s bar. Even the k=50 subtotal with no repairs is 0.633 s.
3. **Certification unreachable.** All four reduced LPs return
   `dual_unbounded_boxed`; the predicted active set over-fixes and the
   reduced primal is infeasible. The repair burden (~580-690 false
   positives) cannot fit under the wall budget.
4. **The 83.2 % premise is a miscount.** The fixable nonbasic-at-bound
   set is 61.6 % (2,381/3,868), not 83.2 %. The dossier's figure counts
   degenerate-basic columns by value. The reduced LP therefore has
   ~1,487 free columns in the best case (not ~649), invalidating the
   glm idea's ~55 ms reduced-solve estimate; the measured reduced solve
   is 0.087-0.305 s.
5. **Primal-iterate prediction has a structural precision ceiling.**
   Even at mu=3e-9, ~680 columns are misclassified — exactly the
   degenerate-basic columns whose primal value sits at a bound. The IPM
   primal iterate cannot distinguish optimal-nonbasic-at-bound from
   degenerate-basic-at-bound. This is not a threshold-tuning problem;
   no global tau clears precision ~0.76 at high recall.

## Production shape

Not applicable — the probe is killed. No production sketch is warranted.
The mechanism class (partial-IPM active-set prediction + reduced-LP
certification) is dead on this fixture for five independent reasons, of
which the STEP-0 timing trigger is the contractually decisive one.

## Build, determinism, and audit

```text
cd /home/evan/dev/linprogx-pc
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
```

All measurements are single-process foreground, median of 2-3 trials, with
one unrecorded C-path warmup before each timing family. No network access,
no solver source inspection (linprogx's own C source was read to locate
the `LINPROGX_IPM_CROSSOVER_SLICE` gate and confirm its absence, and to
read the `bound_status` codes), no git operations, no per-problem tuning
(k and tau are global scalars swept as stated above), `eps=2e-5` fixed
throughout, certificate gates at the existing `tol=1e-8` DS path and
`eps=2e-5` original-space residual tolerance.

The cold DS baseline (4,399 pivots, 0.401-0.414 s, objective
`-72555248.129846`) reproduces the dossier's certified reference exactly,
confirming the fixture and the presolve path are unchanged.
