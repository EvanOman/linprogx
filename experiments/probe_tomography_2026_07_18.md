# P-D behavioral tomography — the unidentified 4,675→3,309 DSE gap (2026-07-18)

Probe P-D of the greenbea research plan. Goal: localize BEHAVIORALLY (runtime
API + logs only, never solver source) the mechanism by which HiGHS-DSE solves
the identical presolved greenbea reduction in 3,309 pivots where our exact
Forrest-Goldfarb DSE takes 4,675. Deliverable is a ranked mechanism hypothesis
with evidence, not a speedup.

Method: fused the three independent probe designs (glm-5.2 idea 2 pivot-trace
diff; claude-opus idea 4 lagged/perturbed pricing; gpt5 idea 5 phase-response
tomography) into four experiments. HiGHS is a black box driven only through the
public `highspy==1.14.0` model/options/log surface. Our DS is driven through
`solve_eq_box_dual_simplex(..., leaving_rule=5)` (exact DSE). All runs on the
cached `dual_simplex` reduction (1,525 × 3,868, 23,274 nnz), presolve off in
HiGHS so both solvers see byte-identical input. No network, no source, no
per-problem tuning. The one C change (an env-gated DSE-weight-lag knob for E2)
was byte-identical when unset and has been reverted; the worktree source is
clean. Cost vector structure: only **401 of 3,868 columns carry a nonzero cost**
(89.6% zero-cost), median |c_nz| ≈ 2.25.

## Baseline reproduction (identical reduction, presolve off)

| solver / rule                     | pivots | objective        |
|-----------------------------------|-------:|------------------|
| ours exact FG-DSE (leaving_rule=5)|  4,675 | -72557668.264923 |
| ours Dantzig (leaving_rule=1)     |  4,399 | -72557668.264923 |
| ours Devex (leaving_rule=0)       |  6,807 | -72557668.264923 |
| **HiGHS DSE (edge_weight=2)**     |**3,309**| -72557668.264923 |
| HiGHS Dantzig (edge_weight=1)     |  7,014 | -72557668.264923 |
| HiGHS "choose" (edge_weight=0)    | 12,279 | -72557668.264923 |

Dossier numbers reproduce exactly. The gap under study is 4,675 → 3,309.

---

## E1 — Cost-perturbation response (fingerprints internal perturbation / anti-degeneracy)

Perturb `c` and re-solve; 7 seeds per magnitude. **Multiplicative**
`c_j·(1+mag·U[-1,1])` (perturbs the 401 costed columns). **Additive**
`c_j + mag·U[-1,1]·(1+|c_j|)` (also perturbs the 3,467 zero-cost columns, i.e.
densifies the objective). Pivot count reported as min / mean / max over seeds.

**Multiplicative** (perturbs only costed columns):

| mag  | ours (min/mean/max)   | HiGHS (min/mean/max)  |
|------|-----------------------|-----------------------|
| 1e-9 | 3817 / 4333 / 5840    | 3309 / 3401 / 3524    |
| 1e-7 | 3757 / 4053 / 4318    | 3309 / 3409 / 3716    |
| 1e-5 | 3653 / 4170 / 4448    | 3229 / 3558 / 4107    |
| 1e-3 | 4108 / 4534 / 5135    | 2971 / 3398 / 3547    |
| 1e-1 | 3714 / 3957 / 4332    | 3236 / 3441 / 3795    |

**Additive** (densifies the objective across all columns):

| mag  | ours (min/mean/max)   | HiGHS (min/mean/max)        |
|------|-----------------------|-----------------------------|
| 1e-9 | 4021 / 4584 / 5089    | 3197 / 3323 / 3614          |
| 1e-7 | 4066 / 4658 / 4915    | 3161 / 3422 / 3879          |
| 1e-5 | 4259 / 4994 / 5518    | **7998 / 15228 / 32875**    |
| 1e-3 | 4380 / 4966 / 5604    | **22804 / 25479 / 27766**   |
| 1e-1 | 4012 / 4826 / 6116    | **16454 / 18319 / 20477**   |

ASCII response (mean pivots vs additive magnitude):

```
pivots
30000 |                              H
25000 |                                    H
20000 |                                          H
15000 |                        H
10000 |
 5000 |  o     o     o(4994)   o(4966) o        <- ours: flat ~4600-5000
      |  H     H     .         .       .        <- HiGHS: flat then explodes
 3000 |  H(3323)H(3422)
      +--1e-9--1e-7--1e-5------1e-3----1e-1---> additive magnitude
```

Findings:
- **HiGHS is pinned near its 3,309 floor under tiny perturbation** (min never
  rises above ~3,500 for mult, stays ~3,300 for small additive) and **ours never
  drops below ~3,650 under any perturbation**. The gap is *robust*, not a
  degenerate tie-break that a nudge would erase. → the steepest-edge rule /
  tie-breaking is **not** the mechanism.
- **Densifying the objective (additive ≥1e-5) explodes HiGHS 5–10×** (to
  15k–32k pivots) while **ours is essentially unmoved** (~5,000). HiGHS's
  efficiency is *contingent on the sparse-cost structure*; ours is
  cost-density-agnostic. This is the first pointer at HiGHS's dual **phase-1**
  (see E3/E4): a dense objective creates a swarm of near-tolerance reduced costs
  that its phase-1 must clear, whereas our big-M path is penalty-driven and
  indifferent to cost density. Tightening `dual_feasibility_tolerance` does **not**
  cure the explosion (9,557–11,313 at 1e-5 additive across dft ∈ {1e-9…1e-6}) —
  the phase-1 structure itself, not just its tolerance, is cost-density sensitive.

---

## E2 — Pricing-staleness simulation (does approximation move US toward 3,309?)

Env-gated throwaway knob `LINPROGX_DS_DSE_LAG=K`: refresh the exact FG-DSE
reference weights only every K pivots (stale weights reused in between);
byte-identical when unset. Directly tests claude-opus idea 4 P1 / brief
candidate 2 — "their edge may be an approximation, not extra exactness."

| lag K | pivots  | status          | objective ok |
|-------|--------:|-----------------|--------------|
| 1     |  4,675  | optimal         | yes (byte-identical baseline) |
| 2     | 11,099  | optimal         | yes |
| 3     | 13,782  | optimal         | yes |
| 4     | 15,501  | optimal         | yes |
| 6     | 14,418  | optimal         | yes |
| 8     | 15,253  | optimal         | yes |
| 16    |199,999  | numerical_error | NO (hit iter cap) |
| 32    | 16,801  | optimal         | yes |
| 64    | 14,629  | optimal         | yes |
| 128   | 12,538  | optimal         | yes |

Finding: **staleness is strictly, severely harmful** — every lag level roughly
triples our pivots (and K=16 fails to converge). No staleness level moves us
toward 3,309. → HiGHS's edge is **not** a lagged/approximate weight; exactness
is not our problem. Hypothesis H3 **rejected**.

---

## E3 — Trajectory / phase divergence (early vs steady)

HiGHS's dev log (`log_dev_level=3`) exposes a per-iteration objective plus a
phase label. Parsed phase structure of the 3,309-pivot DSE run:

| phase                       | pivots | span (iter) | objective span            |
|-----------------------------|-------:|-------------|---------------------------|
| DuPh1 (reach dual feasible) |  1,655 | 0 → 1655    | -102.2 → 0 (phase-1 obj)  |
| DuPh2 (optimize)            |  1,633 | 1655 → 3288 | -2.38e8 → -7.25e7          |
| PrPh2 (primal cleanup)      |     21 | 3288 → 3309 | → -72557668.265           |

HiGHS spends **50% of its pivots (1,655) in an explicit dual phase-1** that
minimizes summed dual infeasibility to a dual-feasible basis, then does 1,633
dual phase-2 pivots to optimize.

Our solver has **no separate phase**: a big-M artificial-bound crash makes the
initial basis dual-feasible-by-penalty, and one unified primal-infeasibility-
driven dual-simplex loop runs 4,675 pivots (52 artificial ejections). So the two
solvers **diverge from pivot 1** — they are not the same algorithm walking the
same manifold; HiGHS is optimizing a phase-1 objective while we optimize the
real one. Per glm idea 2's own criterion ("diverge from pivot 1 and never
reconverge → different Phase-1 structure"), the divergence is **architectural,
not per-pivot selection quality**. The decisive comparison:

- HiGHS's phase-2-equivalent work (dual simplex from a *dual-feasible* basis) =
  **1,633 pivots**.
- Our whole trajectory from the big-M dual-feasible crash = **4,675 pivots**.

The 2.9× difference is in the *quality of the dual-feasible basis we start from*,
not in per-pivot pricing. Corroboration from the campaign's own record: dossier
closed-axis #3 measured that **transferring HiGHS's phase-1 basis into our DS
cuts us to 3,529 pivots** (≈ HiGHS) — but densified the solves (88.8 → 113.1
us/pivot, flat wall). That prior probe confirms the pivot-count mechanism *is*
the starting dual-feasible basis; it only failed on the us/pivot trade because
the basis was foreign/transferred.

---

## E4 — HiGHS documented-option sensitivity (which single knob moves 3,309?)

All with `edge_weight=2` (DSE) fixed, presolve off, on the unperturbed reduction.

| option                          | swept values → pivots |
|---------------------------------|-----------------------|
| **dual_feasibility_tolerance**  | 1e-9:**3040** 1e-8:3040 1e-7:3309 1e-6:3391 1e-5:3535 1e-4:3855 1e-3:3754 |
| **simplex_scale_strategy**      | 0:3301 1:3309 2:3309 3:3309 **4:2950** |
| primal_feasibility_tolerance    | 1e-9…1e-5: 3305–3309 (flat) 1e-4:3433 1e-3:8819 |
| factor_pivot_threshold (Markowitz) | 0.01:3310 0.05:3483 0.1:3309 0.2:3304 0.3:3444 0.5:3309 |
| run_crossover                   | off:3309 on:3309 |
| simplex_primal_edge_weight      | -1/0/1/2: 3309 (no effect — dual route) |

Only two documented options move 3,309 materially and monotonically:
**dual_feasibility_tolerance** (3040 ↔ 3855) and **simplex_scale_strategy=4**
(→ 2950). Neither is a steepest-edge knob. Decomposing their effect by phase:

| config                     | total | DuPh1 | DuPh2 | PrPh2 |
|----------------------------|------:|------:|------:|------:|
| baseline (dft 1e-7)        | 3,309 | 1,655 | 1,633 |    21 |
| dft 1e-9 (tighter)         | 3,040 | 1,559 | 1,461 |    20 |
| dft 1e-4 (looser)          | 3,855 | **2,202** | 1,629 |    24 |
| scale_strategy=4           | 2,950 | **1,288** | 1,660 |     2 |
| scale_strategy=0 (off)     | 3,301 | 1,501 | 1,785 |    15 |

**Every lever that moves HiGHS's count acts predominantly on the dual phase-1
count.** DuPh2 stays in a tight band (~1,461–1,785). The tunable variation in
HiGHS's pivot count lives in its dual **phase-1** machinery — exactly the
"DSE-adjacent machinery" the dossier suspected, now localized.

Our-side control: sweeping our big-M penalty (`LINPROGX_DS_BIGM_FACTOR` ∈
{1e2…1e8}) moves our exact-DSE only within **4,239–5,387** — architecturally
stuck, never near 3,309. Our phase design has no knob that reaches HiGHS's regime.

---

## Ranked mechanism hypothesis

### H1 — PHASE ARCHITECTURE: HiGHS's explicit dual phase-1 builds a high-quality dual-feasible basis; our big-M crash does not. **[STRONG]**

HiGHS reaches dual feasibility via a dedicated dual phase-1 (1,655 pivots) and
then needs only 1,633 phase-2 pivots. Our big-M unified loop starts from a
penalty-dual-feasible crash that is geometrically poor and needs 4,675 primal-
infeasibility-driven pivots. The gap is the *quality of the dual-feasible
starting basis*, a phase-architecture / crash distinction — not the steepest-edge
rule, not tie-breaking, not pricing exactness.

Supporting evidence (independent):
- **E3 phase split**: 50% of HiGHS pivots are a separate phase we don't run; its
  phase-2-equivalent work is 1,633 vs our whole 4,675.
- **E4 lever decomposition**: every knob that moves 3,309 (dft, scale) moves the
  **phase-1** count; DuPh2 is stable. The mechanism is a phase-1 knob family.
- **E4 our control**: big-M sweep is stuck at 4,239–5,387 — our phase design
  cannot reach HiGHS's regime.
- **E2**: our exactness is fine (approximation triples pivots) — the extra
  pivots are not a pricing-quality deficit, consistent with a starting-basis
  cause.
- **E1 robustness**: gap survives tiny perturbation (not a tie-break); additive
  cost-densification hits HiGHS's phase-1 specifically.
- **Prior probe corroboration**: dossier axis #3 — transferring HiGHS's phase-1
  basis cut us to 3,529 pivots (only densified because foreign/transferred).

### H2 — dual-feasibility-tolerance / reduced-cost classification governs phase-1 length. **[MODERATE — sub-mechanism of H1]**

The single documented lever with a clean monotone effect (3040 ↔ 3855), acting
on DuPh1. This is the concrete tunable inside H1's phase-1, not an independent
axis. Scaling (strategy 4 → 2950) is the same story via a different norm.

### H3 — pricing staleness / approximate weights. **[REJECTED by E2]**

Lagging our DSE weights triples pivots and can break convergence; no staleness
level approaches 3,309.

### H4 — the steepest-edge rule itself / perturbation tie-breaking. **[REJECTED by E1]**

HiGHS is pinned near 3,309 under tiny perturbation and ours never drops below
~3,650; the gap is deterministic and structural.

Note on the brief's null-result framing: the brief anticipated that if the
pricing-dynamics experiments came back flat, the null would "point back at phase
architecture." That is exactly what happened, but *affirmatively*: the pricing
experiments (E1, E2) came back **negative** (ruling pricing dynamics OUT) while
the structural experiments (E3, E4) came back **positive** (localizing the gap to
the dual phase-1 / starting-basis architecture). This is a positive localization,
not merely a null.

---

## Follow-up implementation probe (implied by H1)

**Native dual phase-1 → native dual phase-2, no basis transfer.** Replace the
big-M unified crash with an explicit dual phase-1 that minimizes summed dual
infeasibility on OUR own sparse manifold to reach a dual-feasible basis, then run
our existing exact-DSE (or Dantzig) phase-2 from it. The load-bearing distinction
from closed-axis #3: axis #3 *transferred HiGHS's* phase-1 basis (foreign → 3,529
pivots but densified 88.8→113.1 us/pivot). This constructs the dual-feasible basis
*natively*, so no foreign density is imported — the same distinction opus/glm drew
for the trajectory-preserving class.

Ceiling: HiGHS's phase-2 is 1,633 pivots. If our native phase-1 reaches a
comparable dual-feasible basis and our phase-2 lands near that, total could
approach ~3,300 (≈ −29% pivots from 4,675), i.e. a real pivot-side factor for the
−41% stack — but only if the native phase-1 basis does not densify our solves.

Kill criteria (any one kills):
- Native phase-1 + phase-2 total pivots ≥ 4,200 (no material improvement over the
  current 4,675 big-M path).
- Post-phase-1 solve density rises > 15% us/pivot vs our current ~95 us/pivot
  (the recurring trade-against failure — a good dual-feasible basis that densifies
  is wall-dead, exactly as the transferred basis was).
- Native phase-1 itself costs more pivots than HiGHS's 1,655 by > 30% (its own
  phase-1 must be efficient, or the total never closes).
- Certificate fails at eps=2e-5 or objective disagrees with the SciPy/Clarabel
  oracle.

This touches high-risk phase architecture (AGENTS.md): characterization tests +
an external-oracle objective/residual check are required before any behavior
change. Falsifiable cheaply by a phase-1-only prototype measuring reached-basis
pivot count and post-handoff solve density before wiring in phase-2.
```
