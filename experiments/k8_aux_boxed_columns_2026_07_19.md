# K8 — Auxiliary refinement: boxed columns participate (2026-07-19)

## Verdict: KILLED

Extending the dual Phase-1 auxiliary so boxed columns participate (box `[-1,1]`
with cost `c_j`, and variants) does **not** improve basis quality. The isolated
effect of boxed participation, measured under identical machinery against the
boxed-fixed control, is strictly negative: participation reduces per-pivot
density but adds 836–1,218 warm-start pivots, raising the pivots x density
product and the total wall. The refined B* neither beats the boxed-fixed control
nor breaks the 47th-settled pivots x density conservation. Kill criterion (stated
before running): *LIVE only if some boxed-participation variant beats the
boxed-fixed control on the machine-independent pivots x density product.* No
variant does.

## What was tested

The derived auxiliary (`dual_phase1_derivation_2026_07_18.md`) is

```
min c'x  s.t. Ax = 0,  x_j in [0,1] (lower-only),  x_j = 0 (boxed)
```

with boxed columns FIXED because a boxed column carries no persistent dual sign
requirement (a bound flip flips the requirement). K8's P2b lead is that HiGHS's
DuPh1 responds to boxed-column costs (+9.2% when costs were added to boxed
columns; `phase1_predictions_2026_07_18.md`), suggesting their auxiliary lets
boxed columns participate. I tested four boxed treatments, holding everything
else identical:

- `fixed`   — boxed `[0,0]` (the P3 formulation; the control).
- `unit`    — boxed `[-1,1]`, cost `c_j` (K8 primary).
- `unit_nz` — boxed `[-1,1]` only for the 226 boxed columns with `c_j != 0`.
- `range`   — boxed `[-(u_j-l_j), (u_j-l_j)]` (literal flip-freedom range), cost `c_j`.

### Method (no external solver; our own machinery)

The auxiliary shares `A` and `c` with the original, so its reduced costs
`d = c - A^T y` are exactly the original's. An optimal auxiliary basis is
therefore dual-feasible for the original when boxed nonbasic columns are mapped
bound-to-bound (aux-lower -> original lower, aux-upper -> original upper). Each
auxiliary was solved with linprogx's own `solve_eq_box_dual_simplex` (which
returns the optimal basis and bound-status directly — no highspy, no solver
source, no network). The returned B* + bound-status warm-started the original
DS via `LINPROGX_DS_WARM_START=1` / `LINPROGX_DS_EXPORT_BASIS=1`.

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`; presolved 1,525 x 3,868 x 23,274.
- Column classes: 3,611 lower-only, 257 boxed (226 with nonzero cost), 0 upper/free.
- Drivers: `experiments/k8_aux_boxed_probe.py` (deterministic structural metrics,
  gates, verdict) and `experiments/k8_timing.py` (clean interleaved wall).
- Raw results: `/tmp/k8-aux-boxed/results.json`.
- eps = 2e-5; certificate-backed optimality; foreground; no per-problem tuning,
  no C changes, no git ops.

All warm starts were accepted with `singular_repairs=0`, `fell_back_to_identity=0`,
`imported_bound_status=1` — i.e. every B* is a genuine, non-repaired, dual-feasible
basis. The moderate Phase-2 pivot counts (not explosions) confirm the bound-to-bound
mapping is dual-feasible.

## Results (deterministic; pivots and densities are machine-independent)

| arm | aux iters | aux obj | boxed basic in B* | warm pivots | FTRAN dens | BTRAN dens | sum dens | **pivots x dens** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cold (no warm) | — | — | — | 4,399 | 0.241 | 0.428 | 0.669 | 2,942 |
| P3 HiGHS B* (dossier) | 1,958 | 0 | — | 3,334 | 0.322 | 0.495 | 0.817 | 2,725 |
| **fixed** (control) | 2,418 | 0.00 | 76 | **2,399** | 0.317 | 0.582 | 0.900 | **2,158** |
| unit (K8 primary) | 2,845 | -1,118.89 | 220 | 3,337 | 0.273 | 0.513 | 0.786 | 2,622 |
| unit_nz | 2,526 | -735.80 | 199 | 3,235 | 0.270 | 0.503 | 0.773 | 2,500 |
| range | 6,287 | -8,727.04 | 237 | 3,617 | 0.283 | 0.500 | 0.784 | 2,834 |

Determinism verified: reruns of `fixed` and `unit` produced identical aux
iters, warm pivots, densities, and bases.

Clean interleaved wall (9 reps, min-of; machine cold = 118.8 us/pivot vs P3's
90.5 => calibration 1.313x):

| arm | warm pivots | us/pivot (min) | us/pivot calibrated | aux wall | pipeline (aux+warm) |
|---|---:|---:|---:|---:|---:|
| cold | 4,399 | 118.8 | 90.5 | — | 0.523s |
| fixed | 2,399 | 144.4 | 110.0 | 0.208s | 0.472s |
| unit | 3,337 | 132.2 | 100.7 | 0.168s | 0.504s |
| unit_nz | 3,235 | 131.9 | 100.5 | 0.144s | 0.469s |
| range | 3,617 | ~140 | 106.0 | 0.373s | 0.756s |

Correctness gates passed on every arm: original objective matches cold to
<= 2e-5 relative (`-72,555,248.1298...`), max equality residual <= 2.4e-7, max
bound violation <= 5e-13.

## Reading the numbers — the isolated effect of participation

Boxed participation makes each pivot **cheaper** (sum density 0.77–0.79 vs the
fixed control's 0.90; calibrated ~100–106 us/pivot vs 110) but adds many more
pivots (3,235–3,617 vs 2,399). The two move together: **pivots x density rises
from 2,158 (fixed) to 2,500–2,834 (participation)**. This is the pivots x density
conservation holding, not breaking. Total wall confirms it — every participation
pipeline (0.469–0.756s) is >= the fixed control (0.472s) and all are ~2x HiGHS.

Why participation hurts, theoretically: boxed columns contribute zero dual
infeasibility, so adding `|d_j|` penalties on them (the primal image of boxed
`[-1,1]` participation) over-constrains the Phase-1 objective Phi. The nonzero
auxiliary optima (`unit` -1,118.89, i.e. min Phi = +1,118.89 > 0) are not valid
dual-infeasibility certificates for greenbea (which is bounded/optimal); they are
the signature of penalizing reduced costs that never needed penalizing, which
pulls `y` off the true dual-feasible interior and yields a lower-quality start.

The two apparent "wins" are artifacts, not participation improving quality:
- `us/pivot < 113` is met by the `fixed` control (110) too, and reflects a
  density DROP compensated by a pivot RISE — conservation intact, wall not improved.
- `unit_nz` warm pivots 3,235 < 3,334 is a K7 effect (our native auxiliary yields
  a different basis than HiGHS's), not participation: the participation-OFF control
  already reaches 2,399, and turning participation ON walks it back up to 3,235.

## Projection against the flip targets

Dossier flip target: beat HiGHS's 0.24s from a B* start needs the DENSE
trajectory at ~65–72 us/pivot PLUS an auxiliary solve under ~0.05s.

| arm | warm pivots | calibrated us/pivot | pipeline | vs HiGHS 0.24s | us/pivot needed @ aux=0.05s |
|---|---:|---:|---:|---:|---:|
| fixed | 2,399 | 110.0 | 0.472s | 1.97x | 79.2 |
| unit | 3,337 | 100.7 | 0.504s | 2.10x | 56.9 |
| unit_nz | 3,235 | 100.5 | 0.469s | 1.95x | 58.7 |
| range | 3,617 | 106.0 | 0.756s | 3.15x | 52.5 |

Best participation arm (`unit_nz`) needs 58.7 us/pivot at an unrealistic 0.05s
auxiliary; it runs at 100.5 (1.71x too slow per pivot) with a 0.144s auxiliary.
No participation variant approaches 0.24s, and none beats the boxed-fixed control.

## Kill criterion and side note

**KILLED.** Boxed participation reduces per-pivot density but inflates pivot
count so total warm-start work (pivots x density) and wall both worsen relative
to the boxed-fixed control; the refined B* does not break the pivots x density
conservation and stays ~2x HiGHS.

Side finding for K7 (not claimed here): solving the P3 (boxed-fixed) auxiliary
with our OWN DS yields a materially better B* (2,399 warm pivots, pivots x
density 2,158) than HiGHS's auxiliary basis (3,334; 2,725). It is still a denser
trajectory than cold (sum density 0.90 vs 0.669) and, once the 0.208s auxiliary
is charged, the pipeline is 0.472s — worse than cold and ~2x HiGHS. It does not
break the conservation either, but it is the sharper native-auxiliary lever and
belongs to the K7 line.
