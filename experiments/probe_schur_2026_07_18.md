# Probe P-F — Schur-complement / bordered-block basis factorization (2026-07-18)

## Verdict: KILLED

All three of the brief's independent kill conditions fire. The bordered-block /
Schur mechanism cannot cut the measured 59–94% solve-vector densities, because
**those densities are a factorization-invariant property of the actual basis**,
not an artifact of fill that a better ordering removes. On the real trajectory
the basis is one giant irreducible block, not block-plus-thin-border, and a
bordered ordering does not even beat COLAMD on factor fill. Projected wall gain
≈ 0% (net negative), far below the 15% bar. No implementation sketch is
produced (gate: projection ≥ 15%).

## Method and contract compliance

- Read in order: research plan (class F), opus idea 2 (the bordered-staircase
  measurement), glm idea 3, dossier. No network, no solver source read, no
  per-problem tuning, no git ops, foreground.
- **Instrumentation** (env-gated, byte-identical off): added a
  `LINPROGX_DS_DUMP_BASIS` hook at the top of the dual-simplex pivot loop in
  `src/linprogx/_csparse.c`. When the env var is set it appends the actual
  `basis` column-index array (JSON) at the fixed trajectory points
  {500, 2000, 3500}; it reads only and changes no solver state. Rebuilt via
  `uv pip install --reinstall -e . --no-build-isolation`.
- **Byte-identical verification** (`experiments/probe_schur_capture.py`, dump
  ON vs OFF on the certified Dantzig route, `leaving_rule=1`):
  status `optimal|optimal`, iterations `4399|4399`, objective
  `-72557668.26492292` (bit-equal), `max|dx| = 0.000e+00`. **IDENTICAL.**
- Analysis: `experiments/probe_schur_analyze.py` (+ inline follow-ups). Bases
  reconstructed from A's CSC (structural cols < n) plus identity cols
  (artificials ≥ n), matching `ds_build_basis_csc`. Solve residuals `‖Bᵀx−eᵣ‖`
  ≤ 5.7e-13, so the reconstruction is faithful.

Presolved greenbea: 1525 × 3868 × 23274, density 0.395% — matches the dossier.
Sampled bases (m = 1525) are ~96–97% structural (structural/artificial =
1462/63, 1465/60, 1478/47 at iters 500/2000/3500), B nnz ≈ 7.0k, density ≈ 0.30%.

## M1 — the solve vector is factorization-invariant (the core kill)

`rho = B⁻ᵀeᵣ` (BTRAN) is a unique vector; its nnz is fixed by B and r, so no
factorization — bordered/Schur or otherwise — can change it. Verified
empirically across four orderings (NATURAL, COLAMD, MMD_AT_PLUS_A, MMD_ATA):

| iter | max nnz disagreement, random rows | max nnz disagreement, 30 densest rows | densest-30 nnz range |
|---|---|---|---|
| 500  | 0 | 0 | [123, 194] |
| 2000 | 0 | 0 | [995, 1000] |
| 3500 | 0 | 0 | [1046, 1099] |

The dense solve vectors that the mechanism targets are **bit-identical**
regardless of ordering. This is the linear-algebra fact the brief's headline
mechanism collides with: "FTRAN/BTRAN solutions … stay block-sparse instead of
filling through the border" is false — the solution vector's support is invariant;
only intermediate *factor* fill can change.

## M2 — density baseline (reconciles with the dossier)

rho nnz over **all** 1525 rows is strongly bimodal — most rows solve sparse, a
dense tail rides ~65–72%:

| iter | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|
| 500  | 1 | 75  | 100  | 153  | 194  | 21 (1.4%) |
| 2000 | 2 | 235 | 995  | 996  | 1000 | 103 (6.8%) |
| 3500 | 3 | 439 | 1039 | 1056 | 1099 | 138 (9.0%) |

The dossier's "rho p50 897 (59%)" is the median over the **actual leaving rows**
the DS selects, which land in this dense tail (p95 = 995–1039 ≈ 65–68%; max
1099 = 72%). Priced pivot row on the densest rho: nnz 620/2719/2727 =
16%/70%/71% of nonbasic columns (dossier: 94% for the actual pivot — same
regime). Both densities are downstream of the invariant rho, hence invariant.

## M3 — block-partition quality on the ACTUAL basis (blocks dissolve)

Border-size sweep on each snapshot B: remove the top-k highest-degree rows AND
cols, then measure connected components of the remaining bipartite graph
(general algorithm, whole curve reported — no tuned threshold). Largest-block
fraction of kept rows:

| border | iter 500 | iter 2000 | iter 3500 |
|---|---|---|---|
| 0%  | 88.7% | 93.8% | 95.7% |
| 1%  | 74.7% | 87.1% | 88.9% |
| 2%  | 66.6% | 82.3% | 84.5% |
| 5%  | 45.1% | 58.7% | 65.3% |
| 10% | 6.8%  | 30.7% | 37.1% |
| 15% | 3.0%  | 8.6%  | 10.2% |

At border 0% the basis is **one dominant connected component (88.7 / 93.8 /
95.7% of rows)** plus dust (singletons). Fragmenting the core below a 60% block
needs 5% of rows+cols in the border; balanced sub-10% blocks need 10–15% border
= 150–230 rows/cols. There is **no thin border that yields blocks** — the
bordered-staircase structure of the raw constraint matrix does not survive into
the actual basis. This is the brief's first kill: blocks dissolve on the real
trajectory. (Consistent with the dossier's "one dominant connected component"
and opus's own RCM finding, bandwidth 2363→1422 only.)

## M4 — factor fill under bordered vs standard ordering (fill falsifier fails)

nnz(L+U) of the actual basis factor. "BORDERED" = local block (degree ≤ p95)
ordered first via RCM, high-degree border rows/cols last, factored NATURAL —
the block+border ordering opus idea 2 proposes. Opus's own falsifier required a
≥ 25% fill **drop** vs the current ordering.

| iter | B nnz | COLAMD | MMD_ATA | BORDERED | bordered / best-standard |
|---|---|---|---|---|---|
| 500  | 7026 | 11825 | 11691 | 10563 | **0.904** |
| 2000 | 6957 | 12769 | 12668 | 16147 | **1.275** |
| 3500 | 7273 | 14748 | 14458 | 22656 | **1.567** |

The bordered ordering beats COLAMD only at iter 500 (10% less fill, short of
25%) and is 27–57% **worse** at iters 2000/3500. COLAMD/AMD already exploit
whatever block structure exists. Fill falsifier: **FAILED** on all three.

## M5 — fill attribution (border coupling is not a removable density driver)

Dropping the border rows/cols and re-solving the local block yields a *different,
incorrect* system, not a cheaper route to the true rho. Because M1 shows the
true rho is factorization-invariant, the dense support is intrinsic to (B, r):
border coupling does not "cause" removable fill in the **output** vector. The
only object a bordered factor can shrink is intermediate L/U fill (M4), and it
fails to.

## Projection arithmetic (< 15%)

Attackable wall pool (dossier per-pivot split): BTRAN 18.9% + FTRAN 17.9% +
pivot-row/PRICE 24.8% = 61.6%.

- **PRICE 24.8%** — priced row = Nᵀrho; rho invariant (M1) ⇒ priced row nnz
  invariant ⇒ PRICE cost fixed. **0% attackable.** Off the table.
- **BTRAN + FTRAN 36.8%** — output vectors invariant; only factor-traversal
  flops (bounded below by output nnz) are movable, and only if fill drops.
  M4: bordered fill vs best standard = 0.90 / 1.27 / 1.57 ⇒ net **+30%** fill
  in aggregate. Best single snapshot (10% fill drop) applied to the flop portion
  gives ≤ 0.10 × 36.8% ≈ **3.7%**, negative once iters 2000/3500 are included.
- **Refactor 5.5% + LU update 6.1%** — cheaper factors would help, but the
  bordered ordering produces *more* fill than COLAMD, so these get slower, not
  faster.
- P-B independently showed these kernels are cache-resident (fp32 gained only
  0.98–1.18×), so there is no memory-bandwidth headroom to convert either.

**Projected wall gain ≈ 0% (net negative). << 15% bar.**

## Why KILLED (all three brief criteria, independently)

1. **Blocks dissolve on the real trajectory** — the actual basis at 500/2000/3500
   is one giant component (88.7–95.7% of rows); no thin border yields blocks
   (needs 10–15% border for balanced sub-blocks).
2. **Border coupling is not the density driver** — solve vectors are
   factorization-invariant (M1 = 0 disagreement across 4 orderings, including on
   the densest rows); the 59–94% densities are intrinsic to (B, r), not
   border-induced fill.
3. **Projection < 15%** — PRICE (24.8%) is invariant and untouchable; the
   bordered ordering fails to beat COLAMD's fill (net +30%), so BTRAN/FTRAN
   flop gains are ~0. Projected wall gain ≈ 0%.

The idea's premise — that dense solve vectors are "the symptom of border-induced
fill" — is falsified: dense rho is the true, ordering-independent BTRAN result
on the leaving rows the DS actually visits. Reducing it would require solving a
different (wrong) system and would break the certificate. Kept the certificate
and residual gates intact throughout (byte-identical trajectory).
