# K9 — Hybrid start: density shaping (2026-07-19)

## Verdict

**LIVE** (on the pre-registered DS-only kill criterion), but a *qualified*
LIVE: the conservation break is real, reproducible, and model-free, yet it is
strictly **pipeline-dominated** by the plain cold crash and cannot on its own
win the greenbea board.

- Kill criterion (pre-registered): *"Kill if the pivots-x-density product never
  drops below 0.37s."* It drops to **0.341s** (k=5), so K9 is not killed.
- Decisive model-free fact: **k=5 runs B\*'s exact 3,333-pivot trajectory
  (vs 3,334) at 7.5% / 9.8% lower FTRAN / BTRAN density.** Same pivots + strictly
  lower solve density ⟹ strictly lower DS wall, independent of any timing model,
  because the dossier establishes us/pivot is monotone in solve density.

## Mechanism

Interpolate between the auxiliary basis **B\*** (few pivots, dense) and the
all-slack basis (cold-like) by replacing the **k densest structural columns of
B\*** with their slack counterparts, k swept globally.

- B\* is reproduced exactly as P3: solve the homogeneous auxiliary
  (`min c'x, Ax=0`, lower-only columns `[0,1]`, boxed columns `[0,0]`) with
  highspy 1.14.0 (public API only, no source read), extract the retained optimal
  basis → 1,070 structural + 455 logical = 1,525 basic (matches P3: aux 1,958
  iters, warm 3,334 pivots, obj `-72555248.12985`, densities 0.322 / 0.495).
- Each swapped structural column is matched to a **distinct complement row**
  (a row whose slack is nonbasic in B\*) via a bipartite matching of the
  structural-on-complement submatrix M (nonsingular by construction). This
  guarantees the swapped slack is nonbasic and yields a nonsingular basis with
  **no singular repairs and no identity fallback** for every small-k arm.
- The 5 densest structural columns swapped at k=5 carry nnz {25, 21, 20, 20, 20}
  vs a **median structural-column nnz of 5** — the swap removes the handful of
  columns that dominate solve-vector fill.

## Measurements (greenbea, 1,525×3,868×23,274; eps=2e-5; 15 reps, medians)

Densities are machine-invariant (our cold/B\* densities equal the dossier's to
3 decimals), so `proj_DS_wall` projects each arm onto the **dossier host** using
only the dossier's us/pivot-vs-density slope, calibrated on its two anchors:
`us/pivot ≈ -13.01 + 154.73·(ftran+btran)` → cold anchors to 0.398s, B\* to
0.378s by construction. `wall_med` is the raw local median (this host runs
~1.5× slower).

| k | pivots | ftran | btran | proj us/piv | proj DS wall | wall_med (local) | gate |
|---:|---:|---:|---:|---:|---:|---:|:--|
| cold | 4,399 | 0.241 | 0.428 | 90.5 | **0.398** | 0.5915 | ok |
| 0 (B\*) | 3,334 | 0.322 | 0.495 | 113.4 | **0.378** | 0.5513 | ok |
| **5** | **3,333** | **0.298** | **0.446** | **102.2** | **0.341** | **0.5039** | ok |
| 10 | 3,950 | 0.301 | 0.451 | 103.5 | 0.408 | 0.6014 | ok |
| 15 | 3,780 | 0.297 | 0.438 | 100.7 | 0.381 | 0.5549 | ok |
| 20 | 3,780 | 0.306 | 0.448 | 103.7 | 0.392 | 0.5669 | ok |
| **25** | **3,518** | **0.284** | **0.439** | **98.9** | **0.348** | **0.5025** | ok |
| 30 | 3,854 | 0.284 | 0.407 | 93.9 | 0.362 | 0.5517 | ok |
| 35 | 3,868 | 0.299 | 0.436 | 100.7 | 0.389 | 0.5589 | ok |
| **40** | **4,047** | **0.269** | **0.374** | **86.5** | **0.350** | **0.5410** | ok |
| 45 | 5,053 | 0.289 | 0.419 | 96.6 | 0.489 | 0.7178 | ok |
| 50 | 4,730 | 0.293 | 0.405 | 95.0 | 0.449 | 0.6626 | ok |
| 60 | 5,032 | 0.283 | 0.400 | 92.7 | 0.466 | 0.7083 | ok |
| 80 | 4,797 | 0.258 | 0.374 | 84.7 | 0.406 | 0.6510 | ok |
| 100 | 4,527 | 0.259 | 0.369 | 84.2 | 0.381 | 0.6230 | FAIL* |
| 200 | 5,905 | 0.242 | 0.343 | 77.5 | 0.457 | 0.7934 | ok |
| 500 | 7,603 | 0.212 | 0.308 | 67.4 | 0.513 | 0.9922 | FAIL† |
| 1,070 | 7,603 | 0.213 | 0.307 | 67.4 | 0.512 | 1.0451 | ok |

\* k=100 returns suboptimal status (obj off by 0.4%); a collapsed arm.
† k=500 exceeds the 31-repair budget and falls back to identity; disqualified.
Both are large-k collapse arms, irrelevant to the small-k winners.

Local walls corroborate the projection: k=5 (0.504s) and k=25 (0.503s) beat
**both** endpoints — B\* (0.551s) by ~9% and cold (0.592s) by ~15%.

## Projection arithmetic vs the flip targets

- Flip target (dossier): a start with **pivots ~3,600–3,900 at us/pivot ~95–100
  → 0.34–0.37s.**
- Result: the sweet spot sits at **lower k than hypothesized**. The best break
  keeps B\*'s pivot count and sheds density rather than trading pivots for
  density:
  - **k=5: 3,333 pivots, proj us/pivot 102.2 → 0.341s** (10% below B\*'s 0.378,
    14% below cold's 0.398). Model-free: same pivots as B\*, strictly lower
    density.
  - **k=25: 3,518 pivots, proj us/pivot 98.9 → 0.348s.**
  - **k=40: 4,047 pivots, us/pivot 86.5 → 0.350s.**
  - Arms landing in the literal 3,600–3,900 pivot band (k=30/35) project to
    0.362 / 0.389s — the density gain there is partly eaten by the pivot rise.
- The pivots × density product is therefore **not conserved**: several small-k
  starts fall 0.34–0.36s, below the 0.378s B\* floor and the 0.398s cold floor.

## Why it is a *qualified* LIVE (honest falsifier caveats)

1. **Pipeline domination — the board loss is untouched.** Every K9 start is
   derived from B\*, so it inherits the auxiliary construction cost
   (0.145s dossier / 0.185s local). Projected pipeline for k=5 is
   **0.145 + 0.341 = 0.486s**, worse than the plain cold crash (0.398s, *no*
   auxiliary) and ~2× HiGHS (0.24s). Density shaping breaks the DS-only
   micro-conservation but is swamped by the auxiliary tax; on its own it cannot
   beat greenbea. It is only useful combined with K7/K8 (which attack the
   auxiliary cost and basis quality).
2. **Non-monotone lottery, not a smooth dial.** The break depends on *which*
   columns are removed: k=5/25/40 win, but k=10/15/20/35 do not (pivots jump to
   3,780–3,950). Only removing the very densest handful helps; past k≈40–45 the
   basis quality collapses (pivots 4,700–7,600). Choosing a single best k would
   be per-problem tuning; the reported finding is that the mechanism *produces a
   break at small k*, strongest at the densest-few swap.

## Correctness gates (all winning arms)

Every small-k arm (k ≤ 80) returned status optimal, objective within 2e-5
relative of the cold reference (`-72555248.12985`), max original equality
residual ≤ 6e-7, max bound violation ≤ 5e-13, and — critically — **0 singular
repairs and 0 identity fallbacks**, i.e. the swapped bases are genuinely
nonsingular warm starts, not repaired collapses.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --offline highspy   # 1.14.0, from local cache; no network
LINPROGX_DS_WARM_START=1 uv run python -m experiments.k9_density_shaping_probe
```

Driver: `experiments/k9_density_shaping_probe.py`. Raw results:
`/tmp/k9-density-shaping/results.json`. No network, no solver source, no
per-problem tuning, no git ops; foreground.
