# K3 — dense-mode SIMD BTRAN/FTRAN bodies (2026-07-19)

**Verdict: KILLED.** The kill criterion was less than 15% improvement in the
combined BTRAN+FTRAN solve-body slice. Every threshold that actually selected
the full-dense bodies instead produced a large slowdown and changed pivot
counts. At the primary 25% density threshold, the combined slice was **87.2x
slower cold** and **75.0x slower from B\***. A 60% threshold restored the
baseline only because it selected the new bodies zero times.

## Falsifier and mechanism

The shipped adaptive dense staging path still walks sparse L/U/spike/eta index
lists. K3 tested a genuinely different body behind the global diagnostic gate
`LINPROGX_DS_FT_DENSE_FULL=<density>`:

- Materialize static L and the current FT U' as dense column-major arrays.
- Keep U' synchronized after every committed FT update by deleting the old row,
  replacing the spike column, and appending a dense eta row.
- FTRAN uses contiguous dense L AXPYs, dense eta dots, and dense U' AXPYs.
- BTRAN uses contiguous dense U' dots, reverse dense eta AXPYs, and dense L
  dots.
- AVX2 processes 16 doubles per unrolled iteration on this Ryzen 5 3600, with a
  scalar portability fallback. The solve bodies never visit factor index lists.

The threshold uses only the running global mean solve density. No problem name,
dimension-specific constant, or per-problem tuning participates. The knob-off
path does not allocate or execute any K3 storage/body.

Exact arithmetic order changes in the SIMD dot reductions, so the bound gates
were those required by K3: identical pivot count, reduced objective relative
agreement within `1e-9`, and original-space residuals within fixed
`eps=2e-5`.

## Method

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`, presolved to 1,525 x 3,868 x
  23,274.
- Starts: native cold crash and the exact retained B* column list from the local
  `/tmp/phase1-predictions/results.json` artifact.
- The B* artifact was reused directly; no auxiliary or external solver ran.
- Solver: Dantzig leaving, `expand=1`, `bfrt=0`, DS tolerance `1e-8`.
- Certificate epsilon: `2e-5`, fixed.
- Attribution: `LINPROGX_DS_SOLVE_SLICE=1`.
- Threshold sweep: 0.25 (the shipped dense-staging crossover), 0.45 (partial
  activation), and 0.60 (off-boundary control).
- Runs were foreground and subprocess-isolated so cached environment knobs
  could not leak between arms. The structured pass has one run per arm; the
  effect is 20x–87x in the attacked slice, far beyond timing noise. Separate
  smoke runs reproduced the same status, pivot counts, objectives, and failure
  direction.
- Build after every C edit used the dossier commands with `UV_OFFLINE=1` added
  to make network access mechanically impossible.
- No network access, Git operation, or solver-source inspection occurred.

Raw structured results:
`experiments/k3_dense_full_results_2026_07_19.json`.

## Measurements

The solve columns are the nested BTRAN+FTRAN body timers, not the broader phase
buckets. Full counts show how often each SIMD body actually ran.

| start | threshold | F/B full calls | pivots | combined solve | solve us/pivot | ratio vs baseline | DS wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | off | 0 / 0 | **4,399** | 0.1862s | 42.33 | 1.00x | 0.5545s |
| cold | 0.25 | 2,587 / 2,558 | **4,408** | 16.2673s | 3,690.40 | **87.18x slower** | 17.5447s |
| cold | 0.45 | 743 / 729 | **4,404** | 3.7363s | 848.39 | **20.04x slower** | 4.3565s |
| cold | 0.60 | 0 / 0 | **4,399** | 0.1770s | 40.23 | 0.95x (noise; no activation) | 0.5262s |
| B* | off | 0 / 0 | **3,334** | 0.1683s | 50.49 | 1.00x | 0.5163s |
| B* | 0.25 | 2,561 / 2,720 | **3,332** | 12.6152s | 3,786.06 | **74.98x slower** | 13.7011s |
| B* | 0.45 | 761 / 779 | **3,335** | 3.9696s | 1,190.27 | **23.57x slower** | 4.6270s |
| B* | 0.60 | 0 / 0 | **3,334** | 0.1612s | 48.34 | 0.96x (noise; no activation) | 0.4958s |

The 25% arm also fails the pivot-count gate on both trajectories: cold shifts
by +9 pivots and B* by -2. The 45% arm shifts cold by +5 and B* by +1. This is
consistent with the permitted-but-path-changing SIMD dot reduction order; it
is independently disqualifying even before the timing result.

## Correctness and off-path gates

All arms returned certificate-backed `optimal`. The active K3 arms preserved
the objectives and residual tolerances despite their path changes:

| start | baseline reduced objective | active-arm reduced objective | max original equality residual | max bound violation |
|---|---:|---:|---:|---:|
| cold | -72,557,668.26492292 | -72,557,668.26492298 | 1.77e-7 | 5.79e-12 |
| B* | -72,557,668.26492676 | -72,557,668.26492676 | 4.77e-7 | 2.88e-12 |

The cold absolute objective delta is `5.96e-8`, or `8.2e-16` relative; the B*
objective is identical. Both therefore pass the `1e-9` relative objective
gate, and every residual is far inside `eps=2e-5`. Knob-off cold and B* runs
exactly reproduce the established pivot counts and reduced objectives. Focused
verification after the final build:

If K3's terse “objective 1e-9” is interpreted as an absolute rather than
relative tolerance, the cold active arm additionally fails that gate; the
verdict is unchanged either way.

- `tests/test_simplex_lu.py`: 23 passed.
- `tests/test_dual_simplex.py tests/test_dual_cleanup.py`: 37 passed.
- `ruff check experiments/k3_dense_full_probe.py`: passed.
- Full `just test-cov`: 522 passed, 7 skipped, 89.16% coverage (85% floor).
- Repository lint, format check, type check, and Bandit: passed. `pip-audit`
  was not invoked because its remote advisory lookup conflicts with the
  campaign's audited no-network rule.

## Projection against the flip targets

The dossier's K3 kill gate is already modest: a 15% BTRAN+FTRAN improvement
would save only `36.8% x 15% = 5.52%` of cold wall, moving the canonical 90.5
us/pivot to about **85.5 us/pivot** and 4,399 pivots to about **0.376s**. That is
still well above the required **54 us/pivot**. BTRAN+FTRAN alone would need an
impossible 109.5% elimination to supply the full 40.3% cold-path reduction.

Using the measured 25% body ratios on the dossier's canonical nested body
costs gives the opposite projection:

- Cold: `(90.5 - 30.70) + 30.70 x 87.18 = 2,736 us/pivot`, about **12.04s**
  for 4,399 pivots and **50.7x** the 54 us/pivot target.
- B*: `(113.4 - 37.16) + 37.16 x 74.98 = 2,863 us/pivot`, about **9.54s**
  for 3,334 pivots before the 0.145s auxiliary, and **39.8–44.0x** the required
  65–72 us/pivot band.

The reason is arithmetic intensity in the wrong direction: the factor remains
sparse even when the solution vector is dense. Replacing roughly tens of
thousands of indexed factor entries with full 1,525-squared triangular sweeps
adds orders of magnitude of zero multiplies. AVX2 removes index loads and
branches but cannot repay that work inflation on this cache-resident factor.

**Final verdict: KILLED.** There is no observed density crossover at which the
full dense L/U/eta SIMD bodies win. Partial activation loses by 20x–24x, broad
activation loses by 75x–87x, and the only neutral threshold never activates.
