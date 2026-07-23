# K4 — branchless/SIMD Harris ratio test (2026-07-19)

## Verdict: LIVE

The env-gated AVX2 Harris path clears K4's falsifier on both required
greenbea trajectories without changing either trajectory:

| trajectory | scalar ratio slice | SIMD ratio slice | median paired reduction | speedup from medians | committed-pivot hash identical |
|---|---:|---:|---:|---:|---|
| cold native crash | 94.461 ms | 66.467 ms | **29.67%** | 1.421x | yes |
| constructive `B*` | 80.937 ms | 49.371 ms | **38.67%** | 1.639x | yes |

The kill criterion was stated before measurement: **KILLED if the ratio slice
improved by less than 25% on either trajectory, or if either trajectory
deviated.** Neither condition fired. K4 is therefore **LIVE**. This is a live
kernel improvement, not an overall greenbea flip by itself; the projection
below remains well above the campaign's full-solve targets.

## Implementation

`LINPROGX_DS_RATIO_SIMD=1` enables the mechanism globally on CPUs reporting
AVX2 support. With the variable unset, the historical scalar loops are used.
The implementation is not selected from problem identity, dimensions, or
timing.

- Pass 1 loads four contiguous `alpha` and reduced-cost values at a time.
  Basis membership, bound status, pivot magnitude, and signed admissibility
  become masks; `vdivpd` computes ratios and `vminpd` reduces the Harris bound.
- Eligible lane masks are compacted from least to most significant bit, so the
  cached candidate list remains in the scalar scan's ascending column order.
- Pass 2 gathers four reduced costs, computes the in-band mask, and reduces the
  largest `|alpha|`, which is the existing solver's Harris pass-2 rule. Equal
  values retain the lowest column index, matching the scalar strict-greater
  tie break.
- Scalar tails handle lengths not divisible by four. The BFRT-sorted variant
  keeps its scalar implementation; the measured ordinary Harris path uses the
  existing deferred bound-flip sweep around the SIMD passes.
- `LINPROGX_DS_PIVOT_TRACE=1` adds a diagnostic FNV-1a-style hash over every
  committed `(leaving position, leaving column, entering column)` triple. It
  is disabled during timing and enabled only in separate trajectory checks.

The compiled extension was audited with `objdump`; the emitted path contains
`vgatherdpd`, `vdivpd`, and `vminpd` instructions on the AMD Ryzen 5 3600 host.

## Measurement protocol

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`
- Presolved shape: 1,525 rows x 3,868 columns x 23,274 nonzeros
- Solver: dual simplex, Dantzig leaving, EXPAND on, `bfrt=0`
- Trajectories: native cold crash and the retained P3 constructive `B*`
- Schedule: nine repetitions of all four trajectory/arm combinations;
  starting order rotated each repetition; foreground; pinned with
  `taskset -c 4`
- Attribution: `phase_us["ratio_test"]`; pivot hashes measured in separate
  untimed verification runs
- Certificate tolerance: `eps = 2e-5`
- Driver: `experiments/k4_ratio_probe.py`
- Raw results: `/tmp/k4_ratio_probe_2026_07_19.json`

The required offline build was run before the baseline and after the C change:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation
```

`UV_OFFLINE=1` mechanically enforced the campaign's no-network rule.

## Detailed timing

Each entry is the median of nine runs. The paired reduction is the median of
the nine within-repetition SIMD/scalar slice ratios and is the kill-gate
measurement.

| trajectory | arm | pivots | ratio us/pivot | DS wall | wall us/pivot |
|---|---|---:|---:|---:|---:|
| cold | scalar | 4,399 | 21.473 | 0.547435 s | 124.445 |
| cold | SIMD | 4,399 | **15.110** | **0.507958 s** | **115.471** |
| `B*` | scalar | 3,334 | 24.276 | 0.516441 s | 154.901 |
| `B*` | SIMD | 3,334 | **14.808** | **0.480756 s** | **144.198** |

| trajectory | reduction from slice medians | median paired reduction | observed wall reduction |
|---|---:|---:|---:|
| cold | 29.64% | **29.67%** | 7.21% |
| `B*` | 39.00% | **38.67%** | 6.91% |

Absolute walls are slower than the dossier's earlier Python/host context, so
the campaign projection uses the dossier's phase shares and reference rates,
not these absolute walls. The stable paired slice ratios are the mechanism's
primary result.

## Trajectory and certificate gates

The SIMD and scalar arms were identical on every behavioral check:

| trajectory | pivots | pivot hash | reduced objective | reduced `x` hash identical | terminal basis/status hashes identical |
|---|---:|---:|---:|---|---|
| cold | 4,399 | `1054624160779546655` | -72,557,668.26492292 | yes | yes |
| `B*` | 3,334 | `10455462928261366500` | -72,557,668.26492676 | yes | yes |

The scalar arm after the change also reproduces the pre-change cold and `B*`
pivot counts, objectives, reduced-`x` hashes, final-basis hashes, and bound-
status hashes exactly. Thus the knob-off path is behaviorally identical to the
pre-change build, not merely identical to another run of the changed build.

| trajectory | original objective | max equality residual | max bound violation |
|---|---:|---:|---:|
| cold | -72,555,248.12984590 | 1.77e-7 | 3.86e-12 |
| `B*` | -72,555,248.12984978 | 4.77e-7 | 2.88e-12 |

All 40 timed/trace runs returned certificate-backed `optimal`, and every
residual is comfortably below `eps = 2e-5`.

## Projection against the flip targets

### Cold trajectory

Using the dossier's 14.9% ratio-test wall share and the paired 29.67% slice
reduction:

```text
projected end-to-end cut = 0.149 * 0.2967 = 0.0442  (4.42%)
projected rate           = 90.5 * (1 - 0.0442) = 86.5 us/pivot
projected 0.38-0.40 s    = 0.363-0.382 s
```

The campaign needs about 54 us/pivot on the cold path. K4 saves a projected
4.0 us/pivot and leaves roughly **32.5 us/pivot** still to remove.

### `B*` trajectory and charged pipeline

The dossier measures the `B*` ratio slice at 17.44 us/pivot. Applying the
paired 38.67% cut:

```text
ratio saving             = 17.44 * 0.3867 = 6.74 us/pivot
projected rate           = 113.4 - 6.74 = 106.7 us/pivot
projected DS wall        = 0.378 * 106.7 / 113.4 = 0.356 s
projected charged wall   = 0.356 + 0.145 = 0.501 s
```

That remains far above the required 65-72 us/pivot, and K4 does not reduce the
0.145 s auxiliary cost toward its separate sub-0.05 s target. The mechanism
therefore breaks its own 25% kernel gate decisively but does not break the
campaign-wide conservation law alone.

## Validation and audit boundaries

- Focused SIMD/scalar trajectory test: passed.
- Ruff lint and format check: passed.
- `ty check`: passed.
- Bandit medium-and-higher scan: passed.
- Full test run: **523 passed, 7 skipped** in 71.67 s.
- Coverage-gated test run: total Python coverage **89%**, above the 85% floor.
- `pip-audit` was not run because it is network-capable and the dossier forbids
  all network access.
- No network access, competing-solver source inspection, per-problem tuning,
  or Git operation was used.

**Final verdict: LIVE.**
