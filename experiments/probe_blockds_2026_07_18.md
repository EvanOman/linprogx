# Probe P-A — greenbea shadow-panel measurement (2026-07-18)

## Verdict: KILLED

The shadow-panel premise fails before block-algorithm engineering. On the
unchanged 4,399-pivot Dantzig trajectory, a width-four panel commits only
**1.281 consecutive pivots per panel** in a non-overlapping walk. Only
**101/4,395 (2.30%)** overlapping width-four panels supply all three subsequent
pivots. The measured width-four BTRAN+pivot-row panel costs **82.82 us** versus
**75.83 us** for four lean scalar microbenchmark kernels (0.916x), or an
idea-favorable **1.434x** batching speedup when normalized against the slower
production scalar buckets. Survival times that favorable batching is only
**0.459x operationally** and **0.564x under an any-hit upper bound**, nowhere
near the required 2.8x.

The best full attacked-pool projection is width two at **0.590x** (a slowdown),
after keeping FTRAN sequential and charging exact minor-pivot maintenance.
Every width fails the research plan's 2.8x kill criterion.

## Contract and method

The probe read, in order, the research plan, dossier, contrarian idea 1, and
GPT-5 idea 1. It reconciles them as a trace-only scalar-trajectory experiment:

- The production solver still chooses and commits one ordinary Dantzig/Harris
  pivot at a time. No rank-k exchange, block assignment, or changed endpoint
  was implemented.
- At each committed pivot, the probe records the top eight primal-infeasible
  Dantzig candidates. Candidates are tracked by **basic-column identity**, not
  basis position: after a pivot, the position contains a different basic
  variable and must not count as survival of the old panel member.
- For `p in {2,4,8}`, survival asks how many of the original `p-1` shadow
  members become the selected leaving variable during `t+1 .. t+p-1`.
  Separately, a consecutive-prefix statistic measures the usable block length
  before the scalar trajectory first asks for a row outside the panel.
- Every 64 pivots, on 69 live basis snapshots, a diagnostic kernel computes
  panels of width 2/4/8. Each snapshot is repeated five times. The scalar
  baseline traverses the LU factors and CSR matrix once per RHS; the panel
  kernel traverses each factor/matrix entry once with RHS width as the inner
  loop. All production solve counters are restored after a sample.
- `LINPROGX_DS_SOLVE_SLICE=1` measures actual FTRAN and BTRAN solve bodies.
  Seven clean direct solves supply medians for wall and phase time.

Instrumentation is gated by `LINPROGX_DS_SHADOW_PANEL=1` and
`LINPROGX_DS_SHADOW_PANEL_BENCH=1`. With either tracing or benchmarking enabled,
status, 4,399 iterations, objective, and the complete returned `x` vector were
identical to the clean baseline. The focused dual-simplex test file reports
**28 passed**, including an explicit gating/read-only test.

No network-capable command or tool was used, no external solver source was
read, no solver parameter was tuned to greenbea, and no Git operation was run.

## Baseline and attacked wall pool

The clean solve postsolved to original-space objective
`-72,555,248.1298459`, with maximum original-space primal residual
`1.769e-7`.

| Quantity | Median time | Direct-wall share |
|---|---:|---:|
| Direct DS wall | 406.464 ms | 100.00% |
| Pivot-row scatter | 81.403 ms | 20.03% |
| BTRAN solve body | 49.230 ms | 12.11% |
| FTRAN solve body | 87.219 ms | 21.46% |
| **Specified attacked pool** | **217.852 ms** | **53.60%** |

The dossier's named shares sum to 61.6%, not 64.7%:
`24.8% + 18.9% + 17.9% = 61.6%`. The contrarian document's 64.7% is a
different pool (BTRAN + pivot row + ratio test + LU update). On the current
binary, the exact solve-slice hook lowers the specified pool further to 53.60%.
The broader `phase_us.btran_rho` bucket is not used here because its timing
boundary also includes the leaving scan; the solve-slice BTRAN timer isolates
the triangular solve body.

At the measured 53.60% share, even an infinite acceleration of this pool can
save only 53.60% overall. A 41% overall reduction would require
`1 / (1 - 0.41/0.5360) = 4.26x` on this measured pool. The plan's fixed 2.8x
criterion is retained below; it is more favorable to the idea.

## Panel survival

### Any shadow member selected in the next `p-1` pivots

Counts exclude the guaranteed lead pivot at `t`.

| p | Eligible panels | 0 future hits | 1 | 2 | 3 | 4 | 5 | 6 | 7 | Mean future hits | Mean members used incl. lead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4,398 | 3,571 (81.20%) | 827 (18.80%) | — | — | — | — | — | — | 0.188 | 1.188 |
| 4 | 4,395 | 2,514 (57.20%) | 1,351 (30.74%) | 429 (9.76%) | 101 (2.30%) | — | — | — | — | 0.572 | 1.572 |
| 8 | 4,385 | 1,514 (34.53%) | 1,337 (30.49%) | 849 (19.36%) | 389 (8.87%) | 184 (4.20%) | 64 (1.46%) | 39 (0.89%) | 9 (0.21%) | 1.267 | 2.267 |

Selection probability falls rapidly with horizon:

| p | t+1 | t+2 | t+3 | t+4 | t+5 | t+6 | t+7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 18.80% | — | — | — | — | — | — |
| 4 | 25.44% | 17.20% | 14.52% | — | — | — | — |
| 8 | 28.69% | 21.16% | 19.09% | 16.24% | 15.17% | 13.73% | 12.61% |

### Consecutive usable block length

The non-overlapping walk starts a panel, consumes panel members only while the
unchanged trajectory continues to select them, then starts a new panel at the
first miss. This is the operational amortization statistic.

| p | Panels in walk | Length 1 | Length 2 | Length 3 | Length 4 | Length 5 | Length 6 | Length 7 | Length 8 | Mean commits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3,774 | 3,149 | 625 | — | — | — | — | — | — | **1.166** |
| 4 | 3,432 | 2,702 | 543 | 139 | 48 | — | — | — | — | **1.281** |
| 8 | 3,255 | 2,478 | 540 | 165 | 46 | 13 | 5 | 7 | 1 | **1.348** |

The larger panel finds more eventual hits but barely extends the consecutive
commit run. Width eight pays for eight rows and obtains only 1.348 consecutive
pivots on average.

## Batching microbenchmark

Times are per attempted panel, averaged over `69 snapshots * 5 repeats`.
`Kernel speedup` compares like-for-like lean scalar and fused diagnostic
kernels. `Production-normalized speedup` compares the measured fused panel
cost with `p * 29.696 us`, the actual production pivot-row+BTRAN cost per
pivot; this intentionally credits the panel with amortizing production's
pattern/bookkeeping overhead even though the diagnostic panel does not
implement it.

| p | Scalar BTRAN | Panel BTRAN | BTRAN speedup | Scalar pricing | Panel pricing | Pricing speedup | Scalar combined | Panel combined | Kernel speedup | Production-normalized speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 18.85 us | 45.95 us | 0.410x | 19.77 us | 19.61 us | 1.008x | 38.62 us | 65.57 us | 0.589x | 0.906x |
| 4 | 37.51 us | 51.58 us | 0.727x | 38.33 us | 31.24 us | 1.227x | 75.83 us | 82.82 us | 0.916x | 1.434x |
| 8 | 77.72 us | 64.41 us | 1.207x | 74.78 us | 45.67 us | 1.638x | 152.49 us | 110.08 us | 1.385x | 2.158x |

The locality mechanism is real but too narrow. Fused pricing reduces CSR
matrix reads by **1.83x / 3.34x / 6.17x** at widths 2/4/8. Because the union
support is denser than each individual support, it executes
**1.093x / 1.198x / 1.296x** as many pricing FMAs as the scalar sparse-support
walk. The multi-RHS BTRAN only becomes faster at width eight. Both BTRAN and
pricing panels agree with scalar outputs at maximum recorded relative error
**0.0**; the timing result is not caused by an approximate solve.

## Exact minor-pivot maintenance

Let the committed entering-column FTRAN be `u = B^-1 a_q`, and let `r` be the
leaving basis position. For each retained candidate position `i != r`, exact
basis replacement requires

```text
rho_i'   = rho_i - (u_i / u_r) rho_r
alpha_i' = alpha_i - (u_i / u_r) alpha_r
```

where `rho_i^T = e_i^T B^-1` and `alpha_i = rho_i^T A_ext`. If the consumed
row is retained as part of a full inverse panel, it is normalized as
`rho_r' = rho_r/u_r` and `alpha_r' = alpha_r/u_r`; it is not needed merely to
carry the unconsumed shadow candidates. A later FTRAN against frozen major
factors must also apply the accumulated minor eta transforms before its
coefficients can be used. The existing scalar FTRAN/update representation
already performs that correction, so FTRAN remains sequential in this probe.

For greenbea, `m=1,525` and `n_total=3,868+1,525=5,393`. One surviving-row
update costs one division plus:

```text
alpha: 2 * 5,393 = 10,786 flops
rho:   2 * 1,525 =  3,050 flops
total:              13,836 flops per surviving row
```

A full width-p panel performs `p(p-1)/2` surviving-row updates:

| p | Full row updates | Full alpha+rho flops | Timed alpha update | Estimated alpha+rho update | Observed mean row updates | Charged mean alpha+rho time |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 13,836 | 7.15 us | 9.17 us | 1.000 | 9.17 us |
| 4 | 6 | 83,016 | 26.66 us | 34.20 us | 3.480 | 19.84 us |
| 8 | 28 | 387,408 | 89.74 us | 115.11 us | 8.919 | 36.67 us |

The observed mean row-update charges are 13,836, 48,148, and 123,406 flops per
attempted width-2/4/8 panel. These costs are in addition to rechecking global
leaving admissibility and the full Harris ratio test after every committed
minor pivot. Those checks were not credited as savings.

## Projection arithmetic

Measured production costs per committed pivot are:

```text
C_BR   = (pivot-row + BTRAN) / 4,399 = 29.696 us
C_F    = FTRAN / 4,399               = 19.827 us
C_pool = C_BR + C_F                  = 49.523 us
```

For width `p`, panel kernel cost `C_panel`, and mean consecutive commits `L`,
the favorable production-normalized batching factor and operational
survival-times-batching factor are

```text
G_batch = p * C_BR / C_panel
G_oper  = G_batch * L / p = L * C_BR / C_panel.
```

An even more favorable any-hit bound replaces `L` with every original panel
member selected anywhere in the next `p-1` pivots, ignoring the intervening
misses. The full attacked-pool projection charges measured alpha+rho update
work and leaves FTRAN sequential:

```text
C_projected = (C_panel + C_minor) / L + C_F
G_pool      = C_pool / C_projected.
```

| p | G_batch | L | G_oper | Any-hit members | Any-hit survival x batching | Projected pool us/pivot | Projected pool speedup | Overall wall reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.906x | 1.166 | 0.528x | 1.188 | 0.538x | 83.95 us | **0.590x** | -37.3% |
| 4 | 1.434x | 1.281 | 0.459x | 1.572 | 0.564x | 99.95 us | **0.495x** | -54.6% |
| 8 | 2.158x | 1.348 | 0.364x | 2.267 | 0.612x | 128.66 us | **0.385x** | -85.6% |

The kill is robust to much more optimistic assumptions:

1. Ignoring update cost and intervening misses, the best measured any-hit
   survival-times-batching factor is only **0.612x**.
2. Replacing the measured kernel with an ideal width-p batch makes the any-hit
   factor equal to the number of used members. The best is still only
   **2.267x at p=8**, below 2.8x.
3. FTRAN is sequential in the reconciled shadow-panel mechanism; including it
   can only reduce the attacked-pool speedup.

Therefore the scalar-endpoint shadow-panel class tested by probe P-A does not
earn implementation funding. The probe did not test changed-endpoint
simultaneous rank-k exchanges.

## Reproduction

```bash
cd /home/evan/dev/linprogx-pa
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_dual_simplex.py -q
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/blockds_shadow_probe.py
```

Raw run output is written to `/tmp/blockds_shadow_probe_2026_07_18.json`.
