# INT — combined K4 Harris + K2-safe PRICE integration (2026-07-19)

## Verdict: SHIP-CANDIDATE

The integrated default-on SIMD path clears the combined cold gate with an
**11.30% paired end-to-end reduction**, preserves the 4,399-pivot cold and
3,334-pivot B* trajectories exactly, and has no greater-than-1% regression on
the DS fixture battery. The B* trajectory improves **10.41%**.

Only K2's safe leaving/PRICE scan was ported. The numerically invalid pivot-row
gather/scatter experiment (reported residual `2.021e9`) is absent.

## Integrated implementation

- K2-safe: four-basis-row AVX2 leaving/PRICE scan for Dantzig and exact-DSE,
  including deterministic lowest-basis-position tie resolution.
- K4: AVX2 Harris pass 1 and pass 2 for the ordinary `bfrt=0` ratio test,
  including ascending candidate compaction and deterministic lowest-column
  tie resolution. BFRT remains on its historical scalar path.
- One global gate: SIMD is on by default; `LINPROGX_DS_SIMD=0` disables both
  kernels and reaches the historical scalar loops.
- Runtime dispatch still requires AVX2. A global workload-shape guard keeps
  both kernels scalar above 8,192 basis rows. This is independent of problem
  identity and confines the branchless kernels to the cache-resident scan
  regime they were designed for.
- `LINPROGX_DS_PIVOT_TRACE=1` exposes K4's diagnostic hash over committed
  `(leaving position, leaving column, entering column)` triples. It is not set
  during timing.

The 8K guard was added falsifier-first. Without it, the first complete
integration campaign improved every target except stocfor3, which regressed
4.37%. Isolated measurements showed why: on stocfor3's 13,864-row, 4.3-nnz/row
presolved system, K4 increased its ratio slice by 27.4% and K2 increased its
PRICE slice by 5.1%. Every profitable fixture has only 707–4,068 presolved
basis rows. With the global guard, stocfor3 executes the same scalar path in
both arms.

## Measurement protocol

- Fixtures: `/tmp/lpsuite`; retained B* basis:
  `/tmp/phase1-predictions/results.json`
- Solver: presolved native equality-box dual simplex, Dantzig leaving,
  `expand=1`, `bfrt=0`, `tol=1e-8`
- Certificate gate: original-space equality and bound residuals at
  `eps=2e-5`
- Schedule: untimed warmup of both arms, then nine foreground A/B pairs per
  trajectory/fixture; arm order alternated by repetition and scenario
- CPU placement: `taskset -c 4`
- SIMD arm: `LINPROGX_DS_SIMD` unset (default on)
- Scalar arm: `LINPROGX_DS_SIMD=0`
- Raw final results: `/tmp/int_kernel_combined_2026_07_19.json`
- Reproducer: `experiments/int_kernel_combined_probe.py`

The reported percentage is `1 - median(on_i / off_i)` over the nine paired
runs. Arm wall columns are their separate medians, so their quotient need not
equal the paired percentage exactly.

## Combined A/B results

| trajectory / fixture | SIMD selection | pivots | knob-off median | default-on median | paired reduction |
|---|---|---:|---:|---:|---:|
| greenbea cold | K2 + K4 | 4,399 | 0.546939 s | 0.496896 s | **11.30%** |
| greenbea B* | K2 + K4 | 3,334 | 0.519348 s | 0.470510 s | **10.41%** |
| woodw | K2 + K4 | 1,338 | 0.118550 s | 0.114266 s | **3.84%** |
| stocfor3 | scalar fallback (`m=13,864`) | 9,604 | 1.824394 s | 1.798147 s | **0.98%** (noise) |
| 80bau3b | K2 + K4 | 6,758 | 0.587588 s | 0.551313 s | **6.36%** |
| cre_d | K2 + K4 | 46,048 | 22.599151 s | 20.695560 s | **8.41%** |

The cold result exceeds the required 8% gate by 3.30 percentage points. No
DS-routed fixture regressed; the worst final paired result is the identical-
path stocfor3 control at -0.98% wall (measurement noise in the favorable
direction).

## Identity and certificate evidence

Every arm returned certificate-backed `optimal`. For every row below, scalar
and SIMD have identical pivot hash, pivot count, reduced objective bits,
reduced-x SHA-256, terminal basis SHA-256, and bound-status SHA-256.

| trajectory / fixture | pivots | committed-pivot hash | reduced objective | max original equality residual | max bound violation |
|---|---:|---:|---:|---:|---:|
| greenbea cold | 4,399 | `1054624160779546655` | -72,557,668.26492292 | 1.77e-7 | 3.86e-12 |
| greenbea B* | 3,334 | `10455462928261366500` | -72,557,668.26492676 | 4.77e-7 | 2.88e-12 |
| woodw | 1,338 | `1988618232812448208` | 1.304476333084228 | 5.68e-14 | 0 |
| stocfor3 | 9,604 | `3030070037749976210` | -39,976.7839436495 | 1.82e-12 | 1.62e-14 |
| 80bau3b | 6,758 | `7552528144506851068` | 529,334.8845520116 | 1.73e-12 | 5.68e-14 |
| cre_d | 46,048 | `11983623201522147527` | 24,454,969.764549237 | 1.14e-13 | 5.00e-14 |

### Byte-identical knob-off

Before the first C edit, the baseline build wrote a canonical byte artifact
covering all six rows: status, iteration count, IEEE-754 objective/residual
bytes, reduced-x bytes hash, basis bytes hash, and bound-status bytes hash.
The final build with `LINPROGX_DS_SIMD=0` reproduced it byte-for-byte:

```text
pre-change SHA-256:  6d33a5672fbe66322df56e3f5b06fede5fe325dbbe49db39da3cb7aaf945c7eb
final knob-off SHA:  6d33a5672fbe66322df56e3f5b06fede5fe325dbbe49db39da3cb7aaf945c7eb
```

## Gate results

| gate | result | evidence |
|---|---|---|
| Combined cold end-to-end reduction >= 8% | **PASS** | 11.30% paired reduction |
| No DS fixture regresses > 1% | **PASS** | worst final result is stocfor3 at -0.98% |
| Cold trajectory identity | **PASS** | 4,399 pivots and hash `1054624160779546655` |
| B* trajectory identity | **PASS** | 3,334 pivots and hash `10455462928261366500` |
| Every DS fixture trajectory identity | **PASS** | all pivot/objective/x/basis/status hashes match |
| Certificate residuals <= `2e-5` | **PASS** | all timed and trace runs certified |
| Knob-off byte-identical to pre-change | **PASS** | canonical artifact SHA-256 matches |
| Full pytest | **PASS** | 522 passed, 7 skipped in 69.59 s |
| `just ci` minus pip-audit | **PASS** | Ruff lint/format, ty, Bandit, coverage suite |
| Coverage floor >= 85% | **PASS** | 89.16% total; 522 passed, 7 skipped |

## Offline build and validation

The prescribed build was run before baseline capture and the extension was
reinstalled after C changes, with `UV_OFFLINE=1` mechanically enforcing the
no-network rule:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation
```

Validation results:

- Full pytest: **522 passed, 7 skipped in 69.59 s**.
- Coverage pytest: **522 passed, 7 skipped in 59.41 s**, total **89.16%**.
- Ruff lint: passed.
- Ruff format check: passed after formatting the new probe.
- `ty check`: passed.
- Bandit medium-and-higher scan: passed.
- `pip-audit` was not run because it is network-capable and the campaign
  forbids network access.
- No network access, competing-solver source inspection, or Git operation was
  used.

**Final verdict: SHIP-CANDIDATE.**
