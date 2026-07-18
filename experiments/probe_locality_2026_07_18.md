# Probe P-E — support-contiguity memory reordering (2026-07-18)

## Verdict: KILLED

The class fails the measurement gate, so no solver permutation and no wall-clock
A/B were implemented.

- On the unchanged 4,399-pivot greenbea trajectory, **95.91–96.92%** of the
  recorded indirect accesses hit a cache line already touched in the same
  kernel call. The current working order is already strongly line-resident.
- A deterministic one-shot hot-support order, built by sorting indices by the
  number of sampled calls containing them, predicts a **0.81% slowdown** on the
  weighted BTRAN/FTRAN/pivot-row pool. It destroys more traversal/prefetch order
  than it recovers through packing.
- Even an impossible **per-call oracle** that repacks every call's exact support
  contiguously predicts only **1.26% kernel-cost reduction** in the cache-resident
  L1/L2 model, versus the required 15%.
- That oracle reduces the count of L1 line fills by 16.32%, but this is not a
  kernel-speed prediction: it assumes line fills are the only cost and all hits
  are free. With 4-cycle L1 hits and 12-cycle L2 hits, the gain is 1.26%; even a
  deliberately hostile 200-cycle miss costs only projects 10.92%. A miss would
  have to cost **1,112 cycles** before the oracle reaches 15%, incompatible with
  P-B's measured cache-resident working set.

The strongest implementable model is below the 15% gate, and the oracle ceiling
is far below it under any cache-resident latency. Per the probe contract, the
real-permutation stage and alternating median-of-9 wall A/B were not reached.

## Contract and setup

Inputs were the class-E research plan, Claude Opus idea 3, GLM-5.2 idea 4,
Codex-contrarian idea 4, the greenbea dossier, and P-B's precision result. The
load-bearing prior is P-B: fp32 containers produced only 0.98–1.18x kernel gains,
so raw bandwidth cannot justify this class. P-E must win through cache-line
utilization and prefetch friendliness.

The prescribed reduced problem and solver route were used:

| Item | Value |
|---|---:|
| Fixture | `/tmp/lpsuite/lp_greenbea.mat` |
| Presolved shape | 1,525 x 3,868 x 23,274 |
| Solver | dual simplex, Dantzig leaving, EXPAND on, shipped FT path |
| Status | optimal |
| Pivots | 4,399 |
| Objective | -72,555,248.1298459 |
| Original-space equality residual | 1.76889e-7 |
| Maximum bound violation | 3.85749e-12 |
| Clean pinned wall (context only) | 0.448683 s |

The build ran offline with the required `/tmp/uv-cache` commands. `perf` is on
the box, but hardware counters are unavailable to this user because
`perf_event_paranoid=4`; `perf stat` exits before measurement. No network access,
external solver-source inspection, Git operation, or per-instance parameter was
used.

## Actual access trace

`LINPROGX_DS_LOCALITY_TRACE=/path` enables a read-only binary trace in the C DS
path. It samples every 16th pivot, spanning the whole trajectory and all
refactor intervals. The trace records only the indirect workspace accesses that
a relabeling can change:

- factor-coordinate work-vector indices in BTRAN and FTRAN;
- `rho` and CSR row-pointer indices in pivot-row formation;
- pivot-row accumulator and touched-set column indices.

Sequential packed factor values, matrix values, and append-only pattern arrays
are excluded because relabeling cannot reduce their line count. The final trace
contains 275 calls of each kernel, 8.068 million indirect accesses, and occupies
64,554,608 bytes. Trace on/off produced identical status, iterations, objective,
and complete reduced `x` vector.

The existing rate histogram independently reproduces the dossier's support
statistics:

| Support statistic | Measured value |
|---|---:|
| rho p50 | 897 |
| pivot-row alpha p50 | 3,625 |
| ratio candidates p50 | 182 |
| consecutive alpha overlap, support-weighted | 99.622% |
| consecutive alpha overlap, median of smaller support | 100.000% |

The access streams themselves are already highly line-reusing:

| Kernel | Sampled calls | Indirect accesses | Already-touched line hits | L1-model hits | Forward/same-line transitions |
|---|---:|---:|---:|---:|---:|
| BTRAN | 275 | 1,016,998 | 96.236% | 96.236% | 61.651% |
| FTRAN | 275 | 1,352,652 | 96.921% | 96.921% | 48.081% |
| Pivot row | 275 | 5,698,020 | 95.908% | 95.335% | 80.813% |

The small difference between already-touched and L1-model hits on pivot row is
capacity eviction across its several work arrays. BTRAN/FTRAN's factor work
vector is only about 12 KiB, so every line touched once in a call remains in the
32 KiB L1 model.

## Cache model

The model uses the measured host's private-cache geometry: 64-byte lines,
32 KiB L1d and 512 KiB L2 per core (AMD Ryzen 5 3600). Each kernel call begins
with a cold 512-line fully-associative LRU cache. This is favorable to the idea:
it discards cross-call persistence despite the 99.622% support-weighted overlap.
Costs are 4 cycles for an L1 hit and 12 cycles for a cache-resident L2 hit.

Three layouts replay the exact same logical access stream:

1. **Baseline:** inherited physical indices.
2. **Static hot pack:** a generic deterministic rule, sorting each logical
   allocation by sampled call-presence frequency, then original index. This is
   privileged relative to a real pre-solve structural ordering because it sees
   the trajectory first; it is therefore favorable as a falsifier.
3. **Per-call oracle:** each call gets its own perfect first-touch packing. This
   cannot be implemented by a one-shot permutation and is only an upper bound.

### Predicted reorder upside

| Kernel | Static line fills | Static modeled cost | Oracle line fills | Oracle modeled cost | Baseline -> oracle prefetchable |
|---|---:|---:|---:|---:|---:|
| BTRAN | **+10.33%** | **+0.72%** | -8.42% | -0.59% | 61.65% -> 71.54% |
| FTRAN | **+11.33%** | **+0.66%** | -11.27% | -0.65% | 48.08% -> 57.18% |
| Pivot row | **+11.46%** | **+0.98%** | -25.99% | -2.22% | 80.81% -> 82.71% |
| **Weighted attacked pool** | **+11.07%** | **+0.81%** | **-16.32%** | **-1.26%** | — |

Weights are the dossier shares BTRAN 18.9%, FTRAN 17.9%, and pivot row 24.8%,
normalized over their 61.6% attacked pool. Applying the oracle's 1.26% kernel
reduction to that pool projects only **0.78% end-to-end wall reduction** (about
3.5 ms on the clean local wall), before permutation setup, factor repacking, or
tie-preservation machinery.

The apparent tension between 16.32% fewer oracle line fills and 1.26% modeled
cost reduction is the point of P-E: more than 95% of accesses are already hits,
so the remaining misses are not a large enough fraction of kernel work.

### Miss-penalty sensitivity (impossible per-call oracle)

| L1-miss cost | Weighted attacked-kernel reduction |
|---:|---:|
| 12 cycles (L2-resident primary model) | 1.264% |
| 40 cycles (L3-like) | 4.459% |
| 200 cycles (deliberately DRAM-like) | 10.924% |
| 1,000 cycles | 14.865% |
| 2,000 cycles | 15.559% |

The interpolated 15% crossing is 1,112 cycles per L1 miss. P-B's results and
the 12–43 KiB target work arrays rule out that regime.

## Why the real permutation was not run

The probe explicitly conditions implementation on at least 15% modeled gain in
the attacked kernels. The feasible one-shot model is negative, and the
cache-resident per-call oracle is only 1.26%. Implementing a symmetric problem
permutation would therefore test a mechanism whose favorable upper bound has
already failed by 13.74 percentage points. The requested trajectory checks and
median-of-9 alternating wall A/B are marked **not reached**, not silently
omitted.

## Reproduction and validation

```bash
cd /home/evan/dev/linprogx-pe
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run pytest \
  tests/test_dual_simplex.py::TestResultDict::test_locality_trace_is_env_gated_and_read_only -q
taskset -c 4 env PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run python experiments/locality_probe.py
```

Raw outputs are `/tmp/probe_locality_2026_07_18.bin` and
`/tmp/probe_locality_2026_07_18.json`. The maintained reproduction harness is
`experiments/locality_probe.py`.

Validation completed cleanly: Ruff lint and format check, `ty check`, Bandit,
and the coverage-gated suite (**523 passed, 7 skipped, 89.16% coverage**).
`pip-audit` was not run because it is network-capable and the probe contract
forbids network access.

**Final verdict: KILLED.** Support-contiguity reordering is not a live greenbea
locality multiplier under the measured cache-resident regime.
