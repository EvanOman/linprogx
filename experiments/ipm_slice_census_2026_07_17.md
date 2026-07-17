# IPM slice census — 2026-07-17

## Scope

Instrumentation only. `LINPROGX_IPM_SLICE=1` adds an `ipm_slice_us` mapping to
the native IPM result. With the variable absent, the key is absent and no
clocks are read. The six exclusive buckets are:

- `setup_order`: native input allocation/copy, Ruiz scaling, normal-equation
  pattern construction, minimum-degree ordering, order evaluation, and the
  assembly map.
- `symbolic`: etree/column counts/factor structure plus any lazy supernode
  symbolic build.
- `refactor`: numeric normal-equation assembly and factorization. Triangular
  solves performed as part of numeric refactorization remain in this bucket.
- `triangular_solves`: IPM backsolves outside numeric refactorization.
- `matvecs_residuals`: scaled-operator and normal-equation matvecs plus the
  main-loop residual/slack/norm scan.
- `other`: all remaining measured native-call wall, including vector scans,
  RHS/step/update work, certificate/polish work, result marshalling, and free.

## Local measurements

The extension was rebuilt with the requested commands. Each table cell is the
median of five isolated direct-IPM worker runs. Parentheses are the bucket's
share of the sum of the six independently computed bucket medians.

| instance | setup/order | symbolic | refactor | triangular solves | matvecs/residuals | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| woodw | 6.385 ms (8.8%) | 0.614 ms (0.8%) | 47.078 ms (64.6%) | 6.512 ms (8.9%) | 5.432 ms (7.5%) | 6.874 ms (9.4%) |
| 80bau3b | 8.070 ms (7.9%) | 0.828 ms (0.8%) | 52.491 ms (51.7%) | 10.847 ms (10.7%) | 9.035 ms (8.9%) | 20.254 ms (20.0%) |
| cre_a | 15.402 ms (20.3%) | 0.980 ms (1.3%) | 37.016 ms (48.7%) | 9.678 ms (12.7%) | 6.058 ms (8.0%) | 6.894 ms (9.1%) |
| pilot87 | 106.389 ms (3.5%) | 7.078 ms (0.2%) | 2,293.830 ms (74.5%) | 406.153 ms (13.2%) | 196.663 ms (6.4%) | 69.858 ms (2.3%) |

| instance | native wall median | slice-median sum | sum / wall | status | iterations | native wall runs |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| woodw | 72.966 ms | 72.895 ms | 99.90% | optimal | 32 | 72.966, 78.396, 73.330, 69.987, 71.899 ms |
| 80bau3b | 100.780 ms | 101.525 ms | 100.74% | optimal | 44 | 100.780, 113.998, 99.162, 104.400, 97.879 ms |
| cre_a | 76.215 ms | 76.028 ms | 99.75% | optimal | 34 | 75.248, 81.285, 75.879, 81.999, 76.215 ms |
| pilot87 | 3,099.298 ms | 3,079.970 ms | 99.38% | optimal | 128 | 2,965.488, 3,042.634, 3,154.653, 3,850.690, 3,099.298 ms |

The slight over-100% value for `80bau3b` is from summing six independent
bucket medians and comparing that sum with an independently computed wall
median. Per invocation, the profiler buckets are exclusive.

## Gates

1. Attribution: pass. The slice-median sums are 99.38%–100.74% of median
   native IPM wall on `woodw`, `80bau3b`, `cre_a`, and `pilot87`; every result
   is optimal.
2. Knob-off path: pass. Nine alternating off/on `woodw` direct-IPM pairs
   produced one identical result SHA-256 in both arms:
   `bc43b1ef92b9c891ccef1309044bb640829bbe9c9f30e73400ca7122ae2a0f14`.
   Off median was 72.541 ms; on median was 72.969 ms; on/off was 1.00590.
   The focused test also compares the full C result, including `x` and `y`,
   exactly after removing only `ipm_slice_us`.
3. Worker JSON: pass. With the env set, each requested fixture returned
   `ipm_slice_us` from `experiments/suite_bench.py --worker`; without it, the
   field is absent. The public result path carries the mapping through
   `SparseSolveResult` without exposing stale IPM data after a fallback.
4. Full suite: pass. `378 passed, 7 skipped in 30.69s`.

Repository CI also passed: `just ci` completed lint, format, type, Bandit,
`pip-audit` (`No known vulnerabilities found`), and coverage-gated tests.
Coverage was 88.77% against the 85% floor; its test tail was
`378 passed, 7 skipped in 35.87s`.

## On-host capture

Use the Modal env-A/B protocol so the off arm measures instrumentation
overhead and every raw B-arm worker result contains the slice mapping:

```bash
uvx modal run tools/modal_bench.py --action bench --mode envab \
  --ref <instrumented-ref> \
  --instances lp_woodw,lp_80bau3b,lp_cre_a,lp_pilot87 \
  --pairs 7 --hosts 3 \
  --env-a "" --env-b "LINPROGX_IPM_SLICE=1"
```

The exact underlying worker form is:

```bash
LINPROGX_IPM_SLICE=1 PYTHONPATH=. uv run python \
  experiments/suite_bench.py --worker /fixtures/lp_woodw.mat linprogx
```

Replace the fixture name for the other three instances. In the three-host
Modal command, raw `host_results[*].envab[instance].pair_results[*].lxB`
objects contain `ipm_slice_us` automatically.

## Files touched

- `src/linprogx/_csparse.c`
- `src/linprogx/sparse.py`
- `experiments/suite_bench.py`
- `experiments/ipm_other_profile.py`
- `tests/test_ipm.py`
- `experiments/ipm_slice_census_2026_07_17.md`

## Pytest tail

```text
tests/test_samples_compare.py ........................                   [ 77%]
tests/test_simplex_lu.py .......................                         [ 83%]
tests/test_solver.py ..................                                  [ 88%]
tests/test_sparse.py ..............................................      [100%]

======================= 378 passed, 7 skipped in 30.69s ========================
```
