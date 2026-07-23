# greenbea IPM-warm-started dual simplex — G2 falsifier

## Verdict

**KILLED.** Neither crossover design gets close to the `<0.35 s` kill bar,
let alone the `<0.30 s` live gate. The best certificate-backed pipeline is
the Bixby-style crash at `k=30`: `0.583 s` end to end, 4,766 DS pivots, status
`optimal`. Its DS stage alone is `0.465 s`, so eliminating IPM and crossover
cost entirely would still miss the kill bar.

The clean local cold reference is 4,399 pivots, `0.414 s`, and `94.2 us/pivot`.
The best accepted warm start therefore does **367 more pivots (+8.3%)** and
takes 41% more wall end to end. The smallest warm pivot count observed is
4,489 at `k=49`, still 90 pivots above cold, and that run exits
`dual_infeasible` rather than producing the required certificate.

No solver source code or network access was used. The fixture is
`/tmp/lpsuite/lp_greenbea.mat`; linprogx presolve produces 1,525 rows, 3,868
columns, and 23,274 nonzeros. Accuracy remained `eps=2e-5`; partial IPM ran
with `tol=1e-9`, and final DS ran its normal stricter `tol=1e-8` certificate
path.

## Extraction and crossover rules

The production-candidate extraction rule is global, not selected from the
greenbea k sweep:

> If IPM reaches a non-optimal native safety termination—the first nonfinite
> merit, the existing iteration-60 pace watchdog, or the global 200-iteration
> cap—extract the solver's already-maintained best
> `max(primal residual, dual residual, mu)` snapshot. A certified IPM result
> never crosses over.

For greenbea the normal safety stop is iteration 59, after the iteration-58
best snapshot (`mu=3.013e-9`); the read-only G1 anatomy independently locates
the degradation at a wholly nonfinite Newton direction after iteration 58.
The k sweep is diagnostic and includes earlier iterates rather than changing
the rule per problem.

Nonbasic structural columns are assigned to the nearest finite bound. Free
columns use the hook's free status, fixed columns use fixed status, and
nonbasic artificial columns are fixed at zero.

- Attempt 1, `superbasic_top_m`: choose the `m` structural columns with the
  largest absolute distance to their nearest bound. Every slice is singular,
  exhausts 31 hook repairs, and falls back to the artificial identity.
- Attempt 2, `bixby_iterate_crash`: prioritize columns by the same iterate
  distance, but accept a structural column only when it has exactly one
  uncovered row and its pivot is at least half the column maximum. Fill
  uncovered rows with identity artificials. Every slice has 1,372 structural
  and 153 artificial basics, with zero singular repairs and no fallback.

`LINPROGX_IPM_CROSSOVER_SLICE=1` is a diagnostic-only C gate that suppresses
IPM certificate cleanup for a deliberately truncated run. This makes the
partial iteration wall honest; the injected DS is still the only accepted
certificate. The default IPM path is unchanged when the gate is absent.

## k sweep

Times are single clean local measurements in seconds. `total` is measured
IPM + crossover + warm DS. Exact data, including phase slices, densities,
residuals, warm-hook metadata, and objective errors, is in
`/tmp/greenbea-warmstart/results.json`.

### Attempt 1 — largest-distance super-basic set

| k | IPM | crossover | DS pivots | us/pivot | total | DS status |
|---:|---:|---:|---:|---:|---:|:---|
| 20 | 0.051 | 0.009 | 7,607 | 101.0 | 0.828 | optimal |
| 30 | 0.068 | 0.009 | 7,603 | 100.5 | 0.841 | optimal |
| 35 | 0.078 | 0.009 | 7,603 | 100.8 | 0.853 | optimal |
| 40 | 0.088 | 0.009 | 7,605 | 101.8 | 0.870 | optimal |
| 45 | 0.097 | 0.009 | 7,605 | 102.4 | 0.884 | optimal |
| 48 | 0.102 | 0.008 | 7,605 | 102.5 | 0.890 | optimal |
| 49 | 0.104 | 0.009 | 7,605 | 101.7 | 0.886 | optimal |
| 50 | 0.117 | 0.011 | 7,605 | 104.7 | 0.924 | optimal |
| 55 | 0.123 | 0.009 | 7,605 | 103.7 | 0.920 | optimal |
| 59 | 0.131 | 0.009 | 7,605 | 102.1 | 0.916 | optimal |
| 60 | 0.128 | 0.009 | 7,605 | 102.6 | 0.917 | optimal |

These are not successful super-basic bases: all rows report
`singular_repairs=31` and `fell_back_to_identity=1`. Imported nearest-bound
statuses explain the small pivot-count differences after the common fallback.

### Attempt 2 — iterate-prioritized Bixby crash

| k | IPM | crossover | DS pivots | us/pivot | total | DS status |
|---:|---:|---:|---:|---:|---:|:---|
| 20 | 0.051 | 0.050 | 4,865 | 99.7 | 0.586 | optimal |
| 30 | 0.068 | 0.050 | 4,766 | 97.5 | 0.583 | optimal |
| 35 | 0.078 | 0.053 | 5,348 | 101.3 | 0.672 | optimal |
| 40 | 0.088 | 0.052 | 5,231 | 101.6 | 0.671 | dual_infeasible |
| 45 | 0.097 | 0.051 | 5,169 | 95.1 | 0.639 | dual_infeasible |
| 48 | 0.102 | 0.050 | 5,059 | 96.1 | 0.638 | dual_infeasible |
| 49 | 0.104 | 0.052 | 4,489 | 98.1 | 0.596 | dual_infeasible |
| 50 | 0.117 | 0.051 | 4,759 | 97.8 | 0.633 | dual_infeasible |
| 55 | 0.123 | 0.063 | 4,870 | 93.9 | 0.643 | optimal |
| 59 | 0.131 | 0.051 | 5,412 | 98.7 | 0.717 | dual_infeasible |
| 60 | 0.128 | 0.054 | 5,412 | 97.4 | 0.709 | dual_infeasible |

Unlike the prior HiGHS transfer, this crash does not consistently densify
DS solves: its range is 93.9–101.6 us/pivot around the 94.2 us/pivot cold
reference. The failure is more basic: it never saves pivots, and the natural
k=60 safety-stop rule does not certify.

## Certificate and determinism gates

The best accepted row (`bixby_iterate_crash`, `k=30`) has:

- final status `optimal`, with the normal DS exit certificate;
- original objective `-72555248.12983723` versus oracle/cold
  `-72555248.12984590` (relative error `1.20e-13`);
- maximum original equality residual `4.69e-7`;
- maximum original bound violation `8.39e-12`;
- warm hook `used=1`, zero repairs, and no identity fallback.

The same global rule was checked on the other named IPM fixtures with the
diagnostic extraction gate removed:

| fixture | normal IPM status | iterations | wall | crossover triggered? |
|:---|:---|---:|---:|:---|
| woodw | optimal | 32 | 0.070 s | no |
| 80bau3b | optimal | 44 | 0.102 s | no |
| cre_a | optimal | 34 | 0.089 s | no |

All selection orders and ties are deterministic (distance descending, column
index ascending); there is no random seed or per-problem parameter.

## Build and validation

The environment was rebuilt offline to preserve the campaign's network ban:

```text
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation
```

Focused pytest tail:

```text
tests/test_sparse.py ...                                                 [ 60%]
tests/test_dual_cleanup.py .                                             [ 80%]
tests/test_ipm.py .                                                      [100%]
============================== 5 passed in 1.57s ===============================
```

The focused set covered sparse IPM, sparse DS, DS/IPM agreement, cre_a IPM
certificate cleanup, and `LINPROGX_IPM_SLICE` numerical inertness. Repository
lint, format check, the probe's focused type check, Bandit, and pip-audit pass.
The repository-wide type recipe is blocked by a pre-existing unrelated
optional-objective diagnostic at `experiments/rr_falsifier_probe.py:364`.

The shipped highspy two-way transfer script could not be rerun after the
offline environment rebuild because `highspy` is not in the project dev extra
or any existing sibling environment. It was not installed because network and
install access are prohibited. This probe uses the already-validated internal
mapping and records the native hook's `used`, repair, fallback, and imported
status fields on every run.

## Files touched

- `src/linprogx/_csparse.c` — diagnostic-only partial-IPM extraction gate.
- `experiments/greenbea_warmstart_probe.py` — deterministic two-design sweep,
  certificate checks, timing, and cross-fixture rule check.
- `experiments/greenbea_warmstart_2026_07_18.md` — this report.
