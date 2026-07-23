# Exact next-pivot BTRAN + pivot-row lookahead falsifier — 2026-07-21

## Verdict

**KILLED at S0 exactness; no concurrency or production implementation was
started.** The rank-one transformation predicts the next Dantzig leaving row
perfectly on greenbea, including across ordinary refactorizations, but its
floating-point pivot row changes the baseline Harris entering choice on 15 of
4,398 predictions. That crosses the campaign's immediate trajectory kill gate.

The read-only diagnostic itself does not perturb the solver. All five paired
baseline/traced runs returned the same status, iteration count, objective, and
pivot hash. The environment knob is absent from the result when off.

Raw measurements are in
`/tmp/lookahead_btran_greenbea_2026_07_21.json`; the syscall-level network audit
is `/tmp/lookahead_btran_network_audit.strace`.

## Probe

For a pivot replacing basis column `r`, with `d = B^-1 a_q` and `B' = B E`, the
probe evaluates the next row `s` against the old factorization:

```text
s != r: rho' = B^-T e_s - (d_s / d_r) rho
s == r: rho' = rho / d_r
```

It then scatters `rho'` through the immutable scaled CSR matrix to form the
complete structural/artificial pivot row. The probe runs only with
`LINPROGX_DS_LOOKAHEAD=1` and `leaving_rule=1`; otherwise no lookahead buffers,
work, or result key exist. It is strictly read-only: the ordinary BTRAN, pivot
row, Harris selection, and pivot remain authoritative.

At the next iteration it compares:

- predicted versus actual Dantzig leaving row;
- dense `rho` and dense structural/artificial pivot row;
- a read-only replay of the baseline Harris pass-1, minimum-ratio flip sweep,
  and pass-2 entering choice;
- refactorization and recomputation boundary survival.

The full greenbea shape is 2,392 x 5,598 before presolve and 1,525 x 3,868 after
presolve. Configuration: `max_iter=50000`, `tol=1e-8`, `expand=1`,
`leaving_rule=1`.

## S0 exactness result

| Measure | Result |
|---|---:|
| Predictions | 4,398 |
| Dantzig leaving matches | 4,398 / 4,398 |
| Invalidations | 0 |
| Refactor predictions / survivals | 33 / 33 |
| Recompute-boundary predictions | 94 |
| `rho` mismatches above 1e-9 | 60 |
| `rho` mismatches at refactors | 2 |
| Pivot-row mismatches above 1e-8 | 25 |
| Pivot-row mismatches at refactors | 0 |
| Maximum absolute `rho` error | 7.495284080505371e-6 |
| Maximum absolute row error | 7.495284080505371e-6 |
| Harris entering-choice mismatches | **15** |
| Flip-set mismatches | 0 |

The error is not a refactor-only phenomenon: 58 of 60 `rho` mismatches and all
25 row mismatches occur away from ordinary refactorizations. The algebra is
exact over the reals, but evaluating an old-factor BTRAN plus a rank-one
correction follows a different floating-point path from the next iteration's
updated-factor BTRAN. Harris's narrow ratio band turns that drift into 15
different entering choices. A consumer using the predicted row would therefore
change the pivot hash; recomputing the row to preserve the hash eliminates the
lookahead's purpose.

## Trajectory and original-space correctness

All five alternating pairs were identical on the authoritative trajectory:

| Measure | Baseline and diagnostic |
|---|---:|
| Status | `optimal` |
| Iterations | 4,399 |
| Pivot hash | `1054624160779546655` |
| Original objective | -72,555,248.1298459 |
| Maximum original residual | 1.7688915798785843e-7 |
| Maximum lower-bound violation | 3.857486786583777e-12 |
| Maximum upper-bound violation | 4.547473508864641e-13 |

This equality demonstrates only that instrumentation is read-only. The 15
shadow Harris mismatches demonstrate that consuming its output would not be.

## S1 economics (counterfactual only)

Paired direct-native medians:

- baseline: 0.628711 s (individual runs 0.612691, 0.628711, 0.696221,
  0.722623, 0.594056 s);
- sequential diagnostic: 1.262687 s (1.227336, 1.262687, 1.281868,
  1.233099, 1.377869 s), or +100.84%. This is expected because the falsifier
  computes both rows and validates them serially.

Median baseline phase totals were 108.744 ms in `btran_rho`, 123.273 ms in
`pivot_row`, 71.669 ms in reduced-cost update, 0.828 ms in bookkeeping,
3.204 ms in pricing update, 45.455 ms in LU update, and 47.302 ms in refactor.
The diagnostic's median shadow work was 59.291 ms for next-leaving scan,
71.858 ms for old-factor BTRAN, 10.000 ms for correction, and 98.184 ms for CSR
scatter; read-only validation added another 42.332 ms.

An aggressively favorable launch immediately after the primal update, while
locally accounting for the known basis replacement, exposes about 171.669 ms
of reduced-cost/bookkeeping/pricing/LU/tail/refactor work. Because all 33
refactor predictions survive, that is a 27.31% gross overlap window against
the 0.628711 s baseline. The necessary shadow path is about 239.333 ms before
validation, exceeding the window by 67.664 ms. Against the 232.017 ms ordinary
`btran_rho + pivot_row` slice, the zero-startup, zero-contention ceiling is
roughly 164.353 ms, or 26.14%.

That ceiling is both optimistic and narrow: it assumes free thread launch and
join, no cache or memory-bandwidth interference, and an earlier launch point
than the sequential falsifier. Launching only after ordinary basis bookkeeping
leaves about 95 ms including refactor, already below the 20% whole-wall gate.
The prior HEAD accounting similarly bounded the overlap at 150.81 ms (25.6%)
before overhead. Economics do not rescue the failed exactness gate, so S2 was
not authorized or attempted.

## Validation and command log

No source, binary, documentation, or behavior from an external solver was
read. All dependency/build/test commands used `UV_OFFLINE=1`; the network trace
contains no `AF_INET` or `AF_INET6` syscall.

Build and validation commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv pip install --reinstall -e . --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run pytest tests/test_dual_simplex.py -k lookahead -q
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run python -m experiments.lookahead_btran_probe --pairs 5 --out /tmp/lookahead_btran_greenbea_2026_07_21.json
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run pytest tests/test_dual_simplex.py tests/test_samples_compare.py -q
strace -f -e trace=network -o /tmp/lookahead_btran_network_audit.strace env UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run pytest tests/test_dual_simplex.py -k lookahead -q
rg -n 'AF_INET|AF_INET6' /tmp/lookahead_btran_network_audit.strace
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run ruff check experiments/lookahead_btran_probe.py tests/test_dual_simplex.py
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run ruff format experiments/lookahead_btran_probe.py tests/test_dual_simplex.py
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv run ruff format --check experiments/lookahead_btran_probe.py tests/test_dual_simplex.py
```

The required two-file test run passed **52 tests**. The focused test passed, and
Ruff check plus format-check passed. The first direct invocation
`uv run python experiments/lookahead_btran_probe.py ...` failed before solver
execution because package-style experiment imports require `python -m`; the
successful module command above is the reproducible form. During
characterization, the focused test was intentionally run red before the result
payload existed, then green after the payload was wired.

## Changed files

- `src/linprogx/_csparse.c`: default-off read-only transformation,
  comparison, semantic Harris replay, timing, and counters.
- `tests/test_dual_simplex.py`: focused environment-gating and non-perturbation
  characterization test.
- `experiments/lookahead_btran_probe.py`: alternating greenbea measurement and
  original-space verification driver.
- `experiments/lookahead_btran_falsifier_2026_07_21.md`: this evidence report.

The C and test changes are diagnostic artifacts, not a production lookahead
implementation.
