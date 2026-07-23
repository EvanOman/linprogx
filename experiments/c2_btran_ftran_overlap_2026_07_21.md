# C2 — overlap BTRAN and FTRAN (2026-07-21)

## Verdict: KILLED

The hardware premise is live but the dual-simplex schedule cannot expose it.
On a captured 1,525-row greenbea basis, a diagnostic persistent two-thread
pair with both right-hand sides supplied in advance reduced combined solve
wall by **49.06%** at the paired median and produced byte-identical FTRAN and
BTRAN outputs.  Production cannot supply that oracle condition: the entering
column, hence the FTRAN right-hand side, is selected from the pivot row made by
BTRAN.  The legally schedulable overlap is therefore **0%**, below C2's 12%
combined-solve gate.

This is a causal kill, not a claim that two cores cannot overlap the factor
walks.  The microbenchmark demonstrates that they can when handed information
that the algorithm does not yet possess.

## Falsifier and dependency chain

The current dual-simplex pivot has this exact dependency order:

```text
x_B -> leaving position
    -> rho = B^-T e_leaving                         [BTRAN]
    -> pivot row alpha_row = rho^T A
    -> Harris admissibility/ratio test -> entering_col
    -> a_entering -> alpha_col = B^-1 a_entering   [FTRAN]
    -> primal/reduced-cost updates -> basis-factor update
```

Although `rho` and `alpha_col` are mathematically independent once both right-
hand sides are fixed, `a_entering` is not known until BTRAN's output has been
consumed by pricing.  Starting FTRAN beside BTRAN would require guessing the
entering column.  That is speculative multi-candidate pricing, not mandate C2,
and it would no longer guarantee that the useful FTRAN is the one run in
parallel.

Cross-pivot overlap is also unavailable: the next pivot's BTRAN uses the basis
factor after the current FTRAN result has been consumed by the factor update.
Thus neither a persistent pool nor single-core chase interleaving can put the
two production solves on the same critical-path interval without changing the
algorithm.

## Method

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`, reduced to 1,525 x 3,868.
- Solver: dual simplex, Dantzig leaving, EXPAND on, `bfrt=0`, tolerance `1e-8`.
- Certificate epsilon: fixed at `2e-5` in original space.
- Baseline: one warmup plus nine measured foreground runs pinned to CPU 4.
- Attribution: existing `LINPROGX_DS_SOLVE_SLICE=1` nested solve timers.
- Identity: `LINPROGX_DS_EXPORT_BASIS=1`; complete reduced-x, final basis, and
  bound-status buffers hashed on all nine runs.
- Hardware probe: final basis captured from that unchanged trajectory and
  refactorized into two identical contexts with independent scratch.  A
  persistent worker was created before timing; caller and worker were pinned
  to distinct physical cores (CPUs 4 and 5).  Eleven samples alternated
  sequential-first and parallel-first order, with 2,048 pairs per sample.
- FTRAN RHS: deterministic median-nnz nonbasic structural column (column 182,
  6 nonzeros).  BTRAN RHS: unit vector at basis position 762.
- Captured basis matrix: 1,525 x 1,525 x 7,734; fresh LU had 2,569 L nonzeros
  and 7,147 U nonzeros.

The microbenchmark deliberately uses a fresh final-basis factor rather than
the trajectory's changing Forrest-Tomlin states.  It is a favorable hardware
falsifier only; the production conclusion rests on the dependency chain.

## Measurements

### Unchanged greenbea baseline (median of nine)

| quantity | measured |
|---|---:|
| wall | 0.564857 s |
| pivots | 4,399 |
| BTRAN solve body | 0.071987 s |
| FTRAN solve body | 0.118576 s |
| combined solve body | 0.190562 s |
| combined share of wall | 33.74% |

The host was busy during this run, so the absolute wall is slower than the
dossier's ~0.37 s local number.  The result uses pinned, within-run ratios and
does not substitute this noisy absolute wall for the dossier's board baseline.

### Persistent two-thread oracle microbenchmark

| quantity | measured |
|---|---:|
| median sequential time, 2,048 pairs | 0.111199 s |
| median parallel time, 2,048 pairs | 0.059638 s |
| median paired combined reduction | **49.06%** |
| byte-identical outputs | 11 / 11 samples |

The paired reductions ranged from 35.38% to 60.89% except for normal host
variation within that range; every sample was a win.  The median comfortably
passes 12% under the artificial condition that both right-hand sides exist at
dispatch time.  It does not pass the production gate because that condition is
unavailable.

### Correctness and determinism

All nine baseline runs had the same 4,399 pivots, reduced objective
`-72,557,668.26492292`, original objective `-72,555,248.12984590`, complete
reduced-x hash, final-basis hash, and bound-status hash.  Every run returned
`optimal`; maximum original-space equality residual was **1.769e-7** and bound
violation was **3.857e-12**, both below `eps=2e-5`.

The diagnostic pair's parallel outputs were byte-identical to its sequential
outputs in all eleven samples.

## Gate and flip arithmetic

There are three distinct ceilings:

| case | combined-solve reduction | overall wall reduction | C2 >=12% solve gate | greenbea -18% need |
|---|---:|---:|---|---|
| oracle microbenchmark, captured fresh factor | 49.06% measured | not transferable | favorable only | not applicable |
| impossible perfect overlap of measured production totals | 37.78% | **12.74%** | yes | **short by 5.26 points** |
| legally schedulable production overlap | **0.00%** | **0.00%** | **FAIL** | **FAIL** |

The perfect-overlap row uses `max(BTRAN,FTRAN)` as the parallel critical path:
`1 - 0.118576 / (0.071987 + 0.118576) = 37.78%`.  Its maximum overall saving is
the hidden shorter solve divided by wall:
`0.071987 / 0.564857 = 12.74%`.  Even an impossible zero-overhead schedule would
not meet the dossier's 18% end-to-end need on this measurement.

The binding verdict is stronger: because FTRAN's RHS is produced downstream of
BTRAN, the actual overlap and saving are zero.  **C2 is KILLED.**

## Implementation and artifacts

- Probe: `experiments/c2_overlap_probe.py`
- Raw results: `/tmp/c2-overlap/results.json`
- Diagnostic C hook: `lu_pair_bench_test` in `src/linprogx/_csparse.c`
- Characterization test: `test_pair_benchmark_preserves_ftran_and_btran_results`
  in `tests/test_simplex_lu.py`

The C hook is not called by any solver path.  Knob-off production behavior is
therefore byte-identical; the baseline hashes above also confirm deterministic
trajectory/output identity after the build.

Builds used the dossier commands with `UV_OFFLINE=1` added to mechanically
forbid network access.  No network operation, external solver-source read,
per-problem solver tuning, or Git operation was performed.

Validation: `just ci` passed with lint, format, type, Bandit, and dependency
audit clean; **523 tests passed, 7 skipped**, and coverage was **89.16%** against
the 85% floor.  The focused sparse-LU file passed **24/24** before the full gate.
