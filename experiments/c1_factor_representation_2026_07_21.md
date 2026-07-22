# C1 — replace the factorization data structure (2026-07-21)

## Verdict: KILLED

The direction-specific factor representation fails C1's explicit standalone
gate against the current solves, so no in-loop implementation was attempted.
The best alternative took **52.495 us per FTRAN+BTRAN pair**, versus
**47.425 us per equivalent pair** for the current solve slice: **10.69% slower**,
not the required **at least 30% faster**.

The result is deliberately separated from a tempting but wrong comparison.
Four-lane padded row storage was **11.64–47.65% faster than a naive standalone
CSC gather-chase** on the same factors. The production solver is already much
better than that emulation, through sparse startup and its live Forrest-Tomlin
machinery. The mandate's gate says “current solves,” so the naive CSC speedup
does not qualify.

## Falsifier and chosen representation

C1 offered dense storage, register-blocked/ELLPACK storage, and split layouts.
The earlier K3 report records that it did in fact materialize dense
column-major L and live U' arrays and lost by 75–87x. Repeating that proven
fully-dense failure would violate the dossier's instruction not to re-attack a
settled wall. This probe therefore selected the next distinct candidate:

- retain CSC columns for transpose-direction dot solves;
- add CSR rows for normal-direction dot solves, avoiding scattered read-modify-
  writes;
- test a stronger ELLPACK-style variant in which every logical factor row is
  padded to four lanes and accumulated through four independent registers;
- build equivalent padded row views for both normal and transpose directions.

The mechanism is global. It uses no problem-name, dimension, or trajectory
special case. A production version would be knob-gated and would retain the
existing path byte-for-byte when disabled, but C1 forbids that implementation
work after the standalone gate fails.

## Captures and method

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`, reduced to **1,525 x 3,868 x
  23,274**.
- Solver trajectory: native cold start, Dantzig leaving, `expand=1`, `bfrt=0`,
  tolerance `1e-8`.
- Captures: the public diagnostic basis export at iteration limits 1,500,
  2,000, and 3,500. The captured B matrices were reconstructed from the public
  reduced matrix plus unit artificial columns.
- Reference factors: SciPy SuperLU, natural basis-column order, standard
  threshold row pivoting and equilibration. This isolates representation and
  triangular traversal from factorization-algorithm changes.
- Effective factor size (`L.nnz + U.nnz - m`): **22,278**, **23,296**, and
  **27,033**, covering the dossier's roughly 23–30k-nonzero regime.
- RHS set: 16 deterministic six-sparse FTRAN right-hand sides and 16
  deterministic unit BTRAN right-hand sides.
- Timing: 100 repetitions per RHS, nine rotated trials, median reported,
  foreground, pinned to CPU 2. The C probe was compiled with
  `-O3 -march=native`; all representations use the same compilation unit.
- Correctness: both directions compared with SciPy's existing LU triangular
  solves. Worst infinity-norm-scaled error across all methods, captures, and
  RHS vectors was **4.13e-14**.

No network access, solver-source inspection, solver-source modification, Git
operation, or competing solver was used.

## Standalone measurements

Times are microseconds for one FTRAN plus one BTRAN. “Delta” is versus the
naive CSC kernel in this standalone harness, not versus the production solver.

| captured pivot | effective factor nnz | naive CSC | split CSR/CSC | split delta | ELLPACK4 | ELLPACK4 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 22,278 | 100.278 | 88.196 | 12.05% faster | **52.495** | **47.65% faster** |
| 2,000 | 23,296 | 91.964 | 88.509 | 3.76% faster | 57.990 | 36.94% faster |
| 3,500 | 27,033 | 106.626 | 108.797 | 2.04% slower | 94.218 | 11.64% faster |

The plain direction split is not compelling: it ranges from 2.04% slower to
12.05% faster. Padding and four accumulators remove enough loop/control
overhead to beat the naive chase, but the win decays as the factor fills.

Storage also roughly doubles. Baseline CSC storage for the three captures was
315–370 KiB. A retained CSR half adds another 315–370 KiB. The four
direction-specific padded views occupy **2.03x** the baseline CSC bytes.

## Gate against the actual current solves

Seven full current-solver runs were collected with the existing nested solve
timer. Every run returned `optimal` in **4,399 pivots**, objective
`-72,555,248.1298459`, maximum original equality residual **1.77e-7**, and
maximum bound violation **3.86e-12**; all are inside fixed `eps=2e-5`.

The foreground median was **0.6816 s** on this pinned host, with **0.2311 s** in
the current solve slice. There were 5,313 FTRAN and 4,433 BTRAN calls per run.
Normalizing the current slice by its actual call count gives **47.425 us per
equivalent FTRAN+BTRAN pair**.

| comparison | us / FTRAN+BTRAN pair | change vs current |
|---|---:|---:|
| current solver, actual trajectory | **47.425** | — |
| best standalone candidate (ELLPACK4, pivot 1,500) | 52.495 | **10.69% slower** |
| C1 gate | at most 33.197 | **requires 30% faster** |

This comparison is generous to the candidate: it selects the best of three
captured factors, gives the candidate native-code compilation, and charges the
current timing for its dynamic factor/update traversal. It still misses the
30% gate by **40.69 percentage points**. Therefore the conditional “THEN
in-loop” step does not activate.

## Flip arithmetic versus the required -18%

The dossier's canonical wall is about **0.370 s**. A flip requires at least
18% end-to-end, or **0.3034 s**. BTRAN+FTRAN account for 36.8% of wall, so a
factor-only mechanism would need a **48.91% solve-slice reduction** to supply
the entire flip:

```text
required solve reduction = 18.0% / 36.8% = 48.91%
measured candidate        = -10.69% (a regression)
shortfall                 = 59.60 percentage points
```

Applying the measured regression to the canonical wall gives:

```text
projected wall ratio = 1 + 0.368 * 0.106907 = 1.03934
projected wall       = 0.370 * 1.03934       = 0.3846 s
flip target          = 0.3034 s
```

That is a **3.93% end-to-end regression**, leaving the projection **26.75%
above** the flip target. Even the candidate's best-case standalone result is
not part of a viable path.

## Validation and artifacts

- Standalone C compiled cleanly with `-Wall -Wextra -Werror`.
- Repository Ruff lint and format check: passed.
- Repository `ty check`: passed.
- Focused LU, dual-simplex, and dual-cleanup tests: **60 passed**.
- Public current-solver certificate gate: passed on all seven runs.
- In-loop implementation: not attempted, as required by the failed 30% gate.

The network-capable audit portion of `just ci` was not run because this
campaign forbids all network access; the checks above were run with
`UV_OFFLINE=1`.

Artifacts:

- Driver: `experiments/c1_factor_probe.py`
- Standalone kernel: `experiments/c1_factor_microbench.c`
- Structured results: `/tmp/c1-factor-probe/results.json`
- Captured stdout: `/tmp/c1-factor-probe/stdout.json`

**Final verdict: KILLED.** ELLPACK4 beats a simplistic CSC implementation but
does not beat the current factor solves; it is 10.69% slower before paying any
in-loop maintenance cost.
