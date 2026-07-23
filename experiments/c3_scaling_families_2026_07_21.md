# C3 — trajectory shaping by scaling family (2026-07-21)

## Verdict: KILLED

The predeclared kill criterion was: kill C3 if no global scaling family cuts
greenbea dual-simplex wall by at least 10% while keeping the DS sentinels and
the fixed `eps=2e-5` certificate/oracle gates clean.

No family reduced greenbea at all.  Relative to the five-run Ruiz median:

- power-of-two-rounded Ruiz was the least-slow alternative at **+31.43% wall**
  and **+10.14% pivots**, but terminated `dual_infeasible` rather than with a
  certificate;
- the only certificate-clean alternative on greenbea, combined
  geometric/Ruiz, was **+79.71% wall** and **+83.50% pivots**;
- Curtis-Reid took 14,474 pivots and terminated `dual_infeasible`;
- geometric mean reached the 50,000-pivot limit with a failed primal point.

This is the opposite of the required 10% win.  The scaling-algorithm axis is
therefore killed for greenbea under these global families.

## Falsifier and global families

The experiment used one process-global diagnostic knob,
`LINPROGX_DS_SCALING`, with no instance names, dimensions, or fixture-specific
thresholds:

| value | deterministic global rule |
|---|---|
| unset / `ruiz` | Historical 10 simultaneous inf-norm Ruiz passes plus one L2 pass. |
| `curtis-reid` | 20 fixed row/column coordinate sweeps of the Curtis-Reid log least-squares equations. |
| `geometric` | 10 simultaneous min/max geometric-mean equilibration passes. |
| `pow2` | Historical Ruiz plus L2, then round every cumulative row and column scale to its nearest power of two. |
| `combined` | Five geometric-mean passes, five Ruiz refinement passes, then one L2 pass. |

Every family retained the historical global activation guard (raw nonzero row
inf-norm ratio at least 100) and `[1e-8, 1e8]` scale clamp.  The unset path
kept the historical Ruiz arithmetic.  After measurement, the killed
experimental implementations were removed; only this evidence report is the
deliverable.

## Method

- Fixtures: `/tmp/lpsuite/lp_{greenbea,woodw,stocfor3,80bau3b,cre_d}.mat`.
- Solver: normal linprogx presolve followed by native equality-box dual
  simplex, Dantzig leaving, `expand=1`, `bfrt=0`, solver tolerance `1e-8`.
- Certificate: returned status `optimal`, original-space equality and bound
  residuals at most the fixed `eps=2e-5`, and original objective relative
  agreement with the SciPy/HiGHS oracle at most `2e-5`.
- Timing: direct presolved DS wall, foreground-only, pinned with
  `taskset -c 4`; one warmup per arm; five timed greenbea runs with rotating
  arm order; one timed run per sentinel after warmup.
- Raw result: `/tmp/c3_scaling_2026_07_21.json`.
- Build was performed offline with `UV_OFFLINE=1` added to both prescribed
  commands.  No network access, Git operation, or external solver source
  inspection occurred.

The machine was timing-noisy during this campaign (Ruiz greenbea range
0.553-0.690 s), so the verdict uses medians and is also independently pinned
by deterministic pivot counts and status/certificate failures.  The margins
are much larger than the noise.

## Greenbea measurements

| family | status | pivots | median wall | wall vs Ruiz | equality residual | bound violation | objective relative delta | certificate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Ruiz | optimal | 4,399 | 0.647140 s | baseline | 1.77e-7 | 3.86e-12 | 4.11e-16 | PASS |
| Curtis-Reid | dual_infeasible | 14,474 | 4.578710 s | **+607.53%** | 4.44e-8 | 8.60e-13 | 2.05e-16 | **FAIL** |
| geometric | iteration_limit | 50,000 | 13.634507 s | **+2,006.89%** | 1.61e10 | 3.23e3 | 8.66e-3 | **FAIL** |
| pow2 | dual_infeasible | 4,845 | 0.850533 s | **+31.43%** | 1.46e-8 | 1.03e-13 | 1.04e-7 | **FAIL** |
| combined | optimal | 8,072 | 1.162965 s | **+79.71%** | 6.45e-7 | 2.28e-11 | 6.18e-14 | PASS |

Pivot count and terminal result were identical across all five timed runs of
each family.  The two `dual_infeasible` rows happen to have primal points and
objectives close to the oracle, but the dossier requires certificate-backed
optimality; they are failures, not usable solutions.

## DS sentinel measurements

All sentinel rows below returned certificate-backed `optimal` and matched the
original-space oracle.  The worst equality residual across the table was
`7.73e-12`; the worst bound violation was `6.91e-13`, both far inside
`eps=2e-5`.  Entries are `pivots / wall seconds` from the single timed run.

| fixture | Ruiz | Curtis-Reid | geometric | pow2 | combined |
|---|---:|---:|---:|---:|---:|
| woodw | 1,338 / 0.1333 | 10,313 / 5.3722 | 7,376 / 3.1865 | **1,159 / 0.1158** | 1,379 / 0.1736 |
| stocfor3 | 9,604 / 2.1258 | 17,219 / 6.1557 | 13,823 / 4.9660 | 9,950 / 2.4726 | 11,461 / 3.1468 |
| 80bau3b | 6,758 / 0.5891 | 6,790 / 0.6237 | 8,018 / 0.8063 | **6,539 / 0.5481** | 7,419 / 0.6776 |
| cre_d | 46,048 / 23.5321 | 46,048 / 27.7834 | 46,048 / 28.1061 | 46,048 / 23.6336 | 46,048 / 23.6361 |

Power-of-two rounding did shape two sentinels favorably: woodw fell 13.4% in
pivots and 13.1% in wall, while 80bau3b fell 3.2% in pivots and 7.0% in wall.
That is real evidence that scaling changes DS trajectories, but it cannot save
the mechanism: power-of-two rounding increases greenbea pivots and fails its
certificate.  Curtis-Reid and geometric mean severely regress woodw and
stocfor3.  Every family took the same 46,048-pivot cre_d trajectory; its
single-run wall differences are not treated as meaningful.

## Knob-off identity and verification

Before the C experiment, a canonical artifact captured status, pivot count,
IEEE-754 objective/residual values, reduced-`x`, terminal basis and bound-status
hashes, and committed-pivot hash for all five fixtures.  The final experimental
build with the scaling knob unset reproduced it exactly:

```text
pre-change SHA-256:  7315ba3b7f47af71f7045fab1f50cbccb3dc9c11c005ceb10413e2adb6b98071
post-change SHA-256: 7315ba3b7f47af71f7045fab1f50cbccb3dc9c11c005ceb10413e2adb6b98071
```

Focused experimental-build verification: `tests/test_dual_simplex.py` passed
29/29, including global-family oracle checks, explicit/unset Ruiz identity,
and invalid-knob rejection.  Ruff lint passed for the probe and test changes.

After removing the killed experimental path and rebuilding, final restored-state
validation passed: Ruff lint and format check, `ty check`, Bandit at medium
severity, and the coverage suite (**522 passed, 7 skipped, 89.16% coverage**
against the 85% floor).  `pip-audit` was intentionally not invoked because its
remote advisory lookup conflicts with the audited no-network rule.

## Flip arithmetic

The dossier says greenbea is at a hosted ratio of 1.215 and needs about an 18%
end-to-end reduction.  A qualifying local candidate must therefore have a
wall ratio at most `0.82` versus Ruiz.

- Least-slow active family, pow2: `0.850533 / 0.647140 = 1.3143`, already
  **60.3% above** the required `0.82` ratio, before disqualification for its
  missing certificate.  Applied mechanically to the hosted ratio, it projects
  `1.215 * 1.3143 = 1.597`, a larger loss.
- Best certificate-clean alternative, combined:
  `1.162965 / 0.647140 = 1.7971`, **119.2% above** the required ratio.  Its
  corresponding hosted projection is `1.215 * 1.7971 = 2.184`.

The mechanism supplies no portion of the needed -18%; it moves in the wrong
direction by at least 31% and, among certificate-clean alternatives, 80%.

**Final verdict: KILLED.**
