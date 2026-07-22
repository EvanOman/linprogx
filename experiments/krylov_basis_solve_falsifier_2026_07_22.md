# Certificate-corrected Krylov basis-solve falsifier — 2026-07-22

## Verdict

**KILLED in the tested scope.** Frozen-last-refactor Krylov and the fixed
matched-Jacobi residual-correction family do not fund `greenbea` under the
current exact-Dantzig / Harris trajectory.

The cost/authority frontier has no surviving point:

- degrees **1, 2, 3, 4, 8, 16, and 32** of matched-Jacobi true-residual
  correction all preserve the native entering column in **0/404** Harris
  tests. The BTRAN residual stalls by degree two at **0.742–0.998** and never
  recovers; every sampled degree therefore requires exact fallback;
- direct full-call `B`+`B^T` residual-pair medians are **25.056–30.695 us**
  versus the **20.028 us** probe ceiling. These are implementation
  observations, not a machine-independent lower bound. Empty-call subtraction
  varied too much between reruns and is explicitly not used for the verdict;
- ordinary frozen-last-refactor LU-preconditioned Krylov is more expensive
  still: every correction degree adds a stale triangular solve to the same
  `B` or `B^T` matvec and reduction work. The current Forrest-Tomlin path is
  already the exact Sherman-Morrison/eta-style recurrence; relabeling those
  exact updates as Krylov is not a new mechanism.

Because every tested correction degree falls back, its fully charged cost is
the current exact solve plus positive scout work: a strict regression without
depending on a microbenchmark subtraction. No production C behavior was
changed, and no diagnostic C was needed.

## Question and fixed gate

The tested claim was that a frozen-refactor or factor-free iterative solve
could replace the current exact true-FT FTRAN and BTRAN while retaining the
fixed `eps=2e-5` certificate contract and exact Dantzig/Harris decisions.
Ambiguous decisions must fall back to the exact solve.

The current comparable solve pair is 47.425 us and the audited solve share is
34.62% of wall. Therefore:

| target | required pool reduction | allowed solve pair |
|---|---:|---:|
| 20% whole-wall probe gate | 57.77% | **20.028 us** |
| 17.7013% board gap | 51.13% | **23.176 us** |

These are favorable gates: the proposed path receives credit for replacing
the entire solve pair, while all work outside it remains unchanged.

## Authoritative state and RHS capture

The diagnostic uses the native linprogx solver itself at fixed prefix limits
512, 1536, 3072, and 4096, then repeats each prefix at `k+1`. Every run returns
exactly the requested iteration count. Each adjacent basis pair differs in
one position, which identifies the actual next pivot without adding a trace:

- BTRAN RHS: `e_leaving` from the changed basis position;
- FTRAN RHS: the scaled structural/artificial column newly present at `k+1`.

The probe reproduces the native 10-pass infinity-norm plus one l2 Ruiz scaling
and constructs each actual 1,525-row basis. The samples span the early,
middle, and late trajectory:

| checkpoint | basis nnz | leaving position | entering column | FTRAN RHS nnz |
|---:|---:|---:|---:|---:|
| 512 | 7,022 | 714 | 1,609 | 15 |
| 1,536 | 7,138 | 1,426 | 514 | 6 |
| 3,072 | 7,116 | 487 | 792 | 16 |
| 4,096 | 7,522 | 1,335 | 3,693 | 4 |

This is not a synthetic matrix or arbitrary right-hand-side microbenchmark.

## Residual-pair timing: contextual, not dispositive

On CPU 4 with one OpenBLAS thread, each actual basis was converted to CSR and
both `B*x` and `B^T*y` were executed in one alternating pair. There were 100
warmups followed by 11 batches of 5,000 pairs.

| checkpoint | full pair median | observed empty-call subtraction |
|---:|---:|---:|
| 512 | 25.056 us | 12.011 us |
| 1,536 | 27.272 us | 13.995 us |
| 3,072 | 28.042 us | 14.588 us |
| 4,096 | 30.695 us | 15.944 us |

The full direct pair exceeds the 20.028 us gate at all four checkpoints, but
it includes Python/SciPy call overhead. Empty-call subtraction is not stable:
the observed net changed materially on independent reruns. It is retained in
the artifact only as narrative timing context. It is **not** a lower bound,
and no degree-two or higher kill is inferred from it.

## Degrees 1 through 32: all unauthoritative

At each basis, maximum bipartite matching first permutes columns onto a
structurally nonzero diagonal. Starting from `x_0=0`, each degree applies the
matched Jacobi inverse to the **current true residual**, then receives the
exact residual-minimizing scalar for free:

`z_k = P D^-1 r_k`, `w_k = B z_k`,
`alpha_k = (w_k^T r_k)/(w_k^T w_k)`,
`x_{k+1}=x_k+alpha_k z_k`, `r_{k+1}=r_k-alpha_k w_k`.

The BTRAN recurrence uses `B^T` and the corresponding permuted residual. This
is stronger than a fixed polynomial coefficient because every degree receives
an exact line search against the true residual.

| checkpoint | degree-1 F/B residual | degree-2 F/B residual | degree-32 F/B residual |
|---:|---:|---:|---:|
| 512 | 0.977682 / 0.742233 | 0.977662 / 0.742233 | 0.977662 / 0.742233 |
| 1,536 | 0.506091 / 0.997509 | 0.344460 / 0.997509 | 0.336399 / 0.997509 |
| 3,072 | 0.684672 / 0.867244 | 0.681566 / 0.866353 | 0.681554 / 0.866353 |
| 4,096 | 0.697791 / 0.917363 | 0.654979 / 0.917206 | 0.569653 / 0.917206 |

These errors are not borderline certificate tolerances. To test the actual
pivot authority rather than relying only on norms, the exact and approximate
BTRAN rows were fed into the current admissibility and Harris two-pass choice.
The EXPAND tolerance was swept across 101 values from zero through its entire
allowed `[0, 1e-8]` range:

- the exact rows choose the native `k+1` entering column in **404/404** cases;
- the approximate rows choose it in **0/404** cases at **every** recorded
  degree: 1, 2, 3, 4, 8, 16, and 32.

The BTRAN correction is already numerically stagnant by degree two. Thus every
sampled result at every tested degree is ambiguous/wrong and requires exact
fallback. Paying any scout degree and then the current exact solve is a strict
regression.

## Candidate-class disposition

### A. Frozen-last-refactor LU-preconditioned Krylov

Krylov with the last exact refactor as preconditioner is structurally
dominated by the current true-FT solve. Each iteration pays the frozen
triangular solve already at the core of FT, then additionally pays a `B` or
`B^T` application, reductions, and orthogonalization. FT instead applies its
stored exact update corrections and returns the exact solve. Refresh and
fallback only add work. This is the assigned class-A arithmetic kill; no
native implementation was justified.

### B. Factor-free, polynomial, and recycled correction

A fixed matched-diagonal polynomial is directly falsified through degree 32,
not merely at degree one. Establishing decision authority requires the true
residual or an equivalently valid global error bound; without it the method
violates the exact-decision/certificate constraint. With it, all tested
degrees fall back on all four actual samples.

This experiment does not claim that every possible learned sparse approximate
inverse or arbitrarily rich recycled subspace is equivalent to matched Jacobi.
Such a mechanism would need a separate globally fixed construction, refresh
charge, application timing, and authority proof. No such concrete funded
candidate emerged here.

This kill is scoped to ordinary frozen-refactor LU-preconditioned Krylov and
the tested fixed matched-Jacobi true-residual correction through degree 32. It
does not claim that all imaginable approximate inverses, recycled spaces, or
basis factorization data structures are impossible.

## Artifact and reproduction

- Probe: `experiments/krylov_basis_solve_probe.py`
- Raw artifact: `/tmp/krylov-basis-solve-falsifier/results.json`
- Artifact mode: `0600`; parent directory mode: `0700`
- Artifact SHA-256:
  `b0721bbcc31c9904ba3e05b8010ff6bea00207c55978b7364faaa4a632f50136`

```bash
cd /home/evan/dev/linprogx-krylov
taskset -c 4 env \
  OPENBLAS_NUM_THREADS=1 \
  UV_OFFLINE=1 \
  UV_CACHE_DIR=/tmp/uv-cache \
  LINPROGX_DS_EXPORT_BASIS=1 \
  uv run python -m experiments.krylov_basis_solve_probe

sha256sum /tmp/krylov-basis-solve-falsifier/results.json
stat -c '%a %s %n' \
  /tmp/krylov-basis-solve-falsifier \
  /tmp/krylov-basis-solve-falsifier/results.json
```

## Validation and integrity

- Native prefix gates: **8/8** exact requested iteration counts.
- Adjacent basis-difference gates: **4/4** exactly one changed position.
- Exact decision replay: **404/404** matches.
- Approximate decision replay at each of degrees 1, 2, 3, 4, 8, 16, and 32:
  **0/404** matches.
- No production solver source or test was edited. The only repository writes
  are this report and its diagnostic Python probe.
- No Git command was run.
- No network tool, API, or package download was used. All `uv` invocations
  used `UV_OFFLINE=1` and the existing local cache.
- No external solver source was read. The experiment uses linprogx native
  prefixes and SciPy only as local linear-algebra/measurement machinery.

**Final verdict: KILLED in scope.** The matched-Jacobi correction remains
remotely unauthoritative through degree 32 and therefore pays exact fallback
on every actual sample. Frozen-LU Krylov adds matrix/reduction work around the
same triangular core that FT already corrects exactly.
