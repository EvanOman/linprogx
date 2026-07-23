# Simultaneous rank-k exchange: S3 exact flip-aware face oracle

Date: 2026-07-22
Verdict: **S3_CLOSE_FIXED_POLICY_RANKK_FACE_EXCHANGE.**

S3 does not justify an in-process rank-k implementation. The exact flip-aware
oracle found eligible true boxes but made zero flips at every fixture endpoint,
so it reproduced the S2b no-flip candidates. Every public bootstrap changed the
intended artificial-bound/nonbasic state, every honest pivot total exceeded
3,519, and every fully charged path was far above `0.80x` cold.

## Fixed authority and refinement

The prior fresh-warm authority remains passed at checkpoints
`{512,1536,3072,4096}`: exact basis import, allowlisted status normalization,
zero singular repairs/fallbacks, repeated iteration-0 snapshots, exact k=1
replay, and deterministic fixed-epsilon no-exchange continuations.

S2b reconstructs the scaled extended matrix, fresh-factors `B`, and preserves
the original cost while refining `B^T y=c_B`. Residuals and all critical-column
reduced costs are accumulated in platform long double. The fixed global cap is
four corrections and the stop is scale-derived. Refinement did not remove all
exact wrong-side signs:

| checkpoint | corrections | wrong-side count | maximum |
|---:|---:|---:|---:|
| 512 | 1 | 255 | `5.2076e-14` |
| 1,536 | 1 | 208 | `3.0601e-14` |
| 3,072 | 4 | 217 | `8.5221e-13` |
| 4,096 | 4 | 261 | `6.8412e-13` |

All original fresh-factor, backward-error, native top-1, and recorded-alpha
gates still passed. Statuses were never relabeled.

## Explicit candidate-generation perturbation

Every wrong-side value was below the predeclared global reduced-cost error
bound. S2b therefore formed only the authorized candidate-generation cost:

`delta_B=0`; wrong-side `LO`, `HI`, and every nonzero `FREE` receive
`delta_j=-r_j`; `FIXED` receives zero.

The repaired shifted reduced costs are exactly zero on repaired columns. This
is explicitly a bounded candidate-generation heuristic, not original-cost
dual feasibility, monotonicity, or a certificate.

| checkpoint | repaired | `||delta||_1` | `||delta||_inf` | old-point objective perturbation |
|---:|---:|---:|---:|---:|
| 512 | 255 | `1.6121e-12` | `5.2076e-14` | `-2.2864e-5` |
| 1,536 | 208 | `1.0554e-12` | `3.0601e-14` | `0` |
| 3,072 | 217 | `1.3802e-11` | `8.5221e-13` | `0` |
| 4,096 | 261 | `1.2545e-11` | `6.8412e-13` | `0` |

## No-flip face census

For each `k={2,3,4}`, `P` is the deterministic top-k primal-infeasible set.
The probe builds `beta`, `sigma`, and `H`, then maximizes `v^T lambda` subject
to the old-nonbasic LO/HI/FREE halfspaces under the candidate cost. A
primal-infeasible old basis is not mislabeled optimal; positive face progress
is permitted.

| checkpoint | k=2 | k=3 | k=4 |
|---:|---|---|---|
| 512 | progress `3.6601e10`, active rank 1 | same, rank 1 | same, rank 1 |
| 1,536 | zero progress, rank 2 | zero progress, rank 3 | zero progress, rank 4 |
| 3,072 | progress `1576.60`, rank 2 | progress, rank 2 | progress, rank 3 |
| 4,096 | progress `3964.65`, rank 2 | progress `4305.97`, rank 3 | progress `4305.97`, rank 4 |

Rank-deficient and zero-progress cases are incomplete searches, not theorem or
class kills. Deterministic Q generation used lexicographic rank-increasing,
pivoted-QR, and conditioned-start policies, deduplicated and globally capped at
eight. Candidate counts were `0/0/0`, `0/0/0`, `1/0/0`, and `1/1/2`.

For every Q, S2b independently checked `Delta_Q=K^-1 v`, where
`K=-diag(sigma)(B^-1 A_Q)_P`, and compared the old-dictionary primal
reconstruction with a fresh `B'` solve. It then re-solved the face vertex and
fresh-factored `B'`. Shifted sign residues within the same global error bound
were retained only as speculative candidates; no shifted dual-feasibility or
monotonicity claim is made. The two k=4 candidates failed the unchanged face or
fresh-basis gates.

## S3 exact flip-aware oracle

Flip eligibility mirrors native BFRT: an old nonbasic must be structural, have
finite distinct true lower and upper bounds, and have no artificial bound.
Logical, fixed, one-sided, and artificially boxed variables never flip.
Eligible counts were 105, 170, 206, and 209 by checkpoint.

For each eligible box S3 removes its old halfspace and introduces free
`zeta_j`, constrained below both `lo_j r'_j` and `hi_j r'_j`. It maximizes the
exact constant-adjusted piecewise shifted-dual progress. Positive `r'` selects
LO, negative `r'` selects HI, and numerical zero preserves prior status. A
focused tiny oracle crosses a boxed kink and verifies the LO-to-HI direction
and exact objective identity.

All twelve fixture endpoints made **zero flips**, so progress and active ranks
match the no-flip census. Flip-aware Q counts were `0/0/0`, `0/0/0`, `1/0/0`,
and `1/1/2`; the same three speculative k=2/3 candidates survived. The
independent primal check uses `delta_R=z'_R-z_R` and the explicitly signed
dictionary identity `K Delta_Q = v - H_R delta_R`, then compares the complete
old-dictionary reconstruction against a fresh `B'` factorization.

## Original-cost restoration and public warm result

Three speculative bases survived: checkpoint 3,072 k=2 and checkpoint 4,096
k=2/3. Original `c` was restored before any public solve. Fresh original-cost
wrong-side maxima were `6.0796e-13`, `5.9530e-13`, and `7.0433e-13`, or
`0.713x`, `0.870x`, and `1.030x` their source `||delta||_inf`; all remained
inside the unchanged global bound.

| checkpoint / k | no-exchange pivots | B' pivots | endpoint certificate | public bootstrap |
|---|---:|---:|---|---|
| 3,072 / 2 | 1,328 | 1,331 | deterministic optimal | mismatch |
| 4,096 / 2 | 302 | 300 | deterministic optimal | mismatch |
| 4,096 / 3 | 302 | 299 | deterministic optimal | mismatch |

Every B' bootstrap imported the exact basis with only allowlisted status
normalizations, used the warm start, had zero repairs/fallback, and both
untraced continuations agreed exactly with the cold objective/residual
fixed-epsilon certificate. However, fresh public initialization changed
`has_art_bound` and `hi_ext`, and the reconstructed candidate nonbasic/primal
state no longer matched the intended B' state. These are explicitly classified
public-warm mismatches and disqualify all three.
The 3,072 candidate also costs three additional pivots, so it cannot fund the
gate even apart from the mismatch.

The independent pivot-funding ceiling is `floor(0.8 * 4399) = 3519`, counting
one rank-k exchange honestly:

| checkpoint / k | prefix + continuation + exchange | funded |
|---|---:|---|
| 3,072 / 2 | 4,404 | no |
| 4,096 / 2 | 4,397 | no |
| 4,096 / 3 | 4,396 | no |

No flip-aware candidate clears either the exact public endpoint gates or the
3,519 pivot bar. This closes the tested no-flip plus exact flip-aware
**fixed-policy rank-k face exchange**. It does not claim every imaginable
rank-k algorithm is impossible.

## Fully charged timing

Cold median was `0.549330 s` at 4,399 pivots. Charge includes prefix,
refinement, repair, every face LP and rejected-Q verification, original-cost
rebuild, and median untraced continuation. Trace bootstrap phases and repeated
timing runs are excluded as inadmissible.

| checkpoint | charged seconds | charged / cold |
|---:|---:|---:|
| 512 | `2.3469` | `4.272x` |
| 1,536 | `2.3197` | `4.223x` |
| 3,072 | `2.8907` | `5.262x` |
| 4,096 | `4.5660` | `8.312x` |

No path approaches the required `<=0.80x` pass threshold. There is no
performance claim, production loop, or in-process implementation.

## Evidence and verification

Durable JSON:
`/tmp/linprogx-rankk-kxmy5mxz/rankk_exchange_falsifier_2026_07_22.json`
(private `0700` directory, `0600` file).

The focused tests cover mixed-precision refinement, bounded repair including
FREE columns, ineligible-repair refusal, deterministic capped Q selection,
speculative shifted-residue labeling, and a boxed-kink flip oracle. Offline
validation ran the full dual-simplex file plus lint and format checks. S3 made
no C edit. The final offline result was `33/33` tests, with lint and format
checks passing. No Git operation, network access, or external solver source
was used.
