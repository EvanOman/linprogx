# Global algebraic-reformulation falsifier — greenbea (2026-07-22)

## Verdict: KILLED in scope

No concrete, globally applicable algebraic reformulation found in this census
earns an implementation falsifier against the **20% whole-wall campaign gate**.
The strongest genuinely distinct construction was a fixed nonsingular left
row transformation, particularly normalization by an actual trajectory basis.
It preserves the exact LP and simplex tableau over the reals, but the natural
`U = B_512^-1` construction makes the coefficient matrix **3.906x** denser and
makes later sampled LU factors **78.48%, 134.26%, and 156.81% larger** than the
corresponding untransformed factors. Its one-checkpoint identity-basis benefit
does not survive the trajectory.

This is a funding kill, not an impossibility theorem. It closes the concrete
natural reformulations below. It does **not** prove that no arbitrary sparse
nonsingular `U`, novel extended formulation, or future decomposition can ever
work. Reopening requires a specified global construction with complete
transformed-matrix charges, original-space certificate reconstruction, and a
four-checkpoint or full-trace funding proof.

No production code changed, so this report does not warrant v3
recertification. The certified board remains **23W-0P-1L**; `greenbea` remains
the sole loss at **1.2150867**.

## Funding invariant

The board ratio requires the following whole-wall reduction:

```text
g_board = 1 - 1 / 1.2150867 = 0.177013459 = 17.7013459%
g_gate  = 0.20                         = 20.0000000%
```

The current comparable FTRAN+BTRAN pair is `47.425 us`, and the measured solve
share is `0.3462`. If a representation attacks only those solves, its allowable
pair cost is

```text
C_pair(g) = 47.425 * (1 - g / 0.3462).

C_pair(g_board) = 23.1764 us
C_pair(g_gate)  = 20.0275 us
```

Therefore merely flipping the board requires a **51.1304%** solve-pair cut,
and the campaign gate requires a **57.7701%** cut. Any extra time for building
or applying the transformation, pricing against a denser transformed matrix,
postsolve, or certificate reconstruction makes these ceilings stricter. The
source values and the same funding formula are in
`experiments/krylov_basis_solve_probe.py`.

## Strongest distinct family: fixed nonsingular left transformation

Consider the equality-and-box LP

```text
minimize    c^T x
subject to  A x = b
            l <= x <= u,
```

and a fixed nonsingular matrix `U`. Replace the equalities by

```text
A' = U A,
b' = U b.
```

The feasible set and objective are unchanged. For any basis `B`, define
`B' = U B`. The exact simplex quantities are invariant:

```text
x'_B = (U B)^-1 (U b)       = B^-1 b                 = x_B,
d'_j = (U B)^-1 (U a_j)     = B^-1 a_j               = d_j,
y'   = (U B)^-T c_B,
y    = U^T y'               = B^-T c_B,
r'_j = c_j - (U a_j)^T y'   = c_j - a_j^T y          = r_j.
```

Thus basic values, FTRAN directions, ratios, reduced costs, and the exact
deterministic pivot trajectory are unchanged. A transformed dual certificate
maps back through

```text
y = U^T y'.
```

The primal vector `x` requires no reconstruction. Original-space feasibility,
bounds, objective agreement, and dual signs must still be checked at the fixed
`eps = 2e-5` gate because floating-point accumulation order changes.

During an artificial-basis phase, trajectory invariance requires transforming
the complete algorithmic extended matrix: `[A, I]` becomes `[U A, U I]`.
Introducing a fresh identity artificial block would still describe the same
original LP, but it would no longer preserve the reference artificial-basis
trajectory. The proxy below consistently transforms the captured extended
bases.

### Why this family was not already closed

- C3 scaling uses diagonal row/column transformations. A general `U` is
  non-diagonal.
- Locality and ordering probes use monomial transformations or factor
  permutations. They do not change the coefficient representation.
- Equality aggregation and other presolve substitutions are noninvertible
  dimension reductions. Here every equality remains and the LP is exactly
  equivalent without postsolving `x`.
- P-F reordered factors of the same `B`. Here the factorized matrix is `U B`,
  even though the exact tableau is invariant.

This made fixed left transformation the strongest genuinely untested
reformulation family in the census.

## Read-only `B_512^-1` funding proxy

The proxy used the current presolved and Ruiz-scaled `greenbea` matrix and the
native bases exported at iterations 512, 1536, 3072, and 4096. The reduced
structural matrix is

```text
m = 1,525
n = 3,868
nnz(A) = 23,274
```

At checkpoint 512, the extended basis has 1,462 structural and 63 artificial
columns, `7,022` nonzeros, and SuperLU factor fill `nnz(L+U) = 9,536`. The
natural normalization

```text
U = B_512^-1
```

makes the reference transformed basis the identity. Its factor fill is `3,050`
including the unit diagonals, a **68.02%** reduction at that single point.

The benefit does not persist:

| actual trajectory basis | original basis nnz | original `nnz(L+U)` | transformed basis nnz | transformed `nnz(L+U)` | transformed fill change |
|---:|---:|---:|---:|---:|---:|
| 512 | 7,022 | 9,536 | 1,525 | 3,050 | -68.02% |
| 1,536 | 7,138 | 10,507 | 15,825 | 18,753 | **+78.48%** |
| 3,072 | 7,116 | 11,183 | 21,636 | 26,197 | **+134.26%** |
| 4,096 | 7,522 | 12,860 | 26,524 | 33,026 | **+156.81%** |

The transformed structural matrix is already much wider in storage:

```text
nnz(|B_512^-1 A| > 1e-8)  = 90,916
nnz(|B_512^-1 A| > 1e-10) = 90,920
nnz(|B_512^-1 A| > 1e-12) = 90,920

90,916 / 23,274 = 3.90633x
```

The conclusion is insensitive to the `1e-10` versus `1e-12` numerical-zero
threshold. The later transformed fill moves in the wrong direction by much
more than the required 57.7701% solve reduction, before charging the **3.906x**
transformed structural matrix to pivot-row pricing, reduced-cost work, or
updates. The concrete basis-normalization construction is therefore killed
without production implementation.

Its main failure modes are:

1. `U` is adapted to one basis while the trajectory moves away from it.
2. Sparsity removed from the reference basis migrates into `U A` and later
   bases.
3. Transformed pricing must still produce the invariant, often-dense tableau
   row; algebraic equivalence does not remove that information.
4. Floating-point factorization and `y = U^T y'` introduce additional
   numerical paths that require original-space certification.
5. Constructing, storing, and applying `U` adds costs not credited above.

### Minimum reopening falsifier for a different `U`

A future left-transform proposal must specify a deterministic global
construction, not merely “precondition the rows.” Before solver integration it
must, on the same four authoritative bases:

1. show nonsingularity and the exact certificate mapping `y = U^T y'`;
2. charge construction, `U A` storage, transformed pricing, and update work;
3. reproduce original-space primal and dual residuals at `eps = 2e-5`;
4. demonstrate a measured or rigorously weighted solve-pair cost no greater
   than `20.0275 us` throughout the samples, or a complete whole-wall bound
   below `0.80` of control; and
5. remain a global policy on the full suite, with no fixture name or
   greenbea-specific threshold.

## Coverage map of other reformulation families

### 1. Noninvertible equality elimination and aggregation

This is prior measured territory, not an open row-transformation family.
General equality aggregation can reduce greenbea toward the 951-row shape, but
linprogx on that reduction takes **5,222 pivots**, versus **4,399** on its own
1,525-row reduction. At the fill frontier, the aggregation implementation's
best pivot result was only about 7%, while the target shape regressed pivots by
24%. Bounded-singleton/ranged-row elimination reaches a propagation fixpoint.
Shape parity does not transfer pivot behavior.

Evidence: `experiments/greenbea_presolve_diff_2026_07_17.md`,
`experiments/greenbea_pivot_gap_2026_07_17.md`, and the aggregation entries in
`docs/HANDOFF.md`.

### 2. Invertible right/column transformations

Let `x = T z + s`. To retain independent box bounds for a general equality-box
LP, an invertible map must preserve the product of one-dimensional intervals
and rays. With no free reduced variables on greenbea, that limits a globally
reversible box-preserving map to axis permutation and diagonal scaling, the
already-tested monomial families.

A genuine shear or rotation is exact only if the original bounds become
coupled inequalities in `z`. Greenbea has

```text
finite lower bounds = 3,868
finite upper bounds =   257
coupled inequalities = 4,125
```

Representing those constraints explicitly moves the proposal into the
extended/nullspace families below; it is not a free trajectory-shaping column
transformation.

### 3. Nullspace representation

For a nominal full-row-rank `A`, write

```text
x = x_0 + N z,
A N = 0,
dim(z) = n - m = 2,343.
```

The equalities disappear, but the 4,125 finite original bounds become 4,125
coupled inequalities on `N z`. A conventional equality-and-slack realization
therefore has a basis dimension of 4,125 instead of 1,525, a **2.705x** row
increase, before charging a generally filled nullspace basis. The natural
basis-inverse construction underlying such an `N` is exactly the density
mechanism exposed by the `B_512^-1 A` proxy. No sparse-nullspace construction
with a 20% funding invariant was found.

### 4. Explicit dual or range-space representation

Dualizing the reduced equality-box LP produces 3,868 stationarity equations,
one for each original column. It requires 1,525 free equality multipliers,
3,868 lower-bound multipliers, and 257 upper-bound multipliers: 5,650 variables
if free variables are represented directly, or 7,175 nonnegative variables if
the free multipliers are split. The factor dimension grows from 1,525 to 3,868,
or **2.536x**.

Operationally this is the primal/dual role-swap direction. Its strongest
public sparse revised-primal trajectory was already **7,427 pivots** versus
the current dual path's **4,399**, with current mandatory exact-pivot kernels
projecting **0.643792 s**. It does not fund a new encoding.

Evidence: `experiments/sparse_primal_falsifier_2026_07_22.md`.

### 5. Column-copy / consensus extended formulations

An exact degree-reduction lift can replace each selected degree-`d` column by
incidence-local copies and `d-1` equality links. This reduces column degree but
adds `d-1` variables and `d-1` basis rows. The measured greenbea tradeoff is:

| columns split | original nnz touched | consensus rows added | new row count | row-count factor |
|---|---:|---:|---:|---:|
| degree > 2 | 22,078 (94.861%) | 18,977 | 20,502 | 13.444x |
| degree > 6 | 9,320 (40.045%) | 8,587 | 10,112 | 6.631x |
| degree > 16 | 1,243 (5.341%) | 1,176 | 2,701 | 1.771x |

No threshold both attacks enough of the matrix to support a 20% win and keeps
the factor dimension near its current size. The exact lift is globally valid
and certificates can be summed through the consensus equations, but its
dimension arithmetic kills the natural family before implementation.

### 6. Dantzig-Wolfe, Benders, and block decomposition

The reduced matrix's bipartite graph has exactly **one connected component**
containing all 1,525 rows and 3,868 columns. More decisively, actual trajectory
bases have one dominant component containing **88.7%, 93.8%, and 95.7%** of
rows at the measured checkpoints. Fragmenting the core into balanced sub-10%
blocks requires a **10-15% row-and-column border**. An exact decomposition
would therefore introduce a substantial linking master plus repeated
subproblem solves rather than expose independent blocks.

Evidence: `experiments/probe_schur_2026_07_18.md` plus the read-only full-matrix
connected-component census.

### 7. Network reformulation

Only a small part of the matrix has native network-column degree. Exactly
3,101 of 3,868 columns have degree greater than two, and those columns contain
22,078 of 23,274 nonzeros (**94.861%**). Converting that majority into a
degree-two network via consensus copies is the first row of the lift table:
18,977 additional rows and a **13.444x** row count. Splitting only high-degree
outliers does not expose enough work: degree greater than 16 touches 5.341% of
nonzeros while increasing rows 77.1%.

The clean-room network aggregation that helped `pds` is not transferable to
this structure; it contracted an existing network family rather than creating
one with a large exact lift.

### 8. Homogeneous, self-dual, slack, and epigraph lifts

These formulations can be globally exact and certificate-reconstructable, but
they add rows, columns, or cone variables while removing no demonstrated
greenbea work. The existing homogeneous-auxiliary evidence is the closest
measured representative: cold reaches the auxiliary-optimal face at pivot
2,060, the best integrated boundary is

```text
2,050 + 10 + 2,183 = 4,243 pivots,
```

only **3.55%** below 4,399. The most optimistic fused-core opportunity is
**4.13%**, and exact alternation bottoms at 4,328 pivots. The current IPM route
also stalls on a dual certificate rather than merely lacking a final embedding
certificate. No particular homogeneous self-dual construction supplies a
measured 20% opportunity, so none earns a high-risk implementation from this
census.

This does not prove all future homogeneous self-dual algorithms impossible; it
states only that an exact lift without a new measured path invariant is
unfunded. Evidence: `experiments/phase_transition_falsifier_2026_07_22.md` and
`experiments/greenbea_ipm_stall_2026_07_18.md`.

## Evidence and integrity

The census read the following repository evidence:

- `AGENTS.md`
- `docs/SESSION-HANDOFF.md`
- `docs/NEXT-GOAL-PROMPT.md`
- `docs/HANDOFF.md`, every entry from `PRESOLVE V2 SHIPPED` through EOF
- `experiments/greenbea_dossier_2026_07_18.md`
- `experiments/creative_attack_dossier_2026_07_21.md`
- `experiments/greenbea_presolve_diff_2026_07_17.md`
- `experiments/greenbea_pivot_gap_2026_07_17.md`
- `experiments/probe_activeset_2026_07_18.md`
- `experiments/probe_blockds_2026_07_18.md`
- `experiments/probe_schur_2026_07_18.md`
- `experiments/c3_scaling_families_2026_07_21.md`
- `experiments/sparse_primal_falsifier_2026_07_22.md`
- `experiments/sifting_falsifier_2026_07_21.md`
- `experiments/krylov_basis_solve_probe.py`

Repository inspection used only read-only `pwd`, `rg`, `sed`, `wc`, `find`,
and `jq` commands. Numerical censuses were inline, in-memory Python/SciPy runs
with `PYTHONDONTWRITEBYTECODE=1`; the native solver was invoked only with its
existing `LINPROGX_DS_EXPORT_BASIS=1` diagnostic to return bases in memory.

There was no network access, external solver-source inspection, package
operation, Git operation, environment mutation, or source edit during the
census. The numerical probe intentionally wrote no raw artifact, so there is
no artifact path or artifact hash to cite. This Markdown report is the only
banked deliverable.
