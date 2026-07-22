# Post-dossier literature and fresh-structure census (2026-07-22)

## Verdict

**NO FUNDED CPU MECHANISM.** A fresh Overmind wave checked three primary works
published after the earlier dossier cutoff and seven orthogonal algorithm,
factor, and structure hypotheses. None has a credible route to greenbea's
required **17.7013% whole-wall reduction** under protocol v3's four CPU vCPUs,
fixed `eps=2e-5`, global-policy rule, and certificate-backed status semantics.

This is a scoped funding closure, not an impossibility theorem. No production
source changed, so no v3 recertification is warranted and the certified board
remains **23W-0P-1L**.

## Primary-work audit

The control plane read algorithms and public results, never external solver
source:

- Steiner et al., [Elasticity in Parallel Sparse Triangular
  Solve](https://arxiv.org/abs/2607.02324), published 2026-07-02.
- Djeloud, [The Dual Simplex Method for Solving Bounded-Variable Linear
  Programs](https://iapress.org/index.php/soic/article/view/3489), published
  2026-07-17.
- Zhang, Ploskas, and Sahinidis, [A novel linear optimization presolve
  technique based on Fourier-Motzkin
  elimination](https://link.springer.com/article/10.1007/s12532-026-00316-3),
  published 2026-04-10.

### Exact stale-synchronous triangular scheduling: KILL

ElasticDivide preserves the triangular system exactly while allowing bounded
scheduler staleness to elongate cross-core dependencies and overlap work. Its
published regime is synchronization-heavy many-core SpTRSV; the fixed v3
machine has four vCPUs and greenbea's changing Forrest-Tomlin factor makes the
schedule dynamic.

The arithmetic closes the opening before implementation:

- FTRAN+BTRAN are 34.62% of measured whole wall. Perfect two-core scaling of
  the entire slice saves only `34.62% / 2 = 17.31%`, below the 17.7013% board
  gap.
- Perfect three-core compute takes `47.425 / 3 = 15.808us` per equivalent
  pair, leaving only `4.220us` for all synchronization, FT-chain work, and
  schedule maintenance under the `20.028us` pair gate.
- Live `U'` averages 57.68 levels and reaches 154. Within an FT interval,
  34.18% of FTRAN and 13.53% of BTRAN level labels change. A static schedule
  therefore cannot be amortized across the roughly 295 solves per refactor.
- Freezing `U` returns to the displaced product-form update path. Scheduling
  only immutable `L/L^T` covers roughly 1,548--1,984 off-diagonal entries and
  the exact local prototype already regressed combined solves 21--23%.

Result: `KILL_ELASTIC_STALE_SPTRSV_V3`.

### Bounded-variable dual support / enhanced long step: KILL

The paper's support method walks sorted reduced-cost breakpoints along one
dual direction, subtracts `|alpha_j| (u_j-l_j)` from the piecewise-linear
slope at boxed variables, flips crossed bounds, and replaces one support
column when the slope changes sign. Modulo signs, this is the generalized
ratio / bound-flipping ratio test already implemented locally.

The local mechanism has no unused greenbea runway:

- greenbea has 3,611 lower-only and only 257 boxed structural columns;
- all 64 edges in the terminal first-64 pool, representing 34 unique entering
  columns, are lower-only, so none supplies a finite-width bound flip;
- the existing BFRT run makes 1,443 rather than 1,399 flips and saves 101
  pivots, but ratio time rises from 73.0ms to 180.5ms and the instrumented
  phase rises from 411.1ms to 530.7ms; and
- every long step still changes one support column, so its exchange width is
  one rather than the predeclared median-width gate of 18.

Result: `KILL_DUAL_SUPPORT_LONG_STEP_GREENBEA`.

### Inequality-only Fourier-Motzkin presolve: KILL

This is genuinely outside the existing equality aggregation audit. The paper
eliminates variables that occur only in inequalities, admits projected rows
only under row/nonzero growth gates, and provides primal, dual, and basis
postsolve reconstruction.

Raw LPnetlib row senses were conservatively inferred only when a row had a
unique zero-cost `[0,inf]` singleton `+/-1` slack. Every one of greenbea's 207
inferred inequality rows passed that unambiguous test. Including finite-bound
inequalities, only 14 non-slack structural columns are eligible:

| incidence `(m+,m-)` | candidate count |
|---|---:|
| `(1,2)` | 11 |
| `(2,1)` | 2 |
| `(1,1)` | 1 |

They touch only eight unique raw model rows and 96 raw nonzeros (0.309%). One
candidate is already removed by current presolve, leaving 13. Four
singleton/bound eliminations remove two or three nonzeros each; the ten
degree-two candidates add 9--11 nonzeros before redundancy checks and overlap
in only two row pairs. Even gifting deletion of all eight touched prepared
rows gives the deliberately unrealistic dense-cubic proxy
`1-(1517/1525)^3 = 1.566%`, an order of magnitude short of the board gap.

The global detector is not vacuous: `woodw` and `pds_10` have zero candidates,
while `cre_a` has 2,687 and `80bau3b` has 9,615. It simply finds no material
greenbea opportunity.

Result: `KILL_INEQUALITY_FME_GREENBEA`.

## Independent factor-science census

### Dynamic HSS/HODLR whole-inverse compression: KILL

For a basis exchange `B' = B E`, with
`E = I + (d-e_r)e_r^T` and `d=B^-1 a_q`, the exact inverse update is

```text
B'^-1 = B^-1 - ((d-e_r)/d_r) (e_r^T B^-1).
```

A hierarchical representation could retain dense off-diagonal blocks as
shared low-rank factors rather than coefficientwise top-K sparsification. It
therefore escapes the earlier traversal, SIMD, fixed-top-K, selected-row, and
global-row-transform constructions. It still must update or recompress the
changing inverse every pivot; leaving updates lazy is the old PFI chain.

Exact inverses of authoritative bases at pivots 512, 1536, 3072, and 4096
gave these two top-level off-diagonal numerical ranks after LU row/column
permutation:

| pivot | relative `2e-5` | relative `1e-9` |
|---:|---:|---:|
| 512 | 30 + 30 | 31 + 30 |
| 1536 | 53 + 39 | 55 + 39 |
| 3072 | 61 + 6 | 65 + 36 |
| 4096 | 9 + 61 | 62 + 67 |

Independent row/column RCM is worse: relative-`2e-5` rank sums are 566, 410,
309, and 250. Even the best LU-permuted sum of 60 requires at least
`60 * 1525 = 91,500` top-level generator coefficients. Charging only one
eight-byte read and write per coefficient per pivot at the favorable measured
52.7GB/s, then applying the established `4399/4873` pair normalization, costs
`25.078us/equivalent pair`. This already exceeds the `20.028us` gate before
deeper levels, applications, recompression, indices, residual authority,
refresh, or exact fallback.

Result: `KILL_DYNAMIC_HIERARCHICAL_INVERSE`.

### Fresh ordering / explicit inverse-factor layout: KILL

For every exact permutation, `P B Q = L U` and
`B^-1 = Q U^-1 L^-1 P`. Ordering may change fill but not the true dense
`B^-T e_r`. Four previously measured orderings give identical solve support;
the best standard factor fills are already 11,691, 12,668, and 14,458.
Bordered ordering helps the early basis by 9.6% but regresses later bases
27--57%. The strongest exact alternate layout then measured 52.495us/pair
versus the current 47.425us/pair.

Result: `KILL_FRESH_SOLVE_ORDERING_SUCCESSOR`.

## Independent structural census

### Exact component / thin-border decomposition: KILL

Prepared greenbea is one connected bipartite component containing all 5,393
row and column nodes. Removing the top 15% highest-degree rows still leaves
92.01% of columns in one component, while the removed border owns 54.81% of
all nonzeros. At 5% removal the core still has 97.70% of columns. Exact
independent-component opportunity is zero; a master carrying more than half
the matrix to detach only 7.99% of columns cannot fund 17.7%.

The census distinguishes other shapes: at a 5% row border `cre_a` leaves only
34.40% of columns in its largest component, while `woodw` still leaves 88.70%
even at 15% and `pds_10` leaves 80.65%.

Result: `KILL_GREENBEA_COMPONENT_BORDER`.

### Generalized-network specialization: KILL

Using the generous structural definition “prepared column degree at most
two,” only 767/3,868 greenbea columns (19.83%) qualify and they carry 5.14% of
nonzeros. The other 80.17% of columns remain a generic LP core. Even gifting
all qualifying matrix work for free caps the opportunity at 5.14%; restricting
the credit to pricing, ratio, and reduced-cost work gives roughly 2.54% whole
wall. The detector correctly identifies network-heavy sentinels (`pds_10` is
79.60% qualifying columns; `80bau3b` is 81.72%) but rejects greenbea.

Result: `KILL_GREENBEA_NETWORK_ROUTE`.

### Exact row sketch / approximate sketch with correction: KILL

Prepared greenbea has full structural row rank 1,525/1,525. Its row-normalized
Gram matrix factors nonsingularly; three random solves have maximum relative
infinity residual `8.48e-14`, with `|diag(U)|` from `7.89e-3` to `1.22`.
Every checked sentinel is also full structural row rank. An exact feasible-set
preserving sketch therefore needs `k >= 1525` and gives zero dimension
reduction. Making the sketch approximate requires an exact correction path and
returns to the already-unfunded approximate-basis, Krylov, sparse-inverse, and
certificate-fallback families.

Result: `KILL_EXACT_ROW_SKETCH_GREENBEA`.

## Independent trajectory census

### Lexicographic criss-cross macro-batching: KILL

A least-index, lexicographically perturbed scalar criss-cross path can cross
the old KKT merit barrier while preserving a finite support-order proof. The
proof belongs to its ordered scalar pivots. Directly jumping to a rank-32
endpoint does not inherit it; replaying the intermediates retains scalar
pricing, ratio, and update costs.

Existing scalar-trajectory panels commit only 1.281 pivots per requested
width-four panel, and only 101/4,395 panels preserve the next three choices.
Public-API sequential suboptimization corroboration takes 4,669 pivots and
0.908128s versus serial's 3,309 and 0.384212s. At the established 90.5us per
pivot, 4,399 decisions cost about 0.398s. Nominal width-32 refactors add about
0.143s before crediting existing refactors, leaving roughly 0.507s—already
above the complete 0.448351702s gate before other costs.

Result: `KILL_SCALAR_CRISS_CROSS_MACROBATCH_ECONOMICS`.

### Homogeneous self-dual / proximal interior globalization: KILL in scope

An interior embedding can move outside the 512 generated endpoint exchanges
and has a clean optimal/infeasible/unbounded certificate map. Its iterates are
not legal complementary bases, however, and crossover is a separate active-set
trajectory. The current greenbea IPM reaches primal residual `7.942e-10` and
complementarity `3.013e-9` but retains nine infinite-side sign violations
(worst `4.287e-7`) and then produces a nonfinite Newton direction at iteration
58. Fully regularized runs take 12.716s without certification; even the
uninstrumented 0.675--0.747s baselines exceed the 0.448351702s gate. The fixed
width-32 crossover successor also reverses 15 entering directions and worsens
its squared-KKT potential 155.067x.

Result: `KILL_INTERIOR_GLOBALIZATION_CURRENT_PROTOCOL`.

## Reproduction and evidence classification

Standalone diagnostics:

- `experiments/fresh_factor_census.py`
- `experiments/fresh_structural_census.py`

Canonical commands and independently reproduced output hashes:

```bash
UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m experiments.fresh_factor_census \
  > /tmp/fresh-factor-census.json
sha256sum /tmp/fresh-factor-census.json
# 3b433be5b8dded385b299e7e2b8e06215b175b7e07c46957ae662d6f340c3d47

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python experiments/fresh_structural_census.py \
  > /tmp/fresh-structural-census.json
sha256sum /tmp/fresh-structural-census.json
# ffb60489015bc86a303a1f3acd32f3f6939438c62d3e1c734e7ad9f720bfad5b
```

Source hashes before commit were
`0d7383e9ea8de1c320c4716a2e90de54d2e1a8b31ec794e287b7a5184df117a0`
for the factor census and
`0fa6b83d4db4cd4049496a5ec1229a362ea90ee801344464215cf68bbd9f314f`
for the structural census. The two workers had no network and changed no
production path.

Inherited authoritative evidence:

- `experiments/remaining_frontier_census_2026_07_22.md`
- `experiments/rich_inverse_falsifier_2026_07_22.md`
- `experiments/lsa_level_sched_2026_07_19.md`
- `experiments/greenbea_pivot_gap_2026_07_17.md`
- `experiments/pdas_lookahead_falsifier_2026_07_22.md`
- `experiments/pdas_block_merit_falsifier_2026_07_22.md`
- `experiments/greenbea_ipm_stall_2026_07_18.md`
- `experiments/probe_schur_2026_07_18.md`

Counts, ranks, residuals, and timings above are measurements. Dense-cubic,
free-work, ideal-scaling, bandwidth, and per-pivot projections are explicitly
favorable opportunity bounds or funding inferences, not native timing lower
bounds. No candidate cleared its preimplementation opportunity/authority gate,
so no production implementation or v3 run was authorized.

## Scoped reopening condition

Reopen only with at least one of:

1. a complete certificate-authoritative factor/solve construction below
   `20.028us/equivalent pair` on the changing greenbea bases;
2. a computable well-founded trajectory whose generator directly produces
   legal rank-at-least-18 endpoint exchanges and can test full completion plus
   the `0.448351702s` charge in a bounded S0;
3. a material greenbea structural reduction with exact primal/dual/basis
   reconstruction and a measured footprint above the 17.7013% gap; or
4. explicit authority for a new head-to-head hardware protocol.
