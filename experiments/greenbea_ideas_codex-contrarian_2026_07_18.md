# greenbea contrarian ideas — 2026-07-18

## Scope and accounting

This thread treats the dossier's closed-axis catalogue as an exclusion set, not
as a list of knobs to rename.  The local public comparison is 0.4160 s versus
0.2446 s, so the required saving is 41.2% (171.4 ms).  On the instrumented
dual-simplex wall, pivot-row + BTRAN + FTRAN + ratio test + reduced-cost update
+ LU update + refactorization account for 97.8% of time.  At 4,399 pivots and
95 us/pivot, holding the trajectory fixed requires about 56 us/pivot; matching
HiGHS's 3,309 pivots on our reduction would still require 74.5 us/pivot, while
matching its raw-route 2,836 pivots would require 86.9 us/pivot.  Those numbers
rule out any proposal that merely improves one small kernel or merely reaches
pivot parity.

No new measurements were needed for this ideation pass.  All claims below are
mechanism hypotheses and every ranking is conditional on its falsifier-first
probe.  Literature references are from memory; no network or external solver
source was used.

## Ranked idea 1 — exact block dual simplex via precomputed pivot-row panels

### Mechanism (8 sentences)

1. Replace each major iteration's single leaving row with a small, globally
   adaptive panel of the top `p` currently infeasible basic rows, with `p`
   chosen from a fixed ladder such as 1/2/4/8 based on the previous panel's
   commit rate rather than on the problem name.
2. From one basis snapshot, compute all `p` rows of `B^-1` with a multi-RHS
   BTRAN, then price them together as `R^T A`, reading each matrix panel once
   and producing `p` complete old-basis pivot rows.
3. Choose the first entering column by the existing exact Harris test, compute
   its one ordinary base-basis FTRAN `u = B^-1 a_q`, and commit that pivot.
4. For every surviving candidate row `i`, update its already-materialized
   pivot row exactly by the Gauss-Jordan relation
   `α'_i = α_i - (u_i/u_r)α_r` (and `α'_r = α_r/u_r`), which is precisely
   the old α information that the killed online cross-pivot-reuse idea did
   not possess.
5. Keep the original LU factors frozen during the minor iterations and represent
   committed replacements by a small dense product `T_k = E_k^-1 ... E_1^-1`,
   so a later entering column uses one solve with the snapshot basis
   followed by the cheap (T_k) correction.
6. Re-run leaving admissibility and the full ratio test after every minor pivot,
   dropping candidates invalidated by earlier pivots; thus the block changes the
   trajectory but never approximates a simplex step.
7. At panel end, install the accepted replacements as one compact update panel
   or refactorize if its pivot-growth test fails, then start a new major
   iteration.
8. This is the suboptimization/s-step family used in parallel revised-simplex
   work (for example Huangfu and Hall, 2018, from memory), but here its purpose
   is cache reuse and BLAS-width work on one CPU, not thread-level speedup.

### Wall term and ceiling

The primary attacked pool is BTRAN 18.9% + pivot-row 24.8% + ratio 14.9% + LU
update 6.1% = **64.7%**; FTRAN remains mostly sequential at first.  A 3x
effective acceleration of that pool saves 43.1% and crosses the 41% wall, while
2x saves only 32.4% and is dead.  Equivalently, ignoring overhead, the attacked
pool needs at least a 2.73x speedup.  Multi-RHS BTRAN, one-pass `p`-row pricing,
SIMD ratio sweeps, and a deferred update panel must therefore all contribute;
"batch BTRAN only" has an 18.9% absolute ceiling and is not this idea.

### Nearest closed axis, and why this differs

The nearest kill is cross-pivot support reuse: α'_k needs the old α_k.  This
proposal pays once to compute a panel containing each candidate's old α_k,
then applies the exact basis-replacement formula; it does not infer α_k from
the previous chosen pivot row.  It is also not a new leaving rule: every minor
step still uses an exact infeasibility scan and Harris ratio test, and invalid
candidates are discarded.

### Falsifier-first probe and kill criterion

Add a diagnostic `p=4` mode that computes four BTRAN/pivot rows at the start
of a block but lets the existing solver commit pivots one at a time.  After each
commit, transform the three shadow rows with the formula above and record (a)
how many remain primal-infeasible, (b) how many pass an exact recomputed ratio
test, (c) pivot growth, and (d) measured multi-RHS BTRAN + panel-pricing time
versus four ordinary iterations.  The probe need not change the returned path
or result.  **Kill** if the mean committed block is below 3, if transformed rows
disagree with a fresh BTRAN/pricing audit above `1e-10` relative error, or
if the projected 64.7% pool speedup is below 2.8x including panel overhead.

### Risks

The 96--99.97% consecutive support overlap is encouraging for panel locality,
but candidate *row* survival may still be low.  Small pivot matrices can amplify
roundoff, so the panel needs growth limits and a p=1 fallback.  Refactor and
final residual/dual-certificate gates remain unchanged.  Determinism requires
original row/column IDs for all ties and a deterministic block-shrink rule.

## Ranked idea 2 — low-precision trajectory scout, direct terminal-basis certificate

### Mechanism (8 sentences)

1. Run the ordinary deterministic pivot logic as a speculative float32 scout,
   recording leaving basis positions, entering columns, flips, and the final
   nonbasic bound statuses, but never treating its numerical result as an
   answer.
2. Use float32 factors and work arrays for the scout, float64 accumulation only
   at prescribed residual checks, and iterative refinement when a solve's
   backward error exceeds a global threshold (Carson and Higham, 2017, from
   memory).
3. Once the scout says optimal, reconstruct its terminal basis directly from
   the trace instead of replaying 4,399 pivots in double precision.
4. Factor that terminal basis from the original float64 matrix, solve once for
   `x_B` and once for `y`, recompute every original-unit reduced cost and
   primal residual, and accept only through the existing certificate gates.
5. If the basis is singular, a bound status is inconsistent, or any certificate
   fails, discard the scout and run the current float64 solver from scratch.
6. A checkpointed double replay from the scout's last stable refactor boundary
   is an optional middle fallback, but full replay is not the primary design.
7. Pure "solve once in double, then replay faster" cannot win a one-call
   benchmark because the first solve has already spent 100% of baseline wall;
   only a cheaper speculative first pass makes trajectory knowledge valuable.
8. The paired board runs are isolated worker subprocesses, so caching a trace
   across the seven benchmark repetitions is neither available nor a legitimate
   source of the claimed per-call speedup.

### Wall term and ceiling

The float scout attacks essentially the full 97.8% pivot-loop pool.  If that
pool runs 2.2x faster, the scout costs 44.5% of baseline; adding 2.2% untouched
wall and an 8% terminal factor/certificate pass projects 54.7% remaining, or a
45.3% win.  At only 1.8x, the same arithmetic is about 64.5% remaining and
misses badly.  The hard economics are therefore **at least about 2.0x scout
speedup with a terminal pass no larger than 8%**, not merely "float is faster."

### Nearest closed axis, and why this differs

The nearest kills are cross-pivot reuse and warm-start/basis transfer.  This
does not reuse online algebra and does not obtain a basis from another solver;
it speculatively discovers a basis inside the same public call, then validates
that basis against the original LP.  The certificate, not agreement with the
float trajectory, is the correctness argument.

### Falsifier-first probe and kill criterion

Build the smallest scout variant: float32 `a_data`, LU values, rho, alpha,
reduced costs, and basic values; keep indices and all decision tie-breaks
unchanged.  On greenbea plus the existing certified DS battery, record scout
wall, terminal-basis factorability, certificate pass rate, fallback rate, and
how early the float trace diverges from double (divergence itself is not a
failure).  Include adversarially scaled tests because Ruiz scaling may hide the
main failure mode on Netlib.  **Kill** if greenbea scout + terminal certificate
is not below 0.245 s locally, if the scout core is below 2.0x, or if more than
5% of otherwise certified-optimal battery cases require a full fallback.

### Risks

Float32 pivot comparisons can select tiny or unstable pivots, so a global
growth/backward-error gate is mandatory.  A terminal optimal basis is easy to
certify, whereas infeasibility and unboundedness need valid rays; speculative
non-optimal statuses should fall back rather than be trusted.  The worst case
does two solves and regresses, so routing must be based on general dimensions
and numerical diagnostics, never `greenbea` identity.  Extra float and double
factor storage is acceptable at this size but must have explicit ownership.

## Ranked idea 3 — certified dormant-column pricing with lazy reduced costs

### Mechanism (9 sentences)

1. Treat the 83.2%-at-bounds fact as a screening hypothesis, not as permission
   to fix continuous variables: a positive reduced cost and a nonzero duality
   gap cannot safely prove an LP variable is exactly at its bound.
2. Maintain the dual vector `y` explicitly after each pivot via its sparse
   rho update, and regard cached reduced costs for dormant columns as stale
   values with rigorous error intervals.
3. When column `j` is refreshed, store exact `r_j = c_j - a_j^T y`, its
   refresh epoch, and block norms of `a_j`.
4. Between refreshes, bound reduced-cost drift by blockwise Cauchy bounds on
   the accumulated dual motion, yielding an interval
   `[r_hat_j - e_j, r_hat_j + e_j]` without touching the column's nonzeros.
5. For a new leaving row rho, bound
   `|alpha_j| = |a_j^T rho|` from above with the same block norms; the
   distance of the reduced-cost interval from zero divided by this upper bound
   is a certified lower bound on `j`'s dual ratio.
6. Price a small active set exactly to obtain a provisional Harris threshold,
   then omit every dormant column whose lower-bound ratio exceeds that
   threshold and whose reduced-cost interval preserves its bound-feasible
   sign.
7. Refresh every column whose certificate fails, repeat until no omitted
   column can beat the threshold, and only then commit the ordinary exact
   pivot.
8. Columns whose intervals remain wide are refreshed in deterministic epochs,
   while refactorization and final optimality perform a full reduced-cost scan.
9. The working nonbasic universe can therefore shrink sharply without ever
   declaring a variable permanently fixed or weakening the final certificate.

### Wall term and ceiling

The attacked pool is pivot-row 24.8% + ratio test 14.9% + reduced-cost update
9.7% = **49.4%**.  Because the 41% target is 83% of that entire pool, screening
must remove roughly 90% of its weighted work while spending less than about
3.5% on interval maintenance to win by itself.  The median ratio-candidate
count of 182 versus alpha support 3,625 says a large rejection population
exists, but it does not prove the cheap norm bounds will separate it.

### Nearest closed axis, and why this differs

The nearest closed axes are bounded-variable presolve and cross-pivot support
reuse.  Presolve is at a propagation fixpoint and cannot safely fix these
continuous variables; this proposal makes reversible, per-pivot pricing
omissions backed by ratio bounds.  It also does not reuse an obsolete alpha:
it proves that the unknown current alpha cannot matter, and computes it exactly
whenever that proof fails.

### Falsifier-first probe and kill criterion

Replay a recorded double trajectory diagnostically and, at each pivot, compute
the proposed block-norm lower ratio bounds before looking at the true alpha.
Count false survivors (safe but expensive), never false exclusions (a bug),
weighted matrix nnz touched, interval refreshes, and the fraction of pivots for
which the provisional active threshold must be reopened.  A second microprobe
times explicit sparse-y updates plus interval maintenance against current
reduced-cost updates.  **Kill** if fewer than 90% of pivot-row/pricing nnz are
certifiably dormant at the exact threshold, if any omitted column beats the
chosen pivot, or if projected end-to-end saving is below 35% (too little margin
to compose into a 41% win cleanly).

### Risks

Triangle-inequality drift bounds may become useless after only a few pivots,
causing a refresh storm.  Coordinate blocks improve tightness but increase
metadata and cache traffic.  Free and one-sided columns need separate sign
logic, and near-zero reduced costs must always be active.  The full exit scan
and original-unit certificate remain authoritative.

## Ranked idea 4 — permutation-stable, SIMD-tiled dual representation

### Mechanism (8 sentences)

1. Apply a deterministic bipartite bandwidth/locality ordering to rows and
   columns after presolve, while carrying original IDs so all mathematical
   tie-breaks remain permutation-invariant.
2. Store the 23,274 nonzeros both in ordinary CSC for basis assembly and in a
   SIMD-width panel format (for example 8 or 16 adjacent columns with
   structure-of-arrays indices and values) for pricing.
3. For sparse rho, retain CSR scatter; once a byte-count model predicts that
   random scatter is dearer, switch to panel-CSC dot products that stream all
   columns and gather the compact rho vector.
4. Fuse only data movement, not decisions: write panel alpha values and
   admissibility masks contiguously, then run the Harris pass and reduced-cost
   update as deterministic vector loops in original-ID order for ties.
5. Put basis-position vectors, FTRAN/BTRAN workspaces, and FT update records in
   the same row ordering used by the factors so sparse solves touch clustered
   cache lines rather than repeatedly translating indices.
6. At refactorization, repack factor/update metadata once; do not reorder the
   mathematical basis or alter the Markowitz stability test.
7. Select CSR versus panel-CSC using a global analytic byte threshold based on
   support size and panel width, not a fixture-specific switch.
8. This attacks the representation used by every expensive step while leaving
   the pivot rule, trajectory semantics, and final double certificate intact.

### Wall term and ceiling

If the layout improves pivot-row, BTRAN, FTRAN, ratio, and reduced-cost update,
the attacked pool is **86.2%**.  Halving that pool saves 43.1% and barely wins;
halving only pivot-row + ratio + reduced-cost work saves 24.7% and is dead.
Thus the proposal lives only if the ordering/panel layout also gives material
solve locality, rather than being a prettier pricing kernel.

### Nearest closed axis, and why this differs

Dense-U FTRAN was killed across bandwidth regimes, block-row uplook is shipped,
and within-pivot support reuse is shipped.  None changes the sparse matrix's
row/column locality, panelizes `A^T rho`, or makes factor/update indices share
one cache ordering.  This is a whole-loop representation experiment with an
identical logical trace, not another dense-U or scan-fusion attempt.

### Falsifier-first probe and kill criterion

Capture one greenbea trajectory and benchmark three read-only kernels under
several *generic* orderings: pivot-row pricing, sparse BTRAN/FTRAN traversal,
and ratio/reduced-cost vector passes.  Require the replay to produce identical
logical entering/leaving original IDs and numerical values within the existing
tolerance; collect cycles, L1/L2/LLC misses, and bytes touched.  Then project
using the dossier's phase weights before integrating anything into the solver.
**Kill** if the weighted attacked pool is not at least 1.8x faster in isolation,
if solve locality regresses, or if the projected end-to-end saving is below
35%—a result below that is useful engineering but not a greenbea campaign.

### Risks

At roughly 23k matrix nonzeros, much of `A` may already fit in private cache,
making graph ordering irrelevant; the real loss may be dependencies rather than
misses.  Panel padding can add more bytes than SIMD removes.  Reordering factor
indices can change fill or numerical tie outcomes unless original IDs and
stability comparisons are carefully separated from storage order.  Duplicate
formats increase setup cost, which is inside the public solve wall.

## Mandated phase-structure verdict — already falsified, not ranked as new

A generic "real dual Phase-1/Phase-2" is not a genuinely new class in this
repository.  `docs/HANDOFF.md` records a Maros-style composite-infeasibility
Phase-1 with true bounds and no boxing: greenbea Phase-2 pivot count worsened by
79%, with objectives and coverage preserved.  Independently, importing the
observed HiGHS Phase-1 exit into linprogx leaves 3,529 pivots and flat wall
because solves densify, while HiGHS itself still needs 1,594 pivots from the
same boundary; that is strong evidence that the label boundary is not the
missing mechanism.  The cheapest behavioral probe one would otherwise run is
exactly this pair: (1) an auxiliary-objective Phase-1 followed by an internal
cost restore, and (2) a phase-boundary basis continuation under both engines,
with phase objective, true-bound dual infeasibility, support density, and wall
recorded.  Its kill criteria would be no clean zero Phase-1 objective, less
than a 20% total-pivot reduction, or a denser continuation that misses 74.5
us/pivot; the historical experiments already meet the kill.  A future phase
proposal is admissible only if it names machinery beyond the auxiliary
objective/boundary basis—block suboptimization in idea 1 is such machinery,
but calling it "Phase-1" would add no explanatory value.

## Bet

My bet is **idea 1, exact block dual simplex**, because it is the only proposal
that (a) attacks enough measured wall without requiring a numerically different
first solve, (b) has an exact algebraic answer to the dossier's cross-pivot
reuse objection, and (c) has a very cheap decisive probe: candidate survival
and panel speed determine whether it lives before a production implementation.
The float scout is the higher-variance second bet and may produce the largest
win, but it needs a genuine >2x core plus a remarkably cheap terminal
certificate; safe dormancy and tiled layout are more likely to yield useful
partial improvements than a standalone 41% flip.  If only one experiment is
funded, run the `p=4` shadow-panel probe and kill it ruthlessly unless it
projects at least 2.8x on the 64.7% pool.
