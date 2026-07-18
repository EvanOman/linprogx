# greenbea performance ideas — GPT-5 thread (2026-07-18)

## Quantitative frame

This report treats the dossier as the evidence boundary and does not reopen any
closed axis.  On the local numbers, the dual-simplex baseline is approximately
`4,399 * 95 us = 0.418 s`; a 41% reduction gives a target near `0.247 s`.
At the current 95 us/pivot, a pivot-only win would therefore need at most about
2,600 pivots.  Conversely, 2,836 pivots at 95 us/pivot would still take 0.269 s,
so even complete pivot-count parity with HiGHS needs about an additional 8%
per-pivot reduction.  The ideas are ranked by my estimate of expected value,
not by implementation ease.

## 1. Exact rank-k dual block exchanges

### Mechanism

Replace a run of scalar dual-simplex pivots with one exact block basis exchange,
in the spirit of dual-simplex suboptimization (for example, the block/parallel
ideas described by Huangfu and Hall, 2018), but make the block the unit of
algebra rather than merely preselecting scalar pivots.  Select `k` strongly
primal-infeasible basic rows, solve their BTRANs as a panel
`B^T Y = E_P`, and form the corresponding pivot-row panel `Y^T A_N` in one
support-coherent pass.  From the union of admissible entering columns, choose a
same-size set `Q` by solving a tiny block assignment/LP whose constraints require
the proposed post-exchange reduced costs to remain dual feasible and whose
objective reduces aggregate primal infeasibility.  Compute
`X = B^-1 A_Q` with a multi-RHS FTRAN, reject a singular or ill-conditioned
pivot block `K = X_P`, and apply the simultaneous basis replacement through the
small `K^-1` Woodbury correction.  Basic values, reduced costs, and edge data are
then updated by rank-k fused sweeps, followed by an exact dual-feasibility and
bound check; a failed check shrinks `k` and ultimately falls back to one scalar
pivot.  The measured 96–99.97% consecutive support overlap is useful because a
panel should traverse almost the same sparse locations once, while the method
never assumes that a new scalar pivot row equals or can be recovered from the
previous one.  A global policy can choose `k` from conditioning and accepted
panel width, with no fixture-specific thresholds.

### Wall term and ceiling

This primarily attacks **us/pivot**, with a possible secondary attack on
**pivots** because the block objective chooses a jointly useful set rather than
accepting a myopic sequence.  Pivot row, BTRAN, FTRAN, ratio test, and
reduced-cost update total 86.2% of measured wall.  If panels make that region
only 1.91 times faster, the whole-wall saving is
`0.862 * (1 - 1/1.91) = 41.1%`; an ideal width-four panel has a much larger
64.7% amortization ceiling before block-selection and verification overhead.
The hard requirement is therefore not fourfold speed, but an accepted average
panel large and coherent enough to deliver roughly 1.9x over those five
buckets.  Any pivot-count reduction creates overhead margin rather than being
required for the best-case arithmetic.

### Why this is not a closed axis

The nearest closed axis is **cross-pivot reuse**, killed because exact scalar
`alpha'_k` requires the old `alpha_k`; the other nearby closure is the family of
one-row **leaving rules**.  This proposal does not reuse a stale scalar alpha or
score one leaving row differently.  It explicitly computes all rows and
entering columns required for a rank-k replacement against the same old basis,
then uses a small exact block inverse to account for their coupling.  The
within-pivot support-reuse and Forrest–Tomlin work already shipped remain useful
inside the panel but are not the proposed mechanism.

### Falsifier-first probe

Add a trace-only block simulator around the current solver: every 16th scalar
iteration, capture the top 8 infeasible rows, perform one block BTRAN, construct
candidate blocks for `k = 2, 4, 8`, and verify the proposed endpoint against a
copy of the basis without changing the production trajectory.  Record accepted
width, `cond(K)`, exact dual-feasibility failures, aggregate infeasibility change,
panel support union, bytes touched, and measured panel BTRAN/FTRAN/matrix-update
time versus the same `k` scalar kernels.  **Kill the class** if median accepted
width is below 3, if more than 10% of attempted panels fall back for numerical
reasons, or if the measured projection including selection and verification is
less than a 41% whole-wall reduction with no pivot-count credit.  The probe
should also compare the block endpoint with `k` ordinary pivots; requiring the
same endpoint is unnecessary, but dual feasibility and monotone aggregate
primal progress are not.

### Risks

The small pivot block can be badly conditioned even when every scalar pivot
would pass Harris, and a simultaneous exchange may expose numerical growth that
scalar Forrest–Tomlin updates avoid.  Dual feasibility must be checked in full
precision after every panel, and the final existing residual/objective
certificate remains mandatory.  Deterministic row/column ordering is needed so
block assignment does not create run-to-run drift.  The fallback makes the
method constraint-compliant, but frequent fallback would erase the ceiling and
is intentionally an early kill.

## 2. Certificate-preserving active-bound sifting

### Mechanism

Turn the measured 83.2% bound-active solution structure into a dynamic restricted
master rather than a presolve deletion.  Keep all basic columns plus a working
set of nonbasic columns whose exact reduced costs are near eligibility or which
have recently participated in a ratio test; all other nonbasic columns remain
fixed at their current bounds and are omitted from pivot-row formation, ratio
passes, and per-pivot reduced-cost updates.  Solve the restricted master with
the unchanged dual-simplex mechanics for a globally fixed epoch, then recompute
the dual vector and exactly price every excluded column in one sparse matrix
pass.  Add every sign-violating or Harris-competitive excluded column as a
batch, evict columns only after a globally defined number of strictly interior
reduced-cost scans, and repeat.  Termination is permitted only after a full
original-column scan proves dual feasibility and the ordinary primal residual
and objective-gap certificate passes.  The first working set should be derived
from current reduced-cost margins and basis membership, not from greenbea-specific
column identities; a short initial epoch can gather these margins without an
IPM or transferred basis.  This is classical sifting/column generation applied
inside the dual-simplex trajectory, but greenbea's high active-bound fraction
and tiny median ratio-candidate set make it a specifically testable mechanism
here.

### Wall term and ceiling

This attacks **us/pivot** first and can also alter **pivots** by removing
distractor entering columns.  The three clearly column-wide buckets—pivot-row
formation (24.8%), ratio test (14.9%), and reduced-cost update (9.7%)—sum to
49.4% of wall.  If exactly the measured 83.2% ultimately active-at-bound columns
could remain dormant, the no-overhead ceiling is
`0.494 * 0.832 = 41.1%`, just enough to reach approximately 0.247 s.
That ceiling is razor-thin: full scans and extra epochs mean the practical route
also needs either a smaller live working set than 16.8%, a modest pivot reduction,
or some benefit in pivot-row locality.  For reference, a 10% pivot reduction
would provide about 42 ms of gross margin at the current rate.

### Why this is not a closed axis

The nearest closed axes are **parallel/dominated-column presolve** and the
shipped **Suhl bounded pivot search**.  Sifting proves no column redundant and
does not permanently eliminate one; it temporarily restricts the master and
restores any column that an exact global pricing scan says matters.  Suhl avoids
searching many already materialized candidates, whereas this proposal avoids
forming their pivot coefficients and updating their reduced costs on most
pivots.  It is also not an IPM crossover or starting-basis transfer, so it does
not inherit the measured dense-factor trajectory.

### Falsifier-first probe

Instrument an unmodified 4,399-pivot run and record, for every nonbasic column,
the pivots on which it enters the ratio-candidate set, actually enters the basis,
or changes reduced-cost eligibility.  Offline, replay epoch lengths of 16, 32,
and 64 pivots and compute the minimum oracle working set that would have retained
every eventual entering column, the number of wake-ups a legal exact scan would
require, and the bytes avoided in the three column-wide buckets.  Then run a
restricted-master shadow that performs the exact full scans but does not change
the production pivot decisions.  **Kill the class** if the live set exceeds 20%
of columns for half the run, if an exact scan is needed more often than every 16
pivots, or if scan overhead plus measured avoided work projects less than a 35%
wall saving before pivot effects; a mechanism with a 41.1% ideal ceiling needs
at least that much measured kernel saving to have a credible path through
overhead.

### Risks

Active at the optimum does not imply inactive throughout the path, and excluded
columns can become attractive rapidly as the dual vector moves.  Delayed pricing
can increase pivot count or create cycling even on a problem with almost no
ordinary degeneracy, so wake-up and eviction rules must be global and
deterministic.  Exact global scans are non-negotiable for the dual certificate;
no reduced-cost margin may be treated as proof without recomputation.  The
method stays within `eps=2e-5` because it changes search order only, not the final
certificate gate.

## 3. Dual-lane mixed-precision simplex with backward-error gates

### Mechanism

Maintain the authoritative problem data, basis identity, pivot comparisons, and
final certificate in FP64, but execute the bandwidth-heavy working lane in FP32.
Factor the basis in single precision, perform FP32 BTRAN/FTRAN and vector sweeps,
and recover FP64-quality solutions with residual-based iterative refinement
against the original FP64 matrix; promote an individual solve or trigger an
FP64 refactor whenever its normwise backward error exceeds a fixed function of
machine epsilon and the current basis condition estimate.  Compute ratio
quantities in FP32 in bulk, then recompute every candidate inside a conservative
Harris uncertainty band in FP64 before selecting a pivot.  Reduced costs and
basic values can use compensated FP32 updates between periodic FP64 rebuilds,
with interval error budgets forcing an early rebuild before a sign decision can
be ambiguous.  LU updates may stay FP32 only while growth and refinement counts
remain bounded; the FP64 lane always owns accepted basis statuses and can replay
the latest update if a gate trips.  This is a global numerical policy—thresholds
derive from backward error and uncertainty, not the fixture—and an exact FP64
residual/reduced-cost scan is still required at termination.  The route is
especially plausible for greenbea because sparse vector work is bandwidth-heavy
and the current factor trajectory, unlike transferred bases, is already the
sparse one we want to preserve.

### Wall term and ceiling

This attacks **us/pivot**.  Excluding refactorization, the listed pivot-row,
BTRAN, FTRAN, ratio, reduced-cost, and LU-update buckets total 92.3% of wall.
Those buckets need an aggregate speedup of
`1 / (1 - 0.41/0.923) = 1.80x` to save 41%; a clean 2x would save 46.2% and
leave about 5.2 percentage points for FP64 checks and promotions.  Including
single-precision refactorization raises the absolute ideal ceiling to 48.9%,
but the proposal should not rely on that extra 5.5% because refinement and
rebuilds are precisely where numerical risk concentrates.

### Why this is not a closed axis

The nearest closed item is **dense-U FTRAN**, which changed the kernel shape and
lost in all bandwidth regimes.  This proposal retains the sparse factorization,
current ordering, Forrest–Tomlin updates, and support behavior; it changes value
representation and adds a verified FP64 correction lane.  It also differs from
cross-pivot reuse because no previous alpha is assumed valid and from tolerance
tuning because all accepted pivots and the final certificate are checked at the
existing tolerances in FP64.

### Falsifier-first probe

Capture 200 representative basis states across both dual phases and write a
standalone solve/ratio microbench that runs FP64 and FP32-plus-refinement on the
same BTRAN/FTRAN right-hand sides.  Measure effective speedup including residual
formation, number of refinement steps, promotion rate, selected pivot identity,
and backward error; separately replay FP32 reduced-cost/basic-value updates until
an FP64 rebuild to measure sign uncertainty growth.  **Kill the class** if the
combined eligible kernels are below 1.8x after verification, if more than 5% of
solves require promotion or more than two refinement steps, or if any accepted
FP32 decision differs from the FP64 decision without having first entered the
uncertainty band.  Only after passing that state-replay test is a full solver
experiment justified.

### Risks

Ill-conditioned bases can make iterative refinement stall, and near-tied ratios
can turn tiny arithmetic differences into a very different trajectory.  A safe
implementation therefore may promote often enough to lose the bandwidth win;
that is a performance falsifier, not a reason to loosen checks.  FP32 LU growth,
subnormal handling, and architecture-dependent fused operations threaten
determinism.  Final primal feasibility, dual signs on one-sided columns, and
objective agreement must all be recomputed from FP64 data before reporting
optimality.

## 4. Early dual-face completion by bounded feasibility

### Mechanism

Exploit a property of the dual-simplex route that the failed IPM route does not
have: once dual phase I has completed, the current iterate carries a dual-feasible
certificate even while its basic primal point is infeasible.  At selected phase-II
checkpoints, form the complementary face of that certificate by fixing every
nonbasic variable with a strict reduced-cost sign at its minimizing bound and
allowing only zero-reduced-cost columns (normally the basis plus any tied
nonbasics) to move.  Ask a basis-free bounded-feasibility kernel—sparse QR with
active-bound exchanges, or a feasibility-only Newton method—to solve `A x = b`
inside that face.  If it succeeds, complementary slackness plus the already
dual-feasible vector gives an immediate optimality certificate without executing
the remaining simplex pivots.  If the exact face is infeasible, optionally test
one globally defined near-zero band by solving a small restricted LP, derive its
new dual vector, and globally price all excluded columns; any violation returns
control to dual simplex rather than being waived.  Checkpoints should be driven
by a global fall in primal-infeasibility norm, not by a greenbea iteration number,
and the feasibility factorization is discarded on failure so no dense basis is
transferred back.  This is a speculative handoff with a cheap, exact success
condition: it either constructs both sides of the certificate or resumes the
known route.

### Wall term and ceiling

This attacks **route** and, viewed from the original route, the remaining
**pivots**.  A checkpoint after 1,500 pivots has spent about
`1,500 * 95 us = 0.143 s`, leaving roughly 0.105 s for face completion under a
0.248 s target; after 1,800 pivots only 0.077 s remains.  In the best case, a
1,000-pivot checkpoint plus a 50 ms feasibility solve would take about 0.145 s,
a 65% reduction.  Unlike pivot parity, this mechanism can cross the 41% line
without lowering us/pivot, but it must find the certifiable face substantially
before 2,600 pivots.

### Why this is not a closed axis

The nearest closure is **IPM-crossover/starting-basis transfer**.  That work
harvested an IPM basis candidate, inserted it into dual simplex, and paid for a
dense solve trajectory.  Face completion starts from the native sparse
dual-simplex trajectory, transfers no basis into simplex, and asks a separate
primal feasibility question whose successful output is a complete certificate,
not a warm start.  It also differs from the killed IPM route: there is no attempt
to repair the nine pinned IPM dual signs because dual feasibility is inherited
from dual simplex before the handoff.

### Falsifier-first probe

Save `(y, reduced costs, basis, x_B)` every 200 phase-II pivots from the existing
run and, offline, solve only the exact complementary-face bounded-feasibility
problem with a dev-only oracle or sparse QR prototype.  Record the first
checkpoint with a feasible face, face dimension/rank, residual, solve wall, and
the full primal-dual gap under the saved `y`; then try a single fixed near-zero
band only if exact-face feasibility never appears.  **Kill the class** if no
certificate appears by pivot 1,800, if the face is rank-deficient in a way that
requires more than 20% of all columns, or if checkpoint cost plus projected
prefix wall exceeds 0.248 s.  This probe needs no production solver changes and
will probably falsify the idea quickly if nondegeneracy makes the correct face
appear only at the terminal basis.

### Risks

The almost complete absence of degeneracy is a double-edged fact: it makes the
face clean, but likely leaves only a square set of zero-reduced-cost basic
columns, in which case feasibility cannot improve until the correct basis has
already been found.  A near-zero expansion can quietly become a second LP solve
and lose the wall budget.  Rank decisions and bound hits require FP64 residual
checks, and the saved dual vector certifies optimality only for an exactly
complementary primal point.  Failed probes must resume without changing status
semantics or accepting an approximate primal as optimal.

## 5. Phase-response tomography for the unidentified 1,090 pivots

### Mechanism

Treat HiGHS as a black-box experimental subject on the identical presolved
matrix and identify the missing DSE-adjacent mechanism from response signatures,
without reading source.  Build deterministic families that independently vary
RHS infeasibility, objective/reduced-cost margins, finite-bound widths, and
row/column scaling while preserving sparsity and, where possible, the known
optimal basis over a small homotopy interval.  For each family, collect only
public runtime behavior: dual phase-I/phase-II iterations, bound flips, objective
and infeasibility traces, reinversion points if exposed in normal logs, final
basis statuses, and wall; collect the analogous native trace from linprogx
Dantzig and exact FG-DSE.  RHS-only sensitivity localizes primal phase machinery,
cost-only sensitivity localizes pricing/perturbation, bound-width sensitivity
localizes ratio/flip interaction, and scale sensitivity distinguishes normalized
edge machinery from raw pivot scoring.  Add paired “lesions” using documented
public options only when already known—one feature changed at a time—and cluster
the difference in the HiGHS-minus-linprogx pivot gap by phase and event type.
The deliverable is a behavioral specification such as “the advantage appears
only after phase transition and tracks bound-width homotopy,” precise enough to
justify one new implementation mechanism rather than another broad rule sweep.
All perturbation grids and seeds are global and predeclared, so this is
identification rather than per-problem tuning.

### Wall term and ceiling

This is a **pivot-mechanism identification route**, not by itself a wall fix.
Eliminating exactly the 1,090 simplex-internal excess pivots would change 4,399
to 3,309 and save about 24.8% at unchanged us/pivot.  Explaining the entire
4,399-to-2,836 difference has a 35.5% pivot ceiling, still short of 41%; a
resulting mechanism must either beat 2,836 pivots (roughly 2,600 is the pure
pivot target) or be paired with the approximately 8% us/pivot cut needed at
2,836.  The reason to retain this lower-ranked idea is that the dossier explicitly
isolates those 1,090 pivots but cannot yet assign them to phase, perturbation, or
ratio interaction—the next implementation choice is otherwise blind.

### Why this is not a closed axis

The nearest closed axis is the **leaving-rule family**, especially exact
FG-DSE, and the nearest completed measurement is **longest-step BFRT**.  This
probe does not propose another edge-weight formula or rerun a named rule.  It
factorially separates phase construction, cost perturbation, bound geometry,
and scaling responses to determine which machinery makes the same DSE label
behave differently.  Starting bases remain fixed on the identical reduction,
so it also does not reopen crash or basis transfer.

### Falsifier-first probe

Use 5–7 logarithmically spaced amplitudes in each of four one-factor homotopies,
plus a small predeclared interaction set, and run both solvers three times per
cell with presolve off.  Before the full grid, do a 12-cell pilot and verify that
phase counts and basis-status changes are observable and deterministic enough
to support inference.  **Kill the probe** if the pilot cannot localize at least
70% of the 1,090-pivot difference to a phase/event region, or if all four
response curves track linprogx FG-DSE within 10% after normalization and reveal
no actionable interaction.  A positive result must predict a held-out
perturbation's gap direction before any solver mechanism is implemented.

### Risks

Black-box response is not causal proof, correlated options can confound the
signature, and perturbations may change the optimal basis rather than merely
expose machinery.  Basis-stability checks and held-out predictions are therefore
required.  No source inspection, network access, or undocumented option mining
is needed or permitted.  Even a successful diagnosis does not close the wall
gap until its inferred mechanism passes the ordinary certificate and timing
gates.

## Self-assessment

I would bet on **exact rank-k dual block exchanges**.  It is the only proposal
here with substantial arithmetic headroom beyond 41%, it turns the unusually
high consecutive-support overlap into fewer sparse traversals without relying
on the algebra already falsified for scalar cross-pivot reuse, and it can change
both halves of the performance equation at once.  Active-bound sifting has a
cleaner prototype but its 41.1% ideal ceiling is almost exactly the required
gain, leaving too little room for full pricing scans; mixed precision has enough
ceiling but a narrow 5-point verification budget.  The block idea is riskier
mathematically, especially around conditioning, yet its trace-only simulator can
kill it cheaply before a production implementation, which makes it the best
campaign bet rather than merely the most ambitious one.
