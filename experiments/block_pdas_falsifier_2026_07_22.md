# Full-KKT block principal-pivot falsifier (2026-07-22)

## Verdict

**KILL — fixed greedy-prefix full-KKT block PDAS.** The standalone
characterization does not converge on greenbea. It accepts only four
exchanges, of widths 60, 8, 2, and 2 (median **5**, below the predeclared
18-column gate), then reaches a non-KKT state where none of the policy's
rank-safe matching prefixes or its deterministic scalar criss-cross fallback
strictly decreases the exact recomputed merit.

The final point has maximum original-space equality residual `1.182e-11`, but
maximum bound violation **62,263.4827**. It is not a solution and does not
receive `optimal` status. Complete projected solution cost is therefore
infinite, not a favorable partial-trajectory timing. The greenbea gate fails,
so the predeclared woodw, stocfor3, and 80bau3b sentinels were correctly not
run.

This kills the concrete globally fixed selection policy below. It does not
prove that every criss-cross method, complementarity formulation, candidate
ordering, or merit globalization is impossible.

## Why this is a new class

The earlier rank-k audit imported four dual-simplex checkpoints, kept the
ordinary dual-feasible face semantics, and tested simultaneous changed-endpoint
exchanges only for `k <= 4`. This probe instead starts at the native crash basis,
permits both primal and dual infeasibility, takes blocks as wide as 64, assigns
every evicted basic to an exact legal endpoint, and reconstructs the full KKT
state after every proposal with a fresh factorization. Its accepted width-60
move lies outside the earlier experiment's scope.

The result shows that broader full-KKT freedom does not automatically rescue
the earlier face-exchange failure. Under this ordering, eliminating artificials
makes the rest of the KKT state much worse, and later greedy matching prefixes
do not recover. It does not establish a local minimum of the merit itself.

## Exact problem and native start

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- Linprogx-presolved shape: 1,525 rows x 3,868 structural columns x 23,274
  nonzeros.
- Scaling: the production ten-pass inf-norm plus one L2 Ruiz policy.
- Augmentation: 1,525 identity artificial columns with cost zero and bounds
  `[0,0]`.
- True structural bounds only: no Big-M endpoint is introduced. A one-sided
  nonbasic is placed at its finite legal endpoint; a genuinely free nonbasic is
  placed at zero.
- Crash: 1,462 structural plus 63 artificial basics. LU diagonal-growth proxy
  `6.45535`; no identity fallback.
- Native identity check: importing the reconstructed basis alone and executing
  one native pivot reproduces the cold one-pivot basis and bound-status vector
  exactly; either mismatch fails closed.

The one-pivot replay is the iteration-zero authority check. The experiment's
subsequent endpoint assignment intentionally differs from native dual simplex:
it refuses artificial bounds, so its state may be dual-infeasible as required
for a full-KKT principal-pivot probe.

## Fixed global policy

For a basis `B`, the probe fresh-factorizes `B` and recomputes

```text
x_B = B^-1 (b - A_N x_N)
y   = B^-T c_B
r   = c - A^T y.
```

For basic row `i`, scaled primal violation is

```text
P_i = max(lo_i - x_i, x_i - hi_i, 0)
      / (1 + max(|finite lo_i|, |finite hi_i|)).
```

For a nonbasic at lower, upper, or free status, respectively,

```text
D_j = max(0, -r_j)/(1+|c_j|),
      max(0,  r_j)/(1+|c_j|),
      |r_j|/(1+|c_j|).
```

Every accepted proposal must strictly decrease the freshly recomputed tuple

```text
M = (sum |x_artificial|,
     max P_i,
     max D_j,
     sum P_i + sum D_j)
```

under ordinary lexicographic ordering. There is no trajectory-derived
tolerance or fixture-specific override.

At each state, the policy takes the top 64 primal and dual violations by
normalized magnitude. A primal source row uses its exact pivot row
`a_j^T B^-T e_i`; a dual source column uses its exact FTRAN
`B^-1 a_j`. Edge direction must move the source violation toward a true legal
bound. For a dual-source edge, if entering motion is `d`, basic motion is
`mu=-q_i d`; its selected endpoint `e` must also be forward-reachable:

```text
t = (e - x_B_i) / mu >= 0.
```

The implementation rejects nonfinite steps and negative steps beyond one fixed
FP allowance, `64*epsilon*(1+|t|)`; an accepted FP-scale negative is clamped to
zero. The same test applies to scalar criss-cross. Candidate order is
deterministic: source severity/rank, then nonnegative step or the primal-source
ratio, pivot magnitude, row, and column.

The selected edges form a row/entering-column matching. Rank safety is checked
before proposal using the basis-exchange minor

```text
Q = [B^-1 a_j]_(replaced rows, entering columns).
```

An edge is retained only while this minor remains numerically full rank. The
batch ladder is globally fixed at `64,32,16,8,4,2,1`; rejection halves the
batch. Width one is a separate least-global-variable-index Bland-style
criss-cross pivot, not Dantzig dual simplex. Singular, ill-conditioned,
nonfinite, or illegal-endpoint proposals fail closed.

## Trace result

Initial merit:

```text
(5587.4245043, 19590.9824693, 137.5780517, 82404.7460495)
```

Accepted proposals:

| attempt | width | artificial mass | max primal | max dual | L1 KKT |
|---:|---:|---:|---:|---:|---:|
| initial | — | 5,587.4245 | 19,590.9825 | 137.5781 | 82,404.7460 |
| 1 | 60 | 0 | 186,470.6581 | 7,129.1168 | 1,992,270.2310 |
| 5 | 8 | 0 | 105,800.9627 | 7,179.5124 | 802,882.0888 |
| 11 | 2 | 0 | 46,986.9087 | 7,179.5124 | 606,366.3565 |
| 17 | 2 | 0 | 32,736.9766 | 7,179.5124 | 516,710.8848 |

The first exchange is legal under the predeclared lexicographic rule because
it reduces the leading artificial-mass component to zero. It simultaneously
increases maximum primal violation by 9.52x, maximum dual violation by 51.8x,
and L1 KKT violation by 24.2x. This is not hidden by a scalarized score: all
four components are recorded before acceptance.

Attempts 18--24 then test the selected rank-safe matching prefixes of widths
52, 32, 16, 8, 4, and 2 followed by the exact scalar criss-cross edge. None
strictly decreases `M`. The run stops after 24 policy proposals and four
accepted exchanges; there is no repeated state.

This is a greedy-prefix selection dead end, not a merit local minimum. A
read-only post-run audit found individually improving forward-valid edges that
the prefix ladder does not propose: 87 of 512 generated single edges, including
12 of the 52 edges in the final rank-safe matching. For example, matching edge
12 lowers maximum primal violation from 32,736.9766 to 32,664.3298, and edge 39
lowers it to 30,353.6911. These observations are not retroactively substituted
into the fixed policy or its artifact; they define the verdict's scope.

Fresh proposal factors contain 9,553--10,262 `L+U` nonzeros. On the final
orchestrator rerun, observed SciPy factor times are 1.041--1.968ms (median
1.492ms). The whole standalone algorithm takes 1.374834s, above the 0.448352s
characterization target, but
Python diagnostic wall is not used as a portable C-runtime lower bound. The
25 fresh factors alone project to only 0.025848s under the predeclared
1.03394ms reference; that floor is likewise not the kill. Nonconvergence and
the certificate failure are decisive.

## Predeclared gates

| gate | result | verdict |
|---|---:|---|
| accepted exchanges <= 256 | 4 | PASS |
| attempted exchanges <= 384 | 24 | PASS |
| median accepted width >= 18 | 5 | **FAIL** |
| no repeated state | 5 unique states; none repeated | PASS |
| original-space `eps=2e-5` certificate | bound violation 62,263.4827 | **FAIL** |
| complete projected cost <= 0.448351702s | infinite without a certified solution | **FAIL** |

The board control is 0.560439628s, the board-flip target is 0.461234528s, and
the stricter 20% characterization target is 0.448351702s. Partial progress is
not projected onto any of them.

## Scope and reopening condition

**Scoped verdict:** `KILL_BLOCK_PDAS_CHARACTERIZATION` for the exact legal-
endpoint, greedy rank-safe-matching-prefix, lexicographic-merit policy above.

A successor may change selection, ordering, or globalization, but it must
predeclare a global rule and demonstrate before a C implementation how it both
ejects artificials and recovers from the observed one-to-two-order-of-magnitude
growth in the remaining KKT components. Legal endpoints, forward-reachable
steps, fresh-state authority, cycle protection, and the unchanged `eps=2e-5`
certificate remain mandatory.

No production C, solver API, or test file changed. A characterization pass
would only have earned sentinel testing and production characterization; it
would not itself have been a ship decision.

## Artifact and verification

- Probe: `experiments/block_pdas_probe.py`
- Probe SHA-256:
  `ee6376250d62a6fb34569d4a5e26dba13263e40723e748dc317f380039b8ed5b`
- Raw result: `/tmp/block-pdas-falsifier/results.json`
- Raw SHA-256:
  `af65ef45e9c027ddebde3761397d4e4670bb4d36d3e1d69d9a0e8e14346fb03a`
- Directory mode: `0700`; result mode: `0600`; result size: 191,546 bytes.

Reproduction:

```bash
UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
uv run python -m experiments.block_pdas_probe
```

The probe performs no network access and reads no external solver source.
