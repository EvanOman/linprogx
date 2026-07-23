# Exact single-exchange lookahead block-PDAS falsifier — 2026-07-22

**KILL — fixed lookahead-selected full-KKT block PDAS.** The standalone
authority run accepts 23 exchanges but stops without a certificate after 39
fresh-factor attempts. Its median accepted width is 4, below the predeclared
18-column gate. At the hard stop, all 512 generated forward-valid scalar
edges have been scored and none is a tolerance-aware strict improver of the
unchanged lexicographic merit. The three sentinel fixtures therefore do not
run.

This is a materially different selection policy from both earlier audits:

- The old fixed-face rank-k audit evaluated limited simultaneous exchanges on
  native simplex checkpoints. This probe instead evolves a complete
  primal/dual active-set state using full-KKT primal and dual violations.
- The repaired greedy block-PDAS probe ordered candidates by violation rank,
  ratio, and pivot magnitude. It accepted widths 60/8/2/2, then stalled even
  though individually improving edges existed outside its prefixes.
- This successor scores **every generated edge's exact algebraic
  single-exchange post-state**, sorts strict improvers by predicted post-merit,
  and only then constructs a rank-safe row/column matching. Its scalar
  fallback is the best scored edge, never Bland and never an unscored edge.

No fixture name, shape, checkpoint, or observed result participates in the
policy. The policy and all gates were fixed before the authority run.

## Fixed policy

The model, scaling, crash, starting endpoint assignment, candidate generation,
and certificate logic are imported unchanged from the repaired
`block_pdas_probe.py`. In particular:

- the ten-pass infinity-norm plus one L2 Ruiz scaling is unchanged;
- every nonbasic structural variable is at a true legal bound, or zero only
  when genuinely free;
- the legal forward endpoint test tolerates only 64 times floating-point
  epsilon at the step boundary;
- the deterministic generator examines the top 64 full-KKT violations and
  emits at most eight forward-valid edges per source;
- the Python crash must match both the native one-pivot basis and bound-status
  authority; and
- completion requires the unchanged original-space `eps=2e-5` residual and
  bound certificate.

For an edge replacing the basic column at row `p` with nonbasic column `j`,
let `v_j` be `j`'s current legal endpoint, `e_i` the selected legal endpoint
of the leaving column, and

```text
d = B^-1 a_j
t = (x_B[p] - e_i) / d[p]
x'_j = v_j + t
x'_B[not p] = x_B[not p] - d[not p] t

h = B^-T e_p
alpha = A^T h
theta = r_j / alpha_j
y' = y + theta h
r' = r - theta alpha
```

The implementation batches the exact `d`, `h`, and `alpha` solves for speed,
but applies precisely these single-exchange equations to each edge. It assigns
the predicted new basis and statuses and recomputes the identical merit

```text
M = (artificial mass,
     max scaled primal violation,
     max scaled dual violation,
     L1 scaled KKT violation).
```

An edge survives only when `M'` is a strict tolerance-aware lexicographic
decrease. The component tolerance is fixed at
`256 * eps_machine * (1 + |before| + |after|)`. Survivors sort by predicted
post-merit, then row, column, leaving status, and source ties. A greedy
row/column matching retains only exchanges whose exact exchange minor remains
full rank. Fresh factors are tried only for exact power-of-two prefixes
64/32/16/8/4/2; a requested width is skipped when the matching is smaller.
No truncated non-power-of-two batch is legal. If none accepts, the best
predicted scalar is fresh-factored;
its full predicted state must match fresh LU under a predeclared
growth-and-scale-derived floating-point tolerance or the probe fails closed.

Only fresh-factor attempts count against the 384-attempt cap. Candidate
generation, exact single-edge scoring, rank checking, and fresh factor time
are recorded separately. Accepted states must strictly reduce `M` under the
same tolerance-aware comparator and must not repeat a prior basis/status key.

## Starting authority

The reduced scaled problem has 1,525 rows, 3,868 structural columns, and
23,274 structural nonzeros. The triangular crash has 1,462 structural and 63
identity-artificial basic columns, does not fall back to the identity, and has
a 6.45535 diagonal-growth proxy. Its warm one-pivot replay matches both the
native cold basis and native bound statuses.

The initial merit is

```text
(5587.424504335054,
 19590.98246931742,
 137.5780516868135,
 82404.7460495044).
```

## Authority trace

The run performs 24 scoring rounds. It generates 12,212 forward-valid edges
in total and classifies 1,627 as predicted strict improvers. It attempts 35
exact power-of-two matching prefixes and four best-predicted scalar fallbacks,
for 39 fresh proposal factors total. Twenty-three proposals accept and 16
reject. The only attempted batch widths are 32, 16, 8, 4, and 2; no width-64
matching is available on this trace.

The first four accepted widths are 32/16/32/8. The first width-32 exchange
legally removes all artificial mass, which is the leading merit component,
while increasing the other three components; this is visible rather than
hidden by a scalar score:

| state | artificial mass | max primal | max dual | L1 KKT |
|---|---:|---:|---:|---:|
| initial | 5,587.4245 | 19,590.9825 | 137.5781 | 82,404.7460 |
| after width 32 | 0 | 14,703,721.4494 | 98,585.9876 | 106,429,039.6000 |
| final | 0 | 21,463.2654 | 17,679.0101 | 587,808.2604 |

The policy escapes the repaired greedy-prefix policy's four-state stall and
does substantial recovery work, but accepted widths collapse. Of the 23
accepted exchanges, only two have width at least 18, four are scalar, and the
median is 4. The full accepted-width sequence is stored in the raw result.

Round 24 starts at the final merit above, generates 512 legal forward-valid
edges, scores every one, finds zero predicted strict improvers, and forms an
empty matching. This is the declared hard stop. There are 24 unique accepted
states including the initial state and no repeated proposal state.

## Prediction versus fresh-factor authority

All four scalar fallbacks pass the mandatory full-state check and are
accepted by fresh authority. Across those checks, the worst maximum absolute
differences are:

| quantity | worst maximum absolute difference |
|---|---:|
| basic primal values | 1.637090e-10 |
| dual values | 4.802132e-10 |
| reduced costs | 4.802132e-10 |
| four-component merit | 1.513399e-9 |

Every basis and status vector matches exactly. Each result record also stores
the relative and absolute tolerances derived from the old/new factor growth
and the compared vector's scale. No scalar mismatch or fail-closed exception
occurs.

For matching prefixes, the raw record contains every constituent edge's
single-exchange prediction, the best and worst selected predicted merits, and
the complete fresh-factor actual merit. The policy does not pretend that
individually scored edges supply an exact simultaneous block prediction;
fresh LU is the block authority.

## Factor and cost evidence

Fresh proposal factors contain 9,851--11,110 `L+U` nonzeros (median 10,705).
On the final orchestrator rerun, observed one-thread SciPy factor times are
1.039--1.601ms (median 1.154ms), and
the largest accepted or rejected factor-growth proxy is 2.287e4, below the
probe's fixed `1e12` fail-closed threshold.

| measured component | seconds |
|---|---:|
| candidate generation | 3.212917 |
| exact single-exchange scoring | 1.857172 |
| rank-safe matching | 0.095205 |
| fresh factors, including initial | 0.047867 |
| complete standalone trace | 5.635893 |

The board control is 0.560439628s, the board-flip target is 0.461234528s, and
the stricter 20% characterization target is 0.448351702s. Scoring alone is
4.14 times the characterization target; the incomplete Python trace is 12.57
times it. The predeclared reference-factor-only floor for 40 factors is
0.041358s, but that favorable floor omits candidate generation and scoring and
still ends at an uncertified state. Since there is no complete solution, the
complete projected cost is infinite. Partial progress is not projected onto
the board.

## Gates

| gate | result | verdict |
|---|---:|---|
| accepted exchanges <= 256 | 23 | PASS |
| fresh-factor attempts <= 384 | 39 | PASS |
| median accepted width >= 18 | 4 | **FAIL** |
| no repeated state | 24 unique states; none repeated | PASS |
| original-space `eps=2e-5` certificate | bound violation 21,460.6764 | **FAIL** |
| complete projected cost <= 0.448351702s | infinite without a certified solution | **FAIL** |

The final reconstructed original-space equality residual is
`2.344865e-11`, but the bound violation is `21,460.6764`; KKT optimality and
the complete certificate are false. This is not a near-certificate.

## Scoped verdict and reopening condition

**Scoped verdict:** `KILL_LOOKAHEAD_BLOCK_PDAS_CHARACTERIZATION` for the fixed
forward-valid candidate generator, exact single-exchange predicted-merit
ordering, greedy rank-safe matching, power-of-two prefix ladder, and best-
predicted-scalar fallback described above.

This result does not kill every block principal-pivot or active-set method. In
particular, it does not test an exact simultaneous-block merit model, a
globalization that can accept temporary lexicographic worsening, candidate
edges outside the unchanged top-64/eight-per-source generator, or a different
merit. Reopening this branch requires a predeclared global policy with a
concrete reason it can cross the observed zero-single-improver state while
still preserving legal endpoints, cycle protection, fresh-factor authority,
the median-width gate, and the unchanged original-space certificate. Merely
reordering the same exact single-edge predictions is exhausted at round 24.

No production C, solver API, or test file changed. A characterization pass
would only have earned the three predeclared dual-simplex sentinels; because
greenbea fails, `woodw`, `stocfor3`, and `80bau3b` were correctly not run.

## Artifact and verification

- Probe: `experiments/pdas_lookahead_probe.py`
- Probe SHA-256:
  `3dcff29eb8155c3c26f358361635a03f6a098d811d7a0866588abfd3b49acf26`
- Raw result: `/tmp/pdas-lookahead-falsifier/results.json`
- Raw SHA-256:
  `9048b95da27ae1d5439a8d7d8ca1e8fafdd86530999ac3298e11189312eb7bd7`
- Directory mode: `0700`; result mode: `0600`; result size: 244,010 bytes.

Reproduction:

```bash
UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
uv run python -m experiments.pdas_lookahead_probe
```

Verification:

```bash
UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run ruff check experiments/pdas_lookahead_probe.py

UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run ruff format --check experiments/pdas_lookahead_probe.py

UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
PYTHONDONTWRITEBYTECODE=1 \
uv run python -m py_compile experiments/pdas_lookahead_probe.py
```

The work used no Git operation, package or network operation, and no external
solver source. It is a Python diagnostic only.
