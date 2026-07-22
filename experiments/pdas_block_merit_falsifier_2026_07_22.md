# Exact simultaneous-block-merit PDAS falsifier — 2026-07-22

**KILL — bounded exact simultaneous-block-merit rescue.** At the certified
round-24 zero-single-improver state, the predeclared 64-edge pool contains
1,548 jointly legal, rank-safe exact pairs and none strictly improves the old
four-component merit. The lexicographically best valid pair is itself worse.
Greedy exact simultaneous augmentation reaches width 8, then every remaining
pool edge either conflicts with the matching or reverses a jointly solved
entering direction. No width-32 or width-64 snapshot exists, so the S0 pass
condition fails and no rescue fresh factor is spent.

This characterization is distinct from both predecessors. The original
full-KKT block-PDAS probe fresh-factored prefixes ordered by violation and
ratio. The exact-single lookahead successor ordered edges by their algebraic
one-exchange post-merit but could not represent interactions among edges. This
probe evaluates the exact simultaneous primal and dual state of a fixed,
bounded interaction path even when every singleton is worse.

No production C, solver API, or test file changed.

## Predeclared policy

The probe first reproduces the exact-single lookahead trajectory to its
round-24 hard stop. It then:

1. Generates the same 512 deterministic forward-valid full-KKT edges.
2. Deduplicates by `(row, entering, leaving_status)` using fixed source ties.
3. Orders all edges by exact scalar old-merit, row, entering column, leaving
   status, source kind, and source index, without requiring a singleton
   improvement, and retains the first 64.
4. Exact-scores all compatible, rank-safe pairs in that fixed pool and selects
   the lexicographically best exact pair even when it is worse than the old
   state.
5. Greedily augments that pair one edge at a time. At every depth it chooses
   the compatible, rank-safe, jointly legal addition with lexicographically
   best exact simultaneous old merit, regardless of intermediate worsening.
6. Snapshots exact sizes 2/4/8/16/32/64 when reached. S0 can pass only if a
   jointly valid width-32 or width-64 snapshot strictly improves the unchanged
   old merit. Only the best eligible wide block may receive fresh-factor
   authority.

There is no truncated prefix, unscored edge, result-dependent retry, alternate
pair seed, or relaxed gate.

## Exact simultaneous algebra

For matched leaving rows `P` and entering columns `Q`, the probe uses cached

```text
D = B^-1 A_Q
H = D[P, :]
```

in the edge-pairing order and solves

```text
H delta = x_B[P] - e_P
x'_P = v_Q + delta
x'_-P = x_B[-P] - D[-P, :] delta

H^T lambda = r_Q
y' = y + B^-T E_P lambda
r' = r - A^T B^-T E_P lambda.
```

The new basis/status arrays then feed the identical old merit

```text
M = (artificial mass,
     max scaled primal violation,
     max scaled dual violation,
     L1 scaled KKT violation).
```

Each subset fails closed unless:

- rows and entering columns are unique;
- the exchange matrix satisfies the existing relative singular-value gate;
- both dense solves meet the fixed
  `4096 * eps_machine * width` normalized backward-residual limit;
- leaving values hit their declared endpoints; and
- every coupled `delta` preserves the legal entering direction, with only the
  existing 64-times-floating-point-epsilon boundary tolerance.

An edge that is forward-valid alone can reverse direction after the coupled
solve. Such a subset is rejected rather than silently flipped.

## Round-24 authority reproduction

The setup exactly reproduces the banked lookahead trajectory:

- accepted widths:
  `32/16/32/8/16/2/1/8/4/4/4/8/1/1/8/4/8/4/4/2/1/2/2`;
- 23 accepted states and 39 fresh proposal-factor attempts;
- terminal merit
  `(0, 21463.265411141812, 17679.010085693157, 587808.2604128214)`;
- round 24 generates 512 edges and has zero strict scalar improvers; and
- expected and reconstructed basis/status SHA-256 both equal
  `58769c2a8748a08a9a8cf064cdd6e8f08ae9d20db3e94bfe0854ca8d2af98211`.

The reconstruction uses a fresh factor and matches the terminal merit
bit-for-bit. This prevents an interaction result on a nearby but different
state from being credited.

## Pair census

All 512 terminal edges are distinct under the fixed exchange identity, and
the pool contains exactly 64. Its pair census is:

| classification | count |
|---|---:|
| possible pool pairs | 2,016 |
| matching-compatible pairs | 1,811 |
| matching conflicts | 205 |
| rank-unsafe compatible pairs | 24 |
| coupled entering-direction reversals | 239 |
| jointly valid pairs | 1,548 |
| strict old-merit-improving valid pairs | **0** |

No pair fails the dense backward-residual or leaving-endpoint gate. Among
direction-invalid pairs, the closest rejected joint-direction margin is
`-1.936302e-14`; the fixed floating-point tolerance is applied before this
classification.

The lex-best valid pair is

```text
r978:j2554:s0:primal978:k20
r920:j2243:s0:primal920:k43
```

with predicted merit

```text
(0,
 21463.26541114174,
 17679.010085693288,
 612745.5592018883).
```

The apparent changes in the maximum primal and dual components are within
their predeclared floating-point comparison tolerances. The decisive L1
component worsens by `24,937.298789`, so the pair is not a strict improvement.
Its condition proxy is 4.51621; primal and dual normalized backward residuals
are respectively zero and `1.262177e-29`.

## Greedy interaction path

Starting from that pair, the exact greedy augmentation path reaches widths
2/3/4/5/6/7/8. It performs 228 exact augmentation evaluations. Across all
depths, 185 examined pool entries conflict with the current matching and 74
otherwise compatible subsets reverse at least one joint entering direction.
After width 8, no compatible jointly legal augmentation remains.

The fixed snapshots are:

| width | decisive component | signed improvement margin | merit conclusion |
|---:|---:|---:|---|
| 2 | L1 KKT | -24,937.298789 | worse |
| 4 | max dual | -54.359588 | worse |
| 8 | max primal | -5.488675 | worse |
| 16 | — | — | not reached |
| 32 | — | — | **not reached** |
| 64 | — | — | **not reached** |

The width-8 snapshot remains jointly legal. Its exchange condition proxy is
`2.940232e4`; maximum primal and dual backward residuals are
`2.060509e-18` and `5.404371e-19`. The maximum leaving-endpoint error is
`1.364242e-12`, and the maximum pre-zeroing new-basic reduced cost is
`9.094947e-13`. These are recorded to distinguish a genuine policy failure
from a bad dense solve.

Because no wide snapshot exists, `eligible_wide_blocks` is empty,
`local_wide_block_pass` is false, `authority` is null, and the rescue spends
zero fresh factors. The probe does not continue a trajectory and does not run
sentinels.

## Cost and unchanged global gates

| component | seconds |
|---|---:|
| exact lookahead-prefix reproduction | 5.833100 |
| terminal reconstruction factor | 0.012412 |
| rescue candidate generation | 0.136861 |
| rescue scalar pool | 0.073910 |
| all pool-pair scoring | 0.507750 |
| greedy augmentation | 0.071424 |
| rescue authority factor | 0 |
| whole probe | 6.963315 |

The bounded rescue work alone takes 0.789946s, 1.76 times the unchanged
0.448351702s characterization gate. The whole diagnostic is 15.53 times that
gate. Python diagnostic timing is not promoted to a production lower bound;
the geometric failure is independently decisive.

There are 41 actual fresh factors including the baseline initial factor and
the terminal reconstruction factor. The reference-factor-only floor is
0.042392s. Proposal-factor attempts remain 39 because algebraic scores and the
reconstruction factor are not policy proposals.

| unchanged gate | result | verdict |
|---|---:|---|
| accepted exchanges <= 256 | 23 | PASS |
| proposal fresh-factor attempts <= 384 | 39 | PASS |
| median accepted width >= 18 | 4 | **FAIL** |
| no repeated state | none | PASS |
| original-space `eps=2e-5` certificate | bound violation 21,460.6764 | **FAIL** |
| complete projected cost <= 0.448351702s | infinite without a certificate | **FAIL** |

The unchanged terminal certificate has equality residual `2.344865e-11` but
bound violation `21,460.6764`; it is not optimal or certificate-backed.

## Scoped verdict and reopening condition

**Scoped verdict:** `KILL_EXACT_SIMULTANEOUS_BLOCK_MERIT_S0` for the first-64
exact-scalar pool, exhaustive compatible-pair census within that pool,
lex-best pair seed, and single-path greedy exact simultaneous augmentation.

This does not kill all simultaneous-block active-set policies. It does not
enumerate interactions outside the first-64 pool, alternate pair seeds,
non-greedy or non-nested higher-order subsets, or paths that deliberately
accept an old-merit worsening into a new state. Reopening requires a
predeclared bounded construction that differs in one of those specific ways
and still preserves coupled entering direction, dense residual, fresh-state,
cycle, width, cost, and original-certificate gates. Reordering the same greedy
path is exhausted.

## Artifacts and verification

- Probe: `experiments/pdas_block_merit_probe.py`
- Probe SHA-256:
  `516c2626cdd8d3743c9124beec74aa0ea8bbdaf5e53f7fef18ed220c0f040f64`
- Raw result: `/tmp/pdas-block-merit-falsifier/results.json`
- Raw SHA-256:
  `b7140e504597593ca55744073f7b06c068119d36bdd339d0a9a99638749fdf1d`
- Raw directory mode: `0700`; result mode: `0600`; result size: 34,964
  bytes.

Reproduction:

```bash
UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
uv run python -m experiments.pdas_block_merit_probe
```

Verification:

```bash
UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run ruff check experiments/pdas_block_merit_probe.py

UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
uv run ruff format --check experiments/pdas_block_merit_probe.py

UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 UV_OFFLINE=1 UV_CACHE_DIR=/tmp/uv-cache \
PYTHONDONTWRITEBYTECODE=1 \
uv run python -m py_compile experiments/pdas_block_merit_probe.py
```

No Git, network, package, or external solver-source operation was used.
