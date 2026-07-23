# Sparse revised-primal falsifier: greenbea (2026-07-22)

## Verdict

**KILLED.** No measured globally applicable sparse revised-primal route clears
the campaign's 20% charged whole-wall gate, so no implementation brief or
production solver change is warranted.

The earlier C6 result killed only linprogx's dense tableau implementation. This
follow-up tests the remaining sparse-revised-primal claim directly:

- the current cold sparse dual path is **4,399 pivots, median 0.560440 s** on
  the measurement host;
- the default exact public-API primal route is **13,809 pivots, median
  1.431130 s**;
- a global public-option census finds a much better primal trajectory under
  scale strategy 4, but it is still **7,427 pivots**; and
- the best observed scale-4 primal wall is **0.606176 s**, already slower than
  the current cold dual control and well above the 20% target of **0.448352 s**.

The result does not project another solver's wall onto linprogx. The decisive
economics use linprogx's current sparse factor and phase timers, plus basis sets
sampled from the public primal trajectory. An intentionally favorable charged
projection is **0.537669 s**, only **4.06%** below the cold control and above
both the 20% probe target and the **0.461235 s** wall implied by the board's
17.7013% gap. It omits ratio testing, primal updates, refactorization, Phase-I
construction, pricing/edge-weight work, and the final certificate.

## Scope and rules

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- linprogx presolved shape: **1,525 rows x 3,868 columns x 23,274 nonzeros**.
- Fixed acceptance epsilon: `2e-5`.
- No per-problem production tuning. Every ablation is a global public option or
  a read-only characterization.
- No solver source was read. HiGHS 1.14.0 was used only through the public
  `highspy` model, option, run, solution, information, and basis APIs.
- No network access or Git operation was used. `UV_OFFLINE=1` mechanically
  constrained every `uv` invocation; highspy came from the existing local uv
  cache in an ephemeral environment.
- No target-repository source was changed. The probe drivers and generated
  evidence live under `/tmp`; this report is the only repository file added.

## Why a sparse revised-primal pivot is not a cheap role swap

For a primal-feasible bounded-variable basis `B`, an exact ordinary revised-
primal pivot has the following state transitions:

1. Select a nonbasic entering column `q` with an improving exact reduced cost.
2. FTRAN the entering column:

   ```text
   d = B^-1 a_q
   ```

3. Ratio-test `d` against the basic bounds to select leaving basis position
   `p`, or perform a finite bound flip.
4. BTRAN the leaving row:

   ```text
   u = B^-T e_p
   ```

5. Form the tableau row `alpha = A^T u` and update the reduced costs. In the
   usual sign convention:

   ```text
   r' = r - (r_q / d_p) alpha
   ```

6. Update the basic primal values, bound statuses, basis membership, and the
   sparse factor.

The BTRAN-equivalent row is necessary to keep reduced costs exact after the
basis replacement. Recomputing the dual vector from `B'^T y' = c_B'` performs
the same class of solve. Restricting the row to a multiple-pricing candidate
set can reduce the `A^T u` scatter, but it does not remove FTRAN, BTRAN, the
basic ratio test, factor updates, numerical refactorization, Phase-I basis
construction, or final full pricing and certificate reconstruction.

The current `LUContext` already supplies the relevant sparse FTRAN, sparse
BTRAN, Forrest-Tomlin update, refactorization, and exact exit reconstruction.
The question is therefore economic before it is architectural.

## Current cold control

Seven foreground cold solves used the current presolve and sparse dual path,
with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`LINPROGX_DS_SOLVE_SLICE=1`, and diagnostic basis export. All seven returned
the same certified optimum and 4,399 pivots.

| measure | result |
|---|---:|
| median whole wall | **560,439.628 us** |
| pivots | **4,399** |
| reduced objective | -72,557,668.26492292 |
| maximum reduced equality residual | 1.026e-7 |
| refactorizations | 33 |
| final structural / logical basics | 1,487 / 38 |
| final basis matrix nonzeros | 7,734 |
| final factor `nnz(L)+nnz(U)` | 9,716 |

The individual walls were 0.547387, 0.560440, 0.572039, 0.568994,
0.566643, 0.526480, and 0.524882 seconds. The median-of-seven is used for
all target arithmetic:

```text
20% probe target       = 0.560439628 * 0.80      = 0.448351702 s
17.7013% board target  = 0.560439628 * 0.822987  = 0.461234528 s
```

## Public-API primal census

### Default route

Seven one-thread, presolve-off, public-API primal solves were exact and
iteration-identical:

```text
iterations: 13,809 on all seven runs
median wall: 1.431130 s
walls: 1.427525, 1.425395, 1.432138, 1.486780,
       1.412287, 1.442355, 1.431130 s
status: Optimal on every run
```

The full runtime log reports a primal start with 1,230 Phase-I primal
infeasibilities and 3,024 dual infeasibilities. At completion it reports zero
primal and dual infeasibilities and P-D objective error `1.027e-16`.

### One-axis global ablations

Each row changes only the named public numeric option. These are behavioral
ablations; no inference about the other solver's implementation is made.

| family | values | exact pivot results |
|---|---|---:|
| primal edge weight | choose, Dantzig, Devex, steepest | 13,809; 37,699; 13,809; 11,962 |
| crash strategy | 0 through 9 | **13,809 for all ten** |
| price strategy | 0, 1, 2, 3 | 13,874; 15,119; 15,119; 13,809 |
| scale strategy | 0, 1, 2, 3, 4 | 10,943; 13,809; 13,809; 13,809; **7,427** |

Every run returned `Optimal`, the reduced objective agreed at
`-72,557,668.264923`, maximum reduced equality residual was at most
`6.138e-8`, and maximum bound violation was at most `7.674e-13`.

### Scale-4 joint edge x price sweep

Because scale strategy 4 materially changes the trajectory, all 16 public
edge-weight x price combinations were solved with it. Crash was not crossed
again because all ten crash values were pivot-identical in the one-axis sweep.

| edge policy | price policies | best / common pivots | best observed wall |
|---|---|---:|---:|
| choose | 0, 1, 2, 3 | **7,427 for all** | **0.606176 s** |
| Dantzig | 0 | 10,558 | 1.000544 s |
| Dantzig | 1, 2, 3 | 12,178 | 0.671673 s |
| Devex | 0, 1, 2, 3 | **7,427 for all** | 0.624563 s |
| steepest | 0, 1, 2, 3 | 8,189 for all | 2.861275 s |

All 16 combinations were certificate-clean at the same objective and maximum
reduced equality residual `6.138e-8`. The strongest available primal
trajectory is therefore 7,427 pivots, not 13,809. The falsifier proceeds on
that stronger result.

## Scale-4 basis trace through linprogx's factor

Public `getBasis()` snapshots were taken at iteration limits 1,000, 3,000,
5,000, and 7,427 on the strongest scale-4 route. Each basis matrix was built
from its structural basics plus logical identity columns, then passed to
linprogx's own `lu_stats_test` and `lu_solve_test` hooks.

For FTRAN support characterization, 96 actual nonbasic structural columns were
sampled deterministically. For BTRAN, 96 deterministic unit-row right-hand
sides were used. Output density is the share of values with magnitude above
`1e-12`. The bulk hook timings include Python list conversion and are not used
in the whole-wall projection; factor fill and output density are the relevant
trajectory evidence.

| checkpoint | structural / logical basics | basis nnz | factor nnz | FTRAN density | BTRAN density | public wall to checkpoint |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 923 / 602 | 5,404 | **5,751** | 3.229% | 1.482% | 0.029748 s |
| 3,000 | 1,445 / 80 | 7,672 | **9,189** | 21.728% | 16.481% | 0.200675 s |
| 5,000 | 1,513 / 12 | 8,281 | **11,107** | 36.038% | 38.437% | 0.445478 s |
| 7,427 | 1,518 / 7 | 8,010 | **9,873** | 15.755% | 15.197% | 0.648717 s |
| current dual endpoint | 1,487 / 38 | 7,734 | **9,716** | 15.146% | 17.980% | 0.560440 s full solve |

The primal basis is genuinely cheap only at the beginning. By iteration 3,000
its factor fill is already 94.6% of the dual endpoint; at iteration 5,000 it is
14.3% higher and both solve outputs are more than twice as dense. The final
primal factor is also slightly denser than the dual endpoint. There is no
trajectory-wide sparse-basis discount capable of funding 3,028 extra pivots.

The default 13,809-pivot route is worse: sampled factor fill rises from 8,074
at iteration 1,000 to 12,707 at 3,000, peaks at 14,573 at 6,000, and finishes
at 12,829.

## Mandatory-kernel economics

### Exact full-pricing current-kernel floor

The nested solve counters separate one mandatory sparse FTRAN and one mandatory
sparse BTRAN per committed pivot from extra dense solves. Dividing current
measured work by its actual call count gives:

| mandatory exact-pivot component | current measured us / pivot |
|---|---:|
| sparse entering-column FTRAN | 22.9653 |
| sparse leaving-row BTRAN | 15.4651 |
| exact tableau-row formation | 26.9095 |
| exact reduced-cost update | 12.5303 |
| sparse factor update | 8.8124 |
| **total before ratio/refactor/setup/certificate** | **86.6826** |

Applying only those current-kernel costs to the strongest 7,427-pivot route:

```text
7,427 * 86.6826 us = 643,791.630 us
```

That is **14.87% slower** than the current 560,439.628 us control and 43.6%
above the 448,351.702 us probe target. It charges none of:

- the primal ratio test over 1,525 basic variables;
- basic primal-value and bound-status updates;
- refactorizations;
- Phase-I crash/basis construction;
- pricing or edge-weight maintenance; or
- final exact `x_B`, `y`, reduced-cost, original-space residual, and gap checks.

### Deliberately favorable sparsity discount

To avoid assuming that the early sparse primal factors cost the same as the
current dual factor, the first 3,000 pivots receive an intentionally excessive
discount. The smallest observed primal factor has only
`5,751 / 9,716 = 0.591910` of the dual endpoint fill. The projection grants
that best ratio to **every one of the first 3,000 pivots and to every mandatory
component**, including tableau-row and reduced-cost work that does not shrink
with factor fill. The remaining 4,427 pivots are charged at current rates with
no penalty for the denser iteration-5,000 factor.

```text
first 3,000  = 3,000 * 0.591910 * 86.6826 us
remaining    = 4,427 * 1.000000 * 86.6826 us
total        = 537,668.795 us
improvement  = (560,439.628 - 537,668.795) / 560,439.628
             = 4.063%
```

Even this over-generous trace-based floor is above the board target
461,234.528 us and the 20% probe target 448,351.702 us, before any omitted
work is charged.

### Partial-pricing escape does not fund the route

As a separate optimistic check, delete tableau-row formation and the full
reduced-cost update entirely, as if exact candidate maintenance were free.
Keep only FTRAN, BTRAN, factor updates, the current refactor amortization, and
scale the current ratio-test time linearly from 3,868 structural columns to
1,525 basic positions:

| retained component | optimistic us / pivot |
|---|---:|
| FTRAN | 22.9653 |
| BTRAN | 15.4651 |
| factor update | 8.8124 |
| refactor amortization | 9.5668 |
| ratio, linearly scaled to 1,525 / 3,868 | 7.2108 |
| **total** | **64.0204** |

```text
7,427 * 64.0204 us = 475,479.680 us
```

This impossible-free-pricing route improves whole wall by at most 15.16%,
still below the 20% probe gate and the 17.7013% board requirement. Phase-I
construction, candidate pricing, primal updates, and the final certificate
remain uncharged. Thus multiple pricing does not rescue the measured 7,427-
pivot trajectory.

These are current-kernel, trace-backed economic floors, not a theorem about
every imaginable primal algorithm. A genuinely different batched algorithm
that proves a much shorter trajectory would be new science and would require a
new characterization. No such algorithm or trace exists here.

## Construction and certificate accounting

A globally applicable primal route must begin with a primal-feasible basis.
greenbea's natural public primal start reports 1,230 Phase-I primal
infeasibilities. The scale-4 trace spends 0.200675 s reaching pivot 3,000; that
work is included in the observed public wall but omitted from the favorable
linprogx projections above.

The known alternative constructors do not help:

- current IPM needs about 0.630 s to reach its best near-feasible late point,
  already beyond the whole 20% target, and still lacks a dual certificate;
- IPM-derived crash bases previously take at least 4,766 certified dual pivots
  after charged IPM/crossover work;
- the exact homogeneous auxiliary costs about 0.145-0.215 s before the main
  simplex run and exhibits the established density tradeoff; and
- active-set prediction misclassifies roughly 680 degenerate-basic columns and
  produces infeasible reduced primal problems.

No construction cost is silently treated as free in the verdict: the primary
projection already fails without charging one.

Every completed public primal ablation returned `Optimal`. Across the one-axis
sweep:

```text
reduced objective                 -72,557,668.264923
maximum reduced equality residual  6.138e-8
maximum bound violation             7.674e-13
```

The current dual result is `-72,557,668.26492292` with residual `1.026e-7`.
The public full-run log reports zero terminal primal and dual infeasibilities
and P-D objective error `1.027e-16`. These are safely inside fixed `eps=2e-5`.
No approximate point, iteration-limit checkpoint, objective match alone, or
uncertified route is counted as success.

## Decision

Stop before implementation. Sparse revised primal does not clear the 20%
charged whole-wall gate:

1. the best global public trajectory is still 7,427 pivots, 68.8% longer than
   the current 4,399-pivot dual trajectory;
2. its factors lose the early sparsity advantage by iteration 3,000;
3. current exact-pivot mandatory kernels project to 0.644 s;
4. an extreme early-sparsity discount still projects to 0.538 s; and
5. even free partial pricing projects to 0.475 s before construction and
   certificate work.

No characterization-first implementation brief follows because the required
economic precondition is false.

## Evidence artifacts and SHA-256

```text
0672a148c819e63e72a0941e472aaf215cc8a33ac2396db509fb17d08a489775  /tmp/sparse-primal-falsifier/results.json
29067342c6741ffa90196fe2696df6193b6b25faaf99bfdc393ba52fed361af5  /tmp/sparse-primal-falsifier/variant_sweep.json
01aeb3a6c856fcb29aa3b2675b88d388d2b92ed21b0a49dbb992b05b66427836  /tmp/sparse-primal-falsifier/joint_scale4_sweep.json
e145d4e30d7bead623a6801194b52358f5296e2245ddcec18ea33500113678dc  /tmp/sparse-primal-falsifier/scale4_trace.json
66d3ff8224cb6ab2611065158baa8efdca68c619d475f4162db3f8280d5e0a6c  /tmp/sparse_primal_falsifier.py
c7b1ef48c82e9efa60596056f6bfc3cb68fbe3a492dc45c2e79804ad91ba8c39  /tmp/sparse_primal_variant_sweep.py
dcb353c8c06037c16d28c2c20191b4751d34af616587f91b8c75c4a8950bb237  /tmp/sparse_primal_joint_sweep.py
09112ea49bae8f365e8f06bccde102b8206a3d253adbe9747581e270fa7434a6  /tmp/sparse_primal_scale4_trace.py
```

The individual public runtime logs are under
`/tmp/sparse-primal-falsifier/primal_{checkpoint,full}_*.log`.

## Exact command audit

All commands ran from `/home/evan/dev/linprogx-perf-worktree`.

Cached/offline public oracle availability:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run --with 'highspy==1.14.0' python - <<'PY'
import highspy
print(highspy.Highs().version())
PY
```

Authoritative cold control, default primal trace, and sampled-basis factor
measurements:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run --with 'highspy==1.14.0' \
  python /tmp/sparse_primal_falsifier.py
```

One-axis global public-option census:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run --with 'highspy==1.14.0' \
  python /tmp/sparse_primal_variant_sweep.py
```

Scale-4 edge x price census:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run --with 'highspy==1.14.0' \
  python /tmp/sparse_primal_joint_sweep.py
```

Scale-4 basis checkpoints through linprogx's current LU hooks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:/tmp \
  UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv run --with 'highspy==1.14.0' \
  python /tmp/sparse_primal_scale4_trace.py
```

Network-token and process audit:

```bash
rg -n -i \
  'curl|wget|git clone|github|requests|urllib|socket|https?://' \
  /tmp/sparse_primal_falsifier.py \
  /tmp/sparse_primal_variant_sweep.py \
  /tmp/sparse_primal_joint_sweep.py \
  /tmp/sparse_primal_scale4_trace.py \
  /tmp/sparse-primal-falsifier/*.log || true

ps -eo pid,etime,args | \
  rg 'sparse_primal_(falsifier|variant_sweep|joint_sweep|scale4_trace)' || true
```

The network-token audit returned no hits. The process audit found no surviving
probe process after completion. Artifact hashes were produced with:

```bash
sha256sum \
  /tmp/sparse-primal-falsifier/results.json \
  /tmp/sparse-primal-falsifier/variant_sweep.json \
  /tmp/sparse-primal-falsifier/joint_scale4_sweep.json \
  /tmp/sparse-primal-falsifier/scale4_trace.json \
  /tmp/sparse_primal_falsifier.py \
  /tmp/sparse_primal_variant_sweep.py \
  /tmp/sparse_primal_joint_sweep.py \
  /tmp/sparse_primal_scale4_trace.py
```

Source and prior-evidence inspection used only `pwd`, `sed`, `rg`, `find`,
`ls`, and read-only Python JSON calculations. One initial bare `python`
calibration failed because that executable is absent, and the first driver
launch failed before solving because `PYTHONPATH=.` was omitted. Neither wrote
an evidence result. No probe was signaled or killed; the longest option sweep
completed normally. No Git command, network-capable fetch command, external
solver-source inspection, or target-repository source edit occurred.
