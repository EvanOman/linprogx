# Integrated dual Phase-1 -> Phase-2 falsifier (2026-07-22)

## Verdict: KILLED

An integrated transition does not remove the greenbea loss. The existing cold
big-M path already performs the useful homogeneous-auxiliary basis work while
it advances the original solve. At cold pivot 2,060, the current basis solves
the exact homogeneous auxiliary in **zero additional pivots**. The best exact
main-prefix -> auxiliary -> main checkpoint uses 4,243 total pivots, only 3.55%
fewer than the 4,399-pivot cold path. Its measured solver-core opportunity is
about 1%; the best solver-core result in the whole retained checkpoint sweep is
only 4.13%.

That is far below both gates:

- a viable characterization-first candidate needed at least 20% charged
  whole-wall opportunity; and
- the certified board ratio of 1.2150867 needs a 17.7013% reduction to flip.

Exact alternating replays also fail: the best retained block alternation uses
4,328 pivots and is slower than cold when all stages are charged. A new combined
leaving or ratio rule that invents a different basis path would be a new pivot
selection family, not transition integration, and has no funding evidence here.

## Question and falsifier

The exact Phase-1 auxiliary is

```text
minimize   c'x
subject to Ax = 0
           0 <= x_j <= 1   for lower-only columns
           x_j = 0         for boxed columns
```

on presolved greenbea. It produces useful starts, including the 2,399-pivot K7
basis, but a separate exact construction costs 0.157--0.215 seconds. This probe
asked whether Phase-1 work could instead serve Phase-2 by:

1. running an auxiliary prefix and transitioning early to the original LP;
2. running an original-LP prefix, completing only the remaining exact
   auxiliary work, then continuing the original LP; or
3. alternating exact original and auxiliary blocks around one shared basis.

The family was live only if a certificate-clean global mechanism exposed at
least 20% charged whole-wall opportunity. All pivot work and all transition
calls were charged. In addition, the native `phase_us` buckets were summed to
form an *optimistic fused-core projection*: this removes Python-call, solver
initialization, and exit overhead that an in-process transition might avoid,
while retaining every measured pivot phase. Failure of that more favorable
projection is the decisive economic falsifier.

## Invariant: the cold path already owns the reusable Phase-1 state

For any basis `B` shared by the auxiliary and original problems,

```text
y(B) = B^-T c_B
r(B) = c - A^T y(B)
```

depend only on `A`, `c`, and `B`. They do **not** depend on the right-hand side
or variable bounds. A phase state `s` changes only the nonbasic bound placement
and primal basic solution:

```text
x_B^s = B^-1 (b_s - A_N x_N^s).
```

Therefore the only durable product that exact Phase-1 can give Phase-2 is its
basis, LU state, and associated reduced-cost state. There is no separate
objective computation whose removal can erase the 2,000-pivot auxiliary.
Changing from `b=0` and auxiliary boxes to the true RHS and bounds must still
reconcile `x_N`, bound statuses, and `x_B`.

The checkpoint replay directly locates that reusable state inside the existing
cold path. Starting the exact auxiliary from the cold basis needs 77 pivots at
cold checkpoint 2,000, 10 at checkpoint 2,050, and **zero** at checkpoint
2,060. A zero-pivot `optimal` auxiliary exit means the cold basis is already
primal- and dual-feasible for the exact auxiliary. Thus the first 2,060 cold
pivots have already performed the Phase-1 basis construction while working on
the original problem.

From that boundary the exact original continuation needs 2,211 pivots:

```text
2,060 integrated prefix + 0 auxiliary + 2,211 continuation = 4,271 pivots
```

The best nearby boundary is checkpoint 2,050:

```text
2,050 integrated prefix + 10 auxiliary + 2,183 continuation = 4,243 pivots
```

Even gifting all ten auxiliary pivots and every phase-transition overhead gives
4,233 effective pivots, a 3.77% reduction. A 20% opportunity requires at most
`floor(0.8 * 4399) = 3,519` effective pivots. The integrated-boundary family is
still 714 pivots above that deliberately generous ceiling.

## Setup

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- Presolved shape: 1,525 rows x 3,868 columns x 23,274 nonzeros.
- Exact auxiliary: zero RHS; `[0,1]` for 3,611 lower-only columns; `[0,0]`
  for 257 boxed columns.
- Native solver: `solve_eq_box_dual_simplex`, Dantzig leaving, BFRT off,
  EXPAND on, `tol=1e-8`.
- Certificate contract: `eps=2e-5`, original-space objective, equality, and
  bound checks.
- Diagnostic gates: `LINPROGX_DS_EXPORT_BASIS=1` and
  `LINPROGX_DS_WARM_START=1`.
- Runtime: `OPENBLAS_NUM_THREADS=1`, `UV_OFFLINE=1`, foreground.

## A. Auxiliary prefix -> exact main continuation

This is the direct attempt to truncate construction. Every prefix basis and
bound-status vector is transferred to an exact original-LP continuation.

| auxiliary cap | aux status / pivots | main status / pivots | total pivots | charged wall | fused-core time | core ratio vs cold |
|---:|---|---|---:|---:|---:|---:|
| 250 | iteration_limit / 250 | optimal / 4,173 | 4,423 | 0.5430s | 0.5355s | 1.0415 |
| 1,000 | iteration_limit / 1,000 | optimal / 3,552 | 4,552 | 0.5031s | 0.4948s | **0.9624** |
| 2,418 | iteration_limit / 2,418 | optimal / 2,399 | 4,817 | 0.5591s | 0.5502s | 1.0701 |

Cold is 4,399 pivots, 0.5188 seconds charged wall, and 0.5142 seconds of native
phase time. The best retained optimistic core ratio is 0.9624, a 3.76%
opportunity. Full exact construction reproduces K7's 2,399-pivot continuation
but costs 4,817 pivots end to end.

## B. Main prefix -> exact auxiliary completion -> exact main continuation

This is the strongest form of making Phase-1 work serve Phase-2: original-LP
pivots run first, only the missing exact auxiliary work is completed, and the
resulting basis resumes the original LP.

| main checkpoint | remaining aux pivots | final main pivots | total pivots | charged wall | fused-core time | core ratio vs cold |
|---:|---:|---:|---:|---:|---:|---:|
| 1,800 | 324 | 2,313 | 4,437 | 0.5300s | 0.5168s | 1.0052 |
| 1,900 | 175 | 2,213 | 4,288 | **0.5052s** | **0.4929s** | **0.9587** |
| 2,000 | 77 | 2,262 | 4,339 | 0.5292s | 0.5160s | 1.0036 |
| 2,020 | 63 | 2,176 | 4,259 | 0.5188s | 0.5068s | 0.9856 |
| 2,040 | 43 | 2,164 | 4,247 | 0.5076s | 0.4953s | 0.9633 |
| **2,050** | **10** | **2,183** | **4,243** | 0.5212s | 0.5089s | 0.9898 |
| 2,060 | **0** | 2,211 | 4,271 | 0.5156s | 0.5030s | 0.9782 |

Checkpoint 2,050 minimizes pivots, while checkpoint 1,900 has the best measured
core time. Neither supplies even one quarter of the 20% live gate. The
non-monotone continuation counts also show that a transition is not a smooth
global dial. Hard-coding checkpoint 2,050 would be prohibited per-problem
tuning even if its economics were adequate.

Exploratory ten-pivot resolution from 1,800 through 2,060 found no lower total;
2,050 remained best. Exact boundaries at 1,950 and 1,990 produced a final
`dual_infeasible` status, so arbitrary phase switching also has a real status
failure mode rather than merely a timing cost.

## C. Exact block alternation

Alternation repeatedly runs a bounded original block and a bounded auxiliary
block from the same exported basis, then finishes with an exact original solve.
This is a read-only replay proxy for synchronized multi-state pivoting.

| block size | main pivots | aux pivots | total pivots | charged wall | final status |
|---:|---:|---:|---:|---:|---|
| 50 | 3,348 | 1,183 | 4,531 | 0.8115s | optimal |
| 200 | 3,418 | 1,124 | 4,542 | 0.6619s | optimal |
| **800** | **3,526** | **802** | **4,328** | **0.5542s** | optimal |

The best alternation cuts only 71 pivots (1.61%) and remains 6.8% slower than
the charged cold wall. Finer exploratory blocks were worse: one-pivot
alternation terminated `dual_infeasible`, while two- and five-pivot alternation
needed 8,879 and 8,506 total pivots. The two primal-infeasibility states want
incompatible basis moves; frequent switching magnifies work instead of sharing
it.

A hypothetical pivot that optimizes a new joint merit across both states is
outside this result: it changes leaving selection and ratio semantics. That is
a new pivot-rule family, not a way to avoid transition construction, and it
would require its own falsifier and funding evidence.

## Certificate evidence

Every retained row in the three tables finishes with native `optimal` status
and passes the original-space `eps=2e-5` gates. Representative results:

| arm | objective | max equality residual | max bound violation |
|---|---:|---:|---:|
| cold | -72,555,248.12984590 | 1.77e-7 | 3.86e-12 |
| checkpoint 1,900 | -72,555,248.12984590 | 1.46e-8 | 4.55e-13 |
| checkpoint 2,050 | -72,555,248.12984598 | 1.45e-8 | 4.55e-13 |
| checkpoint 2,060 | -72,555,248.12984589 | 6.14e-8 | 4.55e-13 |
| alternation block 800 | -72,555,248.12984626 | 6.14e-8 | 4.55e-13 |

No near-objective or primal-only result was accepted when native status failed.

## Raw artifact

- Results: `/tmp/phase-transition-falsifier/results.json`
- Size: 7,645 bytes
- SHA-256:
  `b6a09a7fcab2b7efb79f84656419bc29d05b40acd53076c9eb8390107a52546c`

Corroborating inherited artifacts were read, not regenerated:

- `/tmp/k7-native-aux/results.json`
- `/tmp/phase1-predictions/results.json`
- `/tmp/k9-density-shaping/results.json`
- `/tmp/c5-pdhg-auxiliary/results.json`

## Reproduction

The raw artifact was generated with this local-only pattern; the checkpoint
and block lists below are exactly the retained rows in the artifact.

```bash
OPENBLAS_NUM_THREADS=1 \
UV_OFFLINE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
LINPROGX_DS_EXPORT_BASIS=1 \
LINPROGX_DS_WARM_START=1 \
uv run python - <<'PY'
from experiments.greenbea_pivot_gap_probe import prepare
from linprogx.presolve import postsolve_x
import numpy as np

original, p = prepare()
n = len(p["c"])
aux_lo = np.empty(n)
aux_hi = np.empty(n)
for j, (lo, hi) in enumerate(zip(p["lo"], p["hi"], strict=True)):
    if np.isfinite(lo) and np.isfinite(hi):
        aux_lo[j] = aux_hi[j] = 0.0
    elif np.isfinite(lo):
        aux_lo[j], aux_hi[j] = 0.0, 1.0
    elif np.isfinite(hi):
        aux_lo[j], aux_hi[j] = -1.0, 0.0
    else:
        aux_lo[j], aux_hi[j] = -1.0, 1.0

def run(kind, cap=50_000, basis=None, status=None):
    if kind == "main":
        b, lo, hi = p["b"], p["lo"], p["hi"]
    else:
        b, lo, hi = np.zeros_like(p["b"]), aux_lo, aux_hi
    warm = {}
    if basis is not None:
        warm = {"initial_basis": basis, "initial_bound_status": status}
    return p["matrix"].solve_eq_box_dual_simplex(
        p["c"].tolist(), b.tolist(), lo.tolist(), hi.tolist(),
        max_iter=cap, tol=1e-8, expand=1, leaving_rule=1, bfrt=0,
        **warm,
    )

def core_seconds(result):
    return sum(result["phase_us"].values()) / 1e6

# Pure auxiliary-prefix transitions.
for cap in (250, 1_000, 2_418):
    aux = run("aux", cap)
    final = run("main", basis=aux["basis"], status=aux["bound_status"])
    print(cap, aux["status"], aux["iterations"],
          final["status"], final["iterations"],
          core_seconds(aux) + core_seconds(final))

# Main-prefix, exact auxiliary completion, exact main continuation.
for cap in (1_800, 1_900, 2_000, 2_020, 2_040, 2_050, 2_060):
    prefix = run("main", cap)
    aux = run("aux", basis=prefix["basis"], status=prefix["bound_status"])
    final = run("main", basis=aux["basis"], status=aux["bound_status"])
    x = np.asarray(postsolve_x(final["x"], p["reduction"]))
    print(cap, prefix["iterations"], aux["iterations"], final["iterations"],
          final["status"], float(original["c"] @ x),
          core_seconds(prefix) + core_seconds(aux) + core_seconds(final))

# Exact alternating-block replay, main first, followed by exact main finish.
for block in (50, 200, 800):
    basis = status = None
    for _ in range(40):
        main = run("main", block, basis, status)
        basis, status = main["basis"], main["bound_status"]
        if main["status"] != "iteration_limit":
            break
        aux = run("aux", block, basis, status)
        basis, status = aux["basis"], aux["bound_status"]
    else:
        main = run("main", basis=basis, status=status)
    print(block, main["status"])
PY

sha256sum /tmp/phase-transition-falsifier/results.json
```

The production artifact generator additionally serialized every table field,
charged wall, original-space certificate, and summary into the JSON path above.

## Git and network audit

- **Git:** no Git command of any kind was run: no status, diff, checkout,
  switch, reset, commit, fetch, clone, or other operation.
- **Network:** no network tool or API was used. There was no `curl`, `wget`,
  HTTP client, browser, web search, or package download. Every `uv` invocation
  explicitly set `UV_OFFLINE=1`.
- **External solver boundary:** no external solver source was read. The probe
  used only linprogx's native dual-simplex API and existing local artifacts.
- **Filesystem:** no solver, test, documentation, or experiment source was
  changed during probing. The only generated result was
  `/tmp/phase-transition-falsifier/results.json`; temporary `/tmp/ds_part*.txt`
  files held read-only slices of the native implementation. The only repository
  write afterward was this assigned report.
- **Command census:** the executables used during the lane were `pwd`, `sed`,
  `rg`, `wc`, `head`, `find`, `jq`, `uv run python`, `sha256sum`, and `stat`,
  plus harness-level process polling. Report banking additionally used
  `apply_patch`, `sed`, `wc`, `rg`, `sha256sum`, and an offline `uv run python`
  assertion; one bare `python` validation attempt failed immediately because
  that command is unavailable. None is an unreported network or Git operation.

**Final verdict: KILLED.** The cold path already integrates the reusable exact
Phase-1 basis work by pivot 2,060. Explicit completion, early transition, and
alternation expose at most about 4% setup-free core opportunity, not the
required 20%, and arbitrary switching can fail certificate status.
