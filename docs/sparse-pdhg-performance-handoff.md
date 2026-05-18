# Sparse PDHG Performance Handoff

**Date:** Monday, May 18, 2026 at 01:41 PM CDT
**Primary worktree:** `/home/evan/dev/linprogx`
**Primary branch:** `sparse-support`
**Primary last commit:** `64d0f69 Add active-set cleanup for sparse PDHG`
**Rust experiment worktree:** `/home/evan/.config/superpowers/worktrees/linprogx/rust-pdhg-prototype`
**Rust experiment branch:** `rust-pdhg-prototype`
**Rust experiment last commit:** `405d0f6 Add experimental Rust PDHG backend (CSRMatrix in linprogx._rsparse)`

Both branches were pushed:

- `origin/sparse-support` at `64d0f69`
- `origin/rust-pdhg-prototype` at `405d0f6`

## High-Level Status

The current C-backed sparse PDHG branch is much faster than the earlier large-benchmark state, but it is still not at parity with HiGHS or Clarabel. DFL001 improved from the previous committed `24.042s` run to `9.467s` by adding a post-PDHG active-set CGLS feasibility cleanup and lowering the DFL001 PDHG budget from `220_000` to `58_000` iterations.

The new CYCLE guardrail still fails for `linprogx-sparse`, which is intentional and useful: it shows the current tuning remains DFL001-specific and does not solve a different sparse Netlib shape.

The Rust experiment was run in a separate worktree and branch. A faithful Rust/PyO3 port did not beat the C extension on DFL001. It reached rough parity on CYCLE and was slower on the small fast cases, so the next performance work should focus on structural algorithm changes rather than a language rewrite.

## Current Benchmark Results

### Netlib DFL001

Problem shape:

- Rows: `6071`
- Columns: `12230`
- Nonzeros: `35632`
- Published objective used by benchmark: `11266396.047`

Latest committed benchmark from `assets/large_dfl001_summary.md`:

| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266398.367904 | 2.321e+00 | 9.467s | max equality residual 1.867e-05; objective scale 1.5e+04 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 5.939s | HiGHS optimal |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 6.879s | max equality residual 1.074e-11 |

Important caveat: the `9.467s` DFL001 run is feasible under the configured `2e-5` equality tolerance, but the objective delta is worse than the previous 220k-iteration result. Previous result was `24.042s`, objective delta `2.455e-02`, residual `1.577e-05`. The current setting trades objective tightness for runtime.

### Netlib CYCLE Guardrail

Problem shape:

- Rows: `1903`
- Columns: `3371`
- Nonzeros: `21234`
- Published objective used by benchmark: `-5.2263930249`

Latest committed benchmark from `assets/cycle_summary.md`:

| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | iteration_limit | -4.574261 | 6.521e-01 | 3.592s | max equality residual 5.469e+00; objective scale 0.06 |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.309s | HiGHS optimal |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.361s | max equality residual 7.276e-12 |

The CYCLE failure is not just a runtime problem. Residual remains several units away from feasibility. This should be treated as an algorithm/scaling robustness issue.

### Rust Experiment

Branch: `rust-pdhg-prototype`

The Rust worker added:

- `rust/` PyO3 crate building `linprogx._rsparse.CSRMatrix`
- `csr_matrix_rust(...)` and `from_scipy_sparse(..., backend="rust")`
- `tests/test_sparse_rust.py`
- `bench_rust_vs_c.py`

Claude Code's final reported comparison:

| Case | Iters | C | Rust | C/Rust ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| grouped_box | 250 | 0.07ms | 0.09ms | 0.82x | Rust slower |
| chain_flow | 250 | 0.06ms | 0.08ms | 0.84x | Rust slower |
| random_feasible | 4000 | 1.71ms | 1.99ms | 0.86x | Rust slower |
| Netlib DFL001 probe | 10000 | 1726ms | 2544ms | 0.68x | Rust slower |
| Netlib CYCLE probe | 10000 | 408ms | 396ms | 1.02x | parity |

Numerics matched the C path in that experiment: same iterations, residuals, and objective values at the tested checkpoints.

## What Changed On `sparse-support`

Key commits:

- `64d0f69 Add active-set cleanup for sparse PDHG`
- `a90b3c7 Add CYCLE sparse benchmark guardrail`
- `19893fd Reduce DFL001 sparse benchmark runtime`
- `354951a Merge remote-tracking branch 'origin/main' into sparse-support`

Important files:

- `src/linprogx/_csparse.c`
  - Existing C CSR matrix path now stores CSC transpose data as well as CSR data.
  - Transpose matvec uses CSC column traversal instead of CSR scatter-add.
  - PDHG loop has fused projection/update work.
  - New active-set CGLS cleanup starts at `active_set_cgls_cleanup`.
  - Cleanup is called after PDHG if final max equality residual is still above tolerance.
- `bench_large.py`
  - DFL001 `linprogx-sparse` run now uses `max_iterations=58_000`, `check_interval=58_000`, `eps=2e-5`, `objective_scale=15_000.0`.
- `bench_cycle.py`
  - Adds CYCLE guardrail benchmark.
- `benchmark_data/netlib_cycle/lp_cycle.mat`
  - SuiteSparse LPnetlib CYCLE data.
- `assets/large_dfl001_*`
  - Latest DFL001 benchmark artifacts.
- `assets/cycle_*`
  - Latest CYCLE benchmark artifacts.
- `README.md`
  - Updated benchmark tables and narrative.

## Verification Already Run

On `/home/evan/dev/linprogx` after `64d0f69`:

```bash
just ci
uv run python bench_sparse_fast.py --iterations 4000 --repeats 10
uv run python bench_large.py
uv run python bench_cycle.py
uv build
git diff --check
```

Results:

- `just ci`: `67 passed`
- `bench_sparse_fast`: fast cases still pass; `random_feasible unit` still hits iteration limit as before
- `bench_large`: DFL001 table above
- `bench_cycle`: CYCLE table above
- `uv build`: built sdist and CPython 3.14 wheel successfully
- `git diff --check`: clean, only LF/CRLF warnings from git

On Rust worktree after `405d0f6`, Claude Code reported:

```bash
cargo build --release
uv sync --extra dev --reinstall-package linprogx
uv run pytest -q
uv run python bench_rust_vs_c.py --dfl001 --cycle ...
```

Result:

- Rust branch tests: `77 passed`
- Rust is not a clear speed win.

## Environment Setup

Main C branch:

```bash
cd /home/evan/dev/linprogx
uv sync --extra dev
```

If you edit `src/linprogx/_csparse.c`, rebuild the editable extension before benchmarking:

```bash
uv pip install -e . --force-reinstall
```

This matters. A previous probe accidentally used a stale `_csparse.cpython-314-x86_64-linux-gnu.so`, which made it look like the active-set cleanup was ineffective.

Rust experiment branch:

```bash
cd /home/evan/.config/superpowers/worktrees/linprogx/rust-pdhg-prototype
uv sync --extra dev --reinstall-package linprogx
```

## Core Benchmark Commands

Run the full local CI:

```bash
just ci
```

Run the small sparse benchmark:

```bash
uv run python bench_sparse_fast.py --iterations 4000 --repeats 10
```

Run DFL001 comparison and regenerate artifacts:

```bash
uv run python bench_large.py
```

Run CYCLE comparison and regenerate artifacts:

```bash
uv run python bench_cycle.py
```

Build package:

```bash
uv build
```

Run Rust-vs-C comparison in the Rust worktree:

```bash
cd /home/evan/.config/superpowers/worktrees/linprogx/rust-pdhg-prototype
uv run python bench_rust_vs_c.py --iterations 4000 --check-interval 250 --repeats 10
uv run python bench_rust_vs_c.py --dfl001 --dfl001-iter 10000 --dfl001-check 10000 --dfl001-eps 1e-4 --dfl001-scale 15000 --cycle --cycle-iter 10000 --cycle-check 10000 --cycle-eps 1e-4
```

## Useful Probe Commands

DFL001 parameter probe for the C branch:

```bash
uv run python -u - <<'PY'
from bench_large import DATA_PATH, EXPECTED_DFL001_OBJECTIVE, _bounds, load_dfl001
from linprogx.sparse import SparseLPProblem, SparseSolver
import time

data = load_dfl001(DATA_PATH)
problem = SparseLPProblem(
    c=data["c"].tolist(),
    A_eq=data["A"],
    b_eq=data["b"].tolist(),
    objective="min",
    bounds=_bounds(data),
    name="dfl001",
)

for scale in [8000, 10000, 12000, 15000, 20000]:
    for iters in [52000, 55000, 58000, 60000, 140000]:
        start = time.perf_counter()
        result = SparseSolver(
            algorithm="pdhg",
            max_iterations=iters,
            eps=2e-5,
            check_interval=iters,
            objective_scale=scale,
        ).solve(problem)
        seconds = time.perf_counter() - start
        objective = result.solution.objective_value
        delta = None if objective is None else abs(objective - EXPECTED_DFL001_OBJECTIVE)
        print(
            scale,
            iters,
            result.solution.status.value,
            result.solution.iterations,
            f"{seconds:.3f}",
            delta,
            result.solution.message,
            flush=True,
        )
PY
```

CYCLE parameter probe:

```bash
uv run python -u - <<'PY'
from bench_cycle import DATA_PATH, EXPECTED_CYCLE_OBJECTIVE, _bounds, load_cycle
from linprogx.sparse import SparseLPProblem, SparseSolver
import time

data = load_cycle(DATA_PATH)
problem = SparseLPProblem(
    c=data["c"].tolist(),
    A_eq=data["A"],
    b_eq=data["b"].tolist(),
    objective="min",
    bounds=_bounds(data),
    name="cycle",
)

for scale in [None, 0.03, 0.06, 0.1, 0.2, 1.0]:
    for iters in [50000, 100000, 200000]:
        start = time.perf_counter()
        result = SparseSolver(
            algorithm="pdhg",
            max_iterations=iters,
            eps=2e-5,
            check_interval=iters,
            objective_scale=scale,
        ).solve(problem)
        seconds = time.perf_counter() - start
        objective = result.solution.objective_value
        delta = None if objective is None else abs(objective - EXPECTED_CYCLE_OBJECTIVE)
        print(
            scale,
            iters,
            result.solution.status.value,
            result.solution.iterations,
            f"{seconds:.3f}",
            delta,
            result.solution.message,
            flush=True,
        )
PY
```

Profiling starting points:

```bash
uv pip install -e . --force-reinstall
perf stat -- uv run python bench_large.py
perf record -g -- uv run python bench_large.py
perf report
```

For C compiler experiments, rebuild with explicit flags and then rerun benchmarks:

```bash
CFLAGS="-O3 -march=native" uv pip install -e . --force-reinstall
uv run python bench_large.py
uv run python bench_cycle.py
```

Do not commit generated `.so` files.

## What We Tried And Learned

### Confirmed Useful

- CSC transpose storage improved transpose matvec by avoiding CSR scatter-add.
- Fusing the PDHG projection/update loop reduced allocation and loop overhead.
- Active-set CGLS cleanup can certify DFL001 feasibility after far fewer PDHG iterations.
- Lowering DFL001 to `58_000` iterations with cleanup is the best committed runtime point so far.

### Tried But Not A Win

- Multiple active-set cleanup passes (`max_passes = 3`) were tested locally and did not help DFL001 threshold cases. It was reverted to `max_passes = 1`.
- A faithful Rust port did not beat C on DFL001. It was close enough to be useful for experiments, but language alone is not the win.
- CYCLE does not improve meaningfully with current active-set cleanup; it remains infeasible by a large residual.

### Important Diagnostic

Before rebuilding the C extension, DFL001 probes still used an old `.so` from May 15 and falsely showed no active-set cleanup improvement. Always rebuild after C edits:

```bash
uv pip install -e . --force-reinstall
```

## Suggested Structural Investigation Next

1. Add column scaling / diagonal preconditioning to PDHG.

   Current native PDHG uses row scaling, but not a full row/column equilibration strategy. CYCLE is rank-deficient, denser, has free and upper-bounded variables, a negative optimum, and degeneracy. That makes it a better test of scaling robustness than DFL001. Implementing column scaling or Ruiz-style equilibration is the most plausible next algorithmic improvement.

2. Make active-set cleanup preconditioned and instrumented.

   Current CGLS cleanup is unpreconditioned and hardcoded to:

   - `margin = 1e-3`
   - `max_passes = 1`
   - `max_iter = 1000`

   Add instrumentation behind an optional debug flag or temporary probe to report:

   - free variable count
   - initial residual norm
   - final residual norm
   - step length
   - number of CGLS iterations
   - whether step was bound-limited

   Then try diagonal column preconditioning in the restricted least-squares solve. The SciPy LSQR prototype previously showed DFL001 can be corrected from around `1e-3` residual to around `2e-5` residual in one pass, so cleanup is viable. The CYCLE failure suggests the restricted solve or active-set selection is not robust enough.

3. Separate feasibility convergence from objective convergence.

   DFL001 at `58_000` iterations is feasible but has objective delta `2.321`. The older `140_000` to `220_000` region gives much tighter objective values at higher runtime. The benchmark currently reports both, but solver status is only feasibility-based. Decide whether the sparse PDHG status should include a dual gap, objective movement, or KKT-style criterion before optimizing only feasibility.

4. Investigate CYCLE rank deficiency / presolve.

   HiGHS and Clarabel solve CYCLE quickly. The current dependency-free path likely needs at least some structural preprocessing:

   - remove or combine duplicate/dependent equalities
   - detect empty rows / fixed variables
   - shift lower bounds more systematically
   - handle free variables and upper bounds in a better-scaled internal representation

   CYCLE should remain the guardrail for any DFL001 improvement.

5. Only after algorithmic changes, optimize raw kernels.

   Potential raw throughput work:

   - 32-bit index storage for matrices with dimensions under `INT32_MAX`
   - fused matvec and residual checks
   - `restrict` qualifiers and better local pointer hoisting in C
   - OpenMP or pthread parallelism for large row/column loops
   - `-march=native` benchmark-only build

   The Rust result says raw language choice is not enough; these need profiling evidence.

## Guardrails For The Next Agent

- Keep DFL001 and CYCLE both in the comparison table. Do not optimize only DFL001.
- Do not silently loosen `eps=2e-5` to make runs look better.
- Do not treat `iteration_limit` with a good objective as solved; equality residual matters.
- Rebuild the extension after every C change.
- Keep SciPy/HiGHS and Clarabel in every large-benchmark result for calibration.
- Watch objective delta as well as residual; DFL001 has a real speed/accuracy tradeoff now.
- If adding build flags or Rust machinery, keep the default dependency story clear. The original branch is meant to remain dependency-free for the C sparse path.

## Open Questions

- Can row/column equilibration make CYCLE converge at all under PDHG?
- Can a preconditioned active-set CGLS cleanup certify CYCLE feasibility after PDHG?
- What is the acceptable DFL001 objective delta for this project: relative `2e-7`, Clarabel-level `3e-2`, or HiGHS-level `1e-4`?
- Should `SparseSolver` expose a debug/stats mode for residuals, cleanup iterations, and scaling diagnostics?
- Should the benchmark report dual/KKT quality instead of only equality residual and objective delta?
