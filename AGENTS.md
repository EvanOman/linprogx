# linprogx Agent Contract

## Purpose

`linprogx` is a benchmarkable learning artifact: a small LP solver whose mechanics stay inspectable while it competes against mature open-source solvers on the included examples and Netlib-scale fixtures. It is not a production replacement for HiGHS, CLP, Gurobi, CPLEX, or Mosek.

Keep the runtime solver dependency-light. SciPy, Clarabel, NumPy, plotting, audit, and coverage tools belong in the `dev` extra for tests, comparisons, and benchmarks. OpenBLAS is the one native runtime dependency for the sparse C extension's dense-tail factorization.

## Architecture

- Dense public API: `src/linprogx/solver.py`, `model.py`, `api.py`, and `cli.py` expose the original two-phase simplex path and JSON CLI.
- Sparse public API: `src/linprogx/sparse.py` exposes CSR matrices, `SparseLPProblem`, and `SparseSolver`.
- C accelerators: `src/linprogx/_cfast.c` accelerates dense tableau operations; `src/linprogx/_csparse.c` owns the C CSR type, sparse PDHG, sparse Cholesky, and IPM kernels.
- Solver comparisons: `src/linprogx/compare.py` and `tests/test_samples_compare.py` use SciPy/HiGHS and Clarabel as correctness oracles. These are required gates, not optional smoke tests.
- Historical performance context lives in `docs/sparse-pdhg-performance-handoff.md`. Treat it as a design journal, not the source of truth for current validation commands.

## Change Boundaries

Routine changes:

- Add or update hand-authored LP samples and benchmark summaries.
- Improve Python API ergonomics while preserving existing return types and statuses.
- Add focused tests, CLI coverage, and comparison checks.
- Tune tolerances only with evidence from tests or benchmark fixtures.

High-risk changes:

- Sparse Cholesky factorization, supernode detection, dense-tail selection, OpenBLAS calls, and triangular solves.
- PDHG restart/adaptation logic and feasibility/certificate acceptance.
- Presolve reductions and reconstruction of primal or dual solutions.
- Status mapping for infeasible, unbounded, max-iteration, and certificate-backed optimal results.
- Memory ownership, pointer lifetimes, threading, and buffer indexing in the C extensions.

For high-risk changes, add characterization tests before changing behavior and include at least one external-oracle comparison or residual/certificate check in verification.

## Validation

Run these locally before handing off substantial changes:

```bash
just ci
```

`just ci` must run lint, format check, type check, security checks, and coverage-gated tests. Use narrower commands while iterating:

```bash
just lint
just format-check
just type
just security
just test-cov
```

Critical correctness tests must fail loudly if SciPy, Clarabel, or NumPy are unavailable. Do not use `pytest.importorskip` in oracle tests that compare against external solvers.

Coverage must stay at or above the configured floor. Lowering the floor is a contract change and needs an explicit reason.

For C extension work, verify the extension builds through `uv sync --extra dev` or `uv run pytest`, and run focused tests covering both the accelerated path and the Python-facing behavior.

## Forward Roadmap

The HiGHS head-to-head campaign on the 24 LPnetlib instances is effectively closed: the board of record is 23 wins, 0 parity, 1 loss (greenbea at 1.215x), measured under the paired multi-host protocol in `tools/modal_bench.py`. The supernodal Cholesky factor, presolve v2, and the bounded-variable dual simplex are all shipped. The greenbea residual was falsified down to a pivot-path/hardware floor across dozens of documented attack waves (`docs/HANDOFF.md`, `experiments/`); reopen it only with a mechanism that clears the funding invariants recorded there. Preserve the certificate gates, `eps=2e-5`, and the no-per-problem-tuning rule in any future performance work.

Sparse solver performance changes should be measured on the included benchmark fixtures, but benchmark timing alone is never enough. The solver must also preserve objective agreement, feasibility residuals, deterministic behavior where promised, and documented status semantics.
