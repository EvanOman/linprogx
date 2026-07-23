# greenbea pivot-gap measurement — 2026-07-17

## Decisive cross: **3,309 HiGHS iterations on our reduction**

HiGHS 1.14.0, dual simplex, presolve **off**, solving linprogx's exact
**1,525×3,868×23,274** presolved greenbea LP, takes **3,309 simplex
iterations**.

That is neither binary endpoint from the preregistration. The original gap is
4,399 - 2,836 = **1,563 pivots**, split on a common LP as follows:

| Component | Pivots | Share of original gap |
| --- | ---: | ---: |
| Simplex-internal: linprogx 4,399 vs HiGHS 3,309 on the same reduction | **1,090** | **69.7%** |
| HiGHS presolve geometry: HiGHS 3,309 on ours vs 2,836 on its own raw-to-reduced path | **473** | **30.3%** |
| Total | **1,563** | **100.0%** |

The cross solution is oracle-clean in original space after linprogx postsolve:
objective **-72,555,248.1298**, maximum equality residual **6.14e-8**, maximum
bound violation **4.77e-12**. linprogx Dantzig reports the same objective with
maximum original-space equality residual **1.77e-7**. Both are inside the
existing 2e-5 gate.

Verdict: the HiGHS advantage **persists on our reduction**, but at 3,309 rather
than ~2,800. The majority is simplex-internal; the remaining 473 pivots are a
real, solver-specific presolve interaction.

## Reverse cross: feasible, and it rejects presolve transferability

Contrary to the expected skip condition, the public `getPresolvedLp()` export
for greenbea contains **951 equalities and zero ranged rows**. It is directly
representable by linprogx's equality-plus-bounds API.

| Solver / ratio test | Reduction | Pivots | Bound flips | Objective incl. HiGHS offset | Reduced equality residual |
| --- | --- | ---: | ---: | ---: | ---: |
| linprogx Dantzig, baseline Harris | linprogx 1525×3868×23274 | **4,399** | 1,399 | -72,555,248.129846 | original-space 1.77e-7 |
| linprogx Dantzig, longest-step `bfrt=1` | linprogx 1525×3868×23274 | 4,298 | 1,443 | -72,555,248.129900 | original-space 4.77e-7 |
| linprogx Dantzig, baseline Harris | HiGHS 951×3158×23609 | **5,222** | 2,263 | -72,555,248.129846 | 2.04e-8 |
| linprogx Dantzig, longest-step `bfrt=1` | HiGHS 951×3158×23609 | 5,056 | 2,503 | -72,555,248.129846 | 5.78e-8 |

HiGHS's smaller reduction makes linprogx **823 pivots worse** under the shipped
ratio test (5,222 vs 4,399), and 758 worse under longest-step BFRT (5,056 vs
4,298). The earlier row-count story does not transfer: HiGHS's aggregation
geometry is favorable to HiGHS's own pivot mechanics, not intrinsically a
shorter basis path.

As a second order check, applying HiGHS presolve to linprogx's already-reduced
LP reaches 951×3158×23620 and takes **2,703** iterations, 133 fewer than HiGHS's
normal raw-to-951×3158 path. Same dimensions are not the same pivot geometry;
elimination order and the resulting coefficients/order matter.

## HiGHS log anatomy

All anatomy runs forced `solver="simplex"`, used maximum accepted
`log_dev_level=3`, and captured logs with `log_file`. The base dual setting was
`simplex_strategy=1` (serial dual); defaults retained were
`simplex_dual_edge_weight_strategy=-1` (choose),
`simplex_scale_strategy=2`, `simplex_price_strategy=3`, and
`simplex_crash_strategy=0`. HiGHS reported six available threads.

| Input / presolve | Reduced shape solved | Dual Phase 1 | Dual Phase 2 | Primal Phase 2 cleanup | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw, HiGHS presolve on | 951×3158×23609 | **1,448** | **1,376** | 12 | **2,836** |
| Raw, presolve off | 2392×5598×31070 | 3,195 | 2,623 | 10 | 5,828 |
| linprogx reduction, presolve off — decisive cross | 1525×3868×23274 | **1,655** | **1,633** | 21 | **3,309** |
| linprogx reduction, HiGHS presolve on | 951×3158×23620 | 1,322 | 1,370 | 11 | 2,703 |

The 473-pivot raw-path advantage over our reduction is spread across both main
phases: 207 fewer Dual Phase-1 pivots, 257 fewer Dual Phase-2 pivots, and nine
fewer cleanup pivots. It is not isolated to feasibility construction.

The decisive-cross trace begins with:

```text
Solving LP without presolve or useful basis
Using dual simplex solver
DuPh1 0 ... Ph1: 30(138.552); Du: 9(102.195)
```

It ends with:

```text
Simplex iterations: DuPh1 1655; DuPh2 1633; PrPh2 21; Total 3309
```

Crash facts: the maximum trace reports no crash-basis size and says it is
solving without a useful basis. The runtime-exposed `simplex_crash_strategy`
accepts numeric values 0 through 9 in highspy 1.14.0, but its meanings are not
published in the generated option file. Every numeric setting produced the
identical 3,309-iteration result on our fixed reduction. This exposed crash
lever has zero measured ownership of the gap.

Bound-flip facts: no `flip`, `bound flip`, or `bound swap` event/count appears
even at `log_dev_level=3`, and highspy exposes no bound-flip/ratio-test option.
HiGHS's bound-flip count therefore remains **not reported**, not zero. It cannot
be isolated further through the allowed black-box surface.

## HiGHS option ablations on the fixed linprogx reduction

The public option meanings for simplex strategy and edge weights come from the
[HiGHS option definitions](https://ergo-code.github.io/HiGHS/dev/options/definitions/);
the [solver guide](https://ergo-code.github.io/HiGHS/dev/solvers/) confirms that
`solver="simplex"` and `simplex_strategy` choose the simplex route. All runs
used presolve off and reached the same original-space objective/residual gate.

| Option family | Value | Meaning / treatment | Iterations |
| --- | ---: | --- | ---: |
| `simplex_strategy` | 0 | choose | 3,309 |
|  | **1** | serial dual, base | **3,309** |
|  | 2 | dual SIP | 3,309 |
|  | 3 | dual PAMI | 4,669 |
|  | 4 | primal | 13,809 |
| `simplex_crash_strategy` | **0–9** | opaque runtime-exposed numeric ablation | **3,309 for every value** |
| `simplex_dual_edge_weight_strategy` | -1 | choose, base | **3,309** |
|  | 0 | Dantzig | 12,279 |
|  | 1 | Devex | 7,014 |
|  | 2 | steepest edge | **3,309** |

The edge-weight table does not reopen linprogx's leaving-rule result. It says
only that HiGHS's own steepest-edge trajectory is its best exposed trajectory
on this representation. linprogx's exact DSE was already measured at 4,675 vs
Dantzig 4,399; rule names do not imply equivalent bases, tie handling, phase
machinery, or ratio tests across solvers.

## linprogx phase, route, and bound-flip anatomy

No new C instrumentation was necessary: the native DS result already returns
`bound_flips`, `artificial_ejections`, degeneracy counters, and all phase timing
buckets.

| Measurement | Shipped Harris (`bfrt=0`) | Longest-step BFRT (`bfrt=1`) |
| --- | ---: | ---: |
| Dantzig pivots | **4,399** | **4,298** |
| Bound flips | **1,399** | **1,443** |
| Artificial-basis ejections | 30 | 30 |
| Degenerate pivots | 1 | 3 |
| Refactorizations | 33 | 28 |
| Ratio-test time | 73.0ms | 180.5ms |
| Total instrumented phase time | 411.1ms | 530.7ms |

The shipped ratio test does bound-flip frequently: 1,399 flips, or 0.318 flips
per pivot. The optional full breakpoint walk adds 44 flips and removes only
**101 pivots (2.30%)**, while more than doubling ratio-test time. It explains
only 101 / 1,090 = **9.3%** of the same-LP internal deficit and still leaves a
989-pivot gap to HiGHS on our reduction. A missing flip capability is killed;
a materially better HiGHS long-step implementation cannot be tested because
HiGHS exposes neither its count nor a disabling option.

Route/phase count for the public 4,399 result:

| Public-auto component | Iterations |
| --- | ---: |
| IPM before DS | **0** |
| Post-IPM DS rescue | **0** |
| Early stall-predictor DS call | **4,399** |

The public message is `stall predictor routed to dual simplex`; it takes the
early DS shortcut before IPM, not the later rescue route. All 4,399 iterations
are inside one native DS call. linprogx's C solver describes this as
"Phase-2 only": its triangular structural crash leaves 30 artificial basis
columns that are ejected during the unified loop. `artificial_ejections=30` is
an event count, not a separable Phase-1 pivot count, so inventing a Phase-1 /
Phase-2 split for linprogx would be false precision.

## Ranked conclusion

1. **Simplex-internal step mechanics — primary, and the basis-transfer probe
   below closes the phase/start alternative.** On the identical 1525×3868 LP,
   HiGHS takes 3,309 and linprogx 4,399. Importing HiGHS's certified
   Phase-1-plus-one basis and exact nonbasic statuses cuts linprogx to 3,529,
   but HiGHS itself resumes from that basis in 1,594 pivots. The common-start
   continuation gap is therefore 1,935 pivots. Basis quality helps, but the
   downstream pivot path remains decisive.

2. **Presolve–simplex interaction — real secondary owner, 473 pivots (30.3%),
   but not transferable.** HiGHS gains 473 when moving from our reduction to
   its own raw presolve path. The reverse cross is decisive against a generic
   "smaller row count means fewer pivots" claim: linprogx regresses by 823 on
   that same 951-row reduction. The prior transferability story stops here.

3. **Bound-flipping ratio test — capability present; current longest-step
   variant owns at most 101 pivots.** Baseline flips 1,399 times. Full BFRT cuts
   only 101 pivots and raises wall work. A missing BFRT cannot own the 1,090;
   an implementation-quality difference remains unmeasured on the HiGHS side.

4. **Crash / Phase-1 starting basis — killed.** Numeric crash strategies 0–9
   are iteration-identical. More decisively, transferring HiGHS's own
   Phase-1 exit into linprogx leaves 3,529 pivots, above the registered 3,200
   kill line. HiGHS-optimal -> linprogx takes zero pivots, proving the transfer
   hook itself is faithful.

HiGHS remains 1,090 pivots ahead on the same reduced LP, but greenbea's scoped
**pivot-count frontier is now closed**: leaving rules, missing flips, generic
row count, exposed crash strategies, and HiGHS-quality Phase-1 basis transfer
all miss their gates. Further wall improvement requires cheaper pivots or a
materially different internal step path, not a commissioned crash/Phase-1 unit.

## Two-way basis transfer — **KILLED (3,529 pivots)**

### Extraction and mapping method

HiGHS was accessed only through documented highspy basis APIs:
`getBasis()` for extraction and `setBasis()` for the reverse transfer.

1. An uninterrupted maximum-log run established the phase boundary:
   `DuPh1 1655; DuPh2 1633; PrPh2 21; Total 3309`.
2. With `simplex_iteration_limit=1655`, `getBasis()` returned a valid basis
   containing 989 structural basics and 536 basic row variables. This is the
   basic set after the 1,655th Phase-1 pivot. Iteration-limit return semantics
   matter: resuming from that returned useful basis performs 52 recovery
   Phase-1 pivots, so it is reported as a boundary basis, not treated as a
   seamless internal checkpoint.
3. The verdict run used `simplex_iteration_limit=1656`. Its trace explicitly
   says `DuPh1 1655; DuPh2 1; Total 1656`, so its valid basis is the Phase-1
   exit plus exactly one Phase-2 pivot. It contains 990 structural basics and
   535 basic row variables. Resuming HiGHS from it starts directly in Dual
   Phase 2 and needs 1,594 iterations (1,572 Dual Phase 2 + 22 cleanup).
4. HiGHS row-basic variables map to linprogx artificial identity columns
   `n+i`; structural basics retain their column indices. Lower/upper/zero
   statuses map to linprogx's nonbasic bound-status codes.
5. linprogx gained diagnostic-only `initial_basis` and
   `initial_bound_status` kwargs. They are rejected unless
   `LINPROGX_DS_WARM_START=1`; final basis/status export requires
   `LINPROGX_DS_EXPORT_BASIS=1`. The normal crash path is otherwise untouched.

### Transfer results

| Start / solver | Pivots | Bound flips | Artificial ejections | Wall | Original residual | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linprogx native crash -> linprogx | 4,399 | 1,399 | 30 | 0.390s | 1.77e-7 | -72,555,248.129846 |
| HiGHS iter-1655 boundary basis only -> linprogx | 3,648 | 143 | 632 | 0.407s | 2.38e-7 | -72,555,248.129847 |
| **HiGHS Phase-1-plus-one basis + exact statuses -> linprogx** | **3,529** | 121 | 652 | 0.399s | 1.46e-8 | -72,555,248.129846 |
| HiGHS optimal basis + exact statuses -> linprogx | **0** | 0 | 0 | 0.003s | 5.78e-8 | -72,555,248.129846 |
| linprogx optimal basis + statuses -> HiGHS | **4** | n/r | n/r | 0.012s | 4.44e-8 | -72,555,248.129846 |

Every solve is optimal and passes the registered 2e-5 relative objective gate;
maximum equality residual is 2.38e-7 and maximum bound violation is 3.86e-12.
The zero-pivot HiGHS-optimal -> linprogx result validates the full basis and
nonbasic-status mapping. The four-pivot reverse result is the expected
near-zero tolerance cleanup.

### Registered verdict

- **LIVE:** fewer than 2,600 linprogx pivots from HiGHS's Phase-1 exit.
- **KILLED:** more than 3,200 pivots.
- **Measured:** **3,529 pivots — KILLED**, 329 beyond the kill line and 929
  above the live line.

The transferred basis saves 870 pivots (19.8%) versus linprogx's native crash,
but does not improve wall: 0.399s versus 0.390s in the recorded run. The basis
makes solves denser: mean FTRAN density rises 0.241 -> 0.326 (+35.4%) and BTRAN
density 0.428 -> 0.475 (+11.1%). Observed wall per pivot rises from 88.8us to
113.1us (+27.4%).

This also closes the wall argument. At the transferred basis's measured rate,
even 2,836 pivots project to about 0.321s, above HiGHS's measured 0.266s on our
reduction; 3,309 pivots project to about 0.374s. HiGHS-level pivot counts would
still require per-pivot parity. No dual-Phase-1/crash unit is commissioned.

## Reproduction and raw artifacts

- Probe: `experiments/greenbea_pivot_gap_probe.py`
- Basis-transfer probe: `experiments/greenbea_basis_transfer_probe.py`
- Structured results: `/tmp/greenbea-pivot-gap/results.json`
- Basis-transfer results: `/tmp/greenbea-basis-transfer/results.json`
- Maximum-level logs:
  `/tmp/greenbea-pivot-gap/{raw_presolve_on,raw_presolve_off,our_reduction_presolve_off,our_reduction_presolve_on}.log`
- Basis logs:
  `/tmp/greenbea-basis-transfer/{highs_phase1_boundary,highs_phase1_plus_one,highs_optimal_basis,highs_from_linprogx_optimal}.log`
- Command:

```bash
cd /home/evan/dev/linprogx-gblog
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation

PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python \
  experiments/greenbea_pivot_gap_probe.py

LINPROGX_DS_WARM_START=1 LINPROGX_DS_EXPORT_BASIS=1 \
  PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python \
  experiments/greenbea_basis_transfer_probe.py
```

Environment: CPython 3.14.3, `highspy==1.14.0` / HiGHS 1.14.0 (git 7df0786),
fixture `/tmp/lpsuite/lp_greenbea.mat`. No HiGHS source was read. HiGHS was
treated strictly through highspy public model/options/info/basis APIs, runtime
logs, and published option documentation. The only solver-source change is the
default-off, env-gated diagnostic basis hook described above; a no-env replay
remains exactly 4,399 pivots / 1,399 flips, and `tests/test_dual_simplex.py`
passes 27/27.
