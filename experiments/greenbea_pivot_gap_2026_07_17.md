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

1. **Simplex-internal phase/start/step architecture — primary, 1,090 pivots
   measured (69.7% of the original gap).** On the identical 1525×3868 LP,
   HiGHS takes 3,309 and linprogx 4,399. Among the named candidates, explicit
   Dual Phase 1 + Phase 2 is the leading surviving discriminator: HiGHS spends
   1,655 / 1,633 / 21 pivots by phase, while linprogx runs one crash-seeded
   Phase-2-only walk. This measurement localizes the deficit to the internal
   engine but does not yet prove whether the decisive submechanism is the
   Phase-1 exit basis or subsequent entering/ratio-step choices.

2. **Presolve–simplex interaction — real secondary owner, 473 pivots (30.3%),
   but not transferable.** HiGHS gains 473 when moving from our reduction to
   its own raw presolve path. The reverse cross is decisive against a generic
   "smaller row count means fewer pivots" claim: linprogx regresses by 823 on
   that same 951-row reduction. The prior transferability story stops here.

3. **Bound-flipping ratio test — capability present; current longest-step
   variant owns at most 101 pivots.** Baseline flips 1,399 times. Full BFRT cuts
   only 101 pivots and raises wall work. A missing BFRT cannot own the 1,090;
   an implementation-quality difference remains unmeasured on the HiGHS side.

4. **Exposed crash option — killed.** Numeric strategies 0–9 are
   iteration-identical, and the log says no useful basis. A deeper built-in
   start-basis difference is folded into rank 1, but the available crash knob
   has no effect.

This is **not** an "advantage is not reproducible" closure. HiGHS remains 1,090
pivots ahead on the same reduced LP. The greenbea pivot frontier stays open,
but it is now a basis/phase-versus-step-mechanics question, not a leaving-rule,
missing-bound-flip, or row-count question.

## Next falsifiable probe: two-way basis transfer

**Probe.** On the same 1525×3868 LP with presolve off:

1. Export linprogx's exact triangular-crash starting basis behind a throwaway
   env-gated diagnostic and pass it to HiGHS with public `setBasis()`.
2. Stop HiGHS at iteration 1,656: the runtime trace verifies Dual Phase 1 ends
   at 1,655 and Phase 2 has taken exactly one pivot, while `getBasis()` reports
   a valid basis. Feed that basis/status assignment to a throwaway linprogx
   initial-basis hook.
3. Record remaining pivots, flips, phase/status, objective, and residual in both
   directions.

**Live criterion for start/phase ownership.** Keep the hypothesis if either
HiGHS rises by at least 600 pivots from 3,309 when forced onto linprogx's crash
basis, or linprogx finishes from the HiGHS Phase-1-exit basis in at most 2,200
additional pivots (near HiGHS's 1,633 Dual Phase 2 + 21 cleanup count).

**Kill criterion.** Kill start/phase-basis quality as owner of the majority if
HiGHS stays at or below 3,500 from linprogx's crash basis **and** linprogx still
needs at least 3,000 additional pivots from the HiGHS Phase-1-exit basis. That
would promote entering/ratio-step mechanics to the sole live explanation of
most of the 1,090 same-LP gap; the next probe would instrument breakpoint
distance/flip absorption per linprogx pivot rather than build another leaving
rule.

## Reproduction and raw artifacts

- Probe: `experiments/greenbea_pivot_gap_probe.py`
- Structured results: `/tmp/greenbea-pivot-gap/results.json`
- Maximum-level logs:
  `/tmp/greenbea-pivot-gap/{raw_presolve_on,raw_presolve_off,our_reduction_presolve_off,our_reduction_presolve_on}.log`
- Command:

```bash
cd /home/evan/dev/linprogx-gblog
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python \
  experiments/greenbea_pivot_gap_probe.py
```

Environment: CPython 3.14.3, `highspy==1.14.0` / HiGHS 1.14.0 (git 7df0786),
fixture `/tmp/lpsuite/lp_greenbea.mat`. No HiGHS source was read. HiGHS was
treated strictly through highspy public model/options/info/basis APIs, runtime
logs, and published option documentation. No solver source changes were made.
