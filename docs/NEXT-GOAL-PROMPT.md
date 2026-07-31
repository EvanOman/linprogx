# Next /goal prompt: close the six LPnetlib losses

Paste everything below into the successor session.

---

You are the mission controller for the next linprogx performance campaign. Use
the `overmind-v2` skill and its durable broker for every agent wave. Keep
synthesis, funding decisions, integration, and final judgment in the parent
session.

## The goal is 39 wins

Move the certified LPnetlib board from **33W-0P-6L** to
**39W-0P-0L against HiGHS** without losing coverage or regressing an existing
win.

The six losses from the protocol-v3 paired campaign are:

| Instance | Route | linprogx / HiGHS | Host range | Pair wins | Current iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| `lp_25fv47` | simplex | 3.531 | 3.342-3.541 | 0/21 | 6,948 |
| `lp_bnl2` | IPM | 2.677 | 2.326-2.771 | 0/21 | 63 |
| `lp_degen2` | simplex | 3.620 | 3.167-3.774 | 0/21 | 1,453 |
| `lp_ganges` | IPM | 1.071 | 0.936-1.127 | 7/21 | 16 |
| `lp_greenbeb` | simplex | 1.376 | 1.352-1.380 | 0/21 | 4,320 |
| `lp_sierra` | simplex | 2.737 | 2.113-2.756 | 0/21 | 725 |

Treat
`assets/modal_bench_c984f7fd5a33_paired_hosts3.json` as the board evidence and
`assets/modal_bench_c984f7fd5a33_suite_hosts3.json` as the three-solver sweep.
The paired artifact decides wins and losses. The suite artifact answers
coverage and broad performance questions but does not replace paired
certification.

The only successful terminal state is a certified 39-cell win. A killed
mechanism is useful evidence, but it is not completion. If the named lanes die,
record the reopening conditions and launch a fresh-eyes wave that proposes
mechanisms outside those closures.

## Preserve the current truth

Read these before dispatching work:

1. `AGENTS.md`
2. `experiments/greenbea_goal_v2_2026_07_29.md`
3. `docs/PROVENANCE.md`
4. `docs/DS2-REWRITE.md`
5. `experiments/dse_churn_class_verdict_2026_07_26.md`
6. `experiments/ds2_chuzr_2026_07_26.md`
7. `experiments/twophase_rule_matrix_2026_07_26.md`
8. `experiments/category_verdict_2026_07_26.md`, including its correction
9. `docs/HANDOFF.md` for the killed-mechanism ledger
10. `assets/lpnetlib_39_results.md`
11. `tools/modal_bench.py`

The latest board overrides stale timing claims in older reports. Old iteration
counts remain useful when the code path is unchanged. Old local wall times do
not score a cell.

Carry these findings forward:

- The six losses contain two separate classes. Four use the simplex route;
  `bnl2` and `ganges` use IPM.
- Exact DSE has already shown large trajectory gains on `25fv47` and `degen2`.
  Historical runs reached roughly 2,600-3,000 pivots on `25fv47` and 630-650 on
  `degen2`. Its extra tau FTRAN and denser trajectory can consume the pivot
  savings. Measure the composition on current code before claiming a win.
- `greenbeb` already receives the shipped controlled-aggregation plus DS2
  composition. It fell from 8,553 to 4,320 pivots and still loses by 1.376x.
  Repeating the greenbea campaign's winning composition is not a new idea.
- `sierra` needs only 725 linprogx pivots versus the historical HiGHS count of
  914, yet its paired wall ratio is 2.737. Its gap is fixed cost or per-pivot
  cost, not pivot count.
- `bnl2` and `ganges` finish in 63 and 16 IPM iterations. Their losses are not
  convergence failures. Profile setup, ordering, symbolic work,
  refactorization, triangular solves, residual scans, presolve, route glue,
  and process-level fixed cost.
- `ganges` is host-sensitive. One host already favors linprogx, while two do
  not. A local win or one favorable host is insufficient.
- The greenbea win came from an interaction that earlier component tests had
  missed: controlled fill-positive aggregation composed with DS2. Continue to
  test funded cross-products, not only isolated knobs.
- The shipped aggregation gate is global: at least 20% row reduction with at
  most 5% realized nnz growth on models already classified as simplex
  stall-risk. Do not weaken it to admit a named fixture.
- `greenbea` is now a mandatory simplex regression control. Preserve its
  0.986x certified win and its original-unit certificate.
- Coverage is 99.40% with a hard floor of 98%. Do not lower the floor, add
  exclusions, or manipulate the denominator.

## The constraints are binding

- Keep `eps=2e-5`.
- Accept certificate-backed optimality only. Reconstruct through presolve and
  verify objective, equality residual, bound violation, dual information, and
  status in original units.
- Use global structural or observed-state rules. Instance names, fixture
  fingerprints, constants selected to suit one loss, and equivalent disguised
  predicates are disqualifying.
- Preserve deterministic behavior where promised and all documented status
  semantics.
- Do not change Modal hardware, timeout, solver pins, warmup policy, or scoring
  rules to manufacture a win.
- HiGHS and Clarabel remain external correctness and timing oracles.
- Do not read new external solver source for these six losses without fresh,
  explicit authority from Evan. Existing source-informed mechanisms may be
  used, but their provenance stays attached. Public papers and black-box solver
  measurements are allowed.
- Never copy external implementation text. Reimplement understood algorithms
  independently.
- Characterize behavior before changing high-risk C paths.
- Run `just ci` before every integration handoff. The SciPy, Clarabel, and
  NumPy oracle tests must remain hard gates.
- Treat the main checkout as read-only during branch work. The mission
  controller creates one worktree per writer before launch. Workers receive a
  worktree as `cwd`, do not create nested worktrees, and stage only their files.
- Never use a metered provider or allow billing-class fallback. Use
  subscription-native workers only.
- Record kills with the same specificity as ships. Do not relabel an old kill
  and run it again unless the brief names the changed mechanism and the old
  reopening condition it satisfies.

## Fund work with measured arithmetic

For a current paired ratio `R`, a candidate that targets a 0.97 final ratio
must measure candidate/shipped at or below `0.97 / R`.

| Instance | Current `R` | Candidate/shipped funding gate |
| --- | ---: | ---: |
| `25fv47` | 3.531 | 0.275 |
| `bnl2` | 2.677 | 0.362 |
| `degen2` | 3.620 | 0.268 |
| `ganges` | 1.071 | 0.906 |
| `greenbeb` | 1.376 | 0.705 |
| `sierra` | 2.737 | 0.354 |

Recompute these gates if a fresh baseline changes materially. A mechanism may
be funded as one member of a measured composition, but projections, operation
counts, and isolated microbenchmarks do not substitute for an end-to-end
candidate/shipped measurement. Require a named causal chain from the changed
slice to the whole-cell gate.

Use iteration counts and CPU time for cheap local falsification. Use Modal
`envab` for code effects and Modal `paired` for the final comparison with
HiGHS. Always pass `--worktree` when uploading a source snapshot. Upload after
every candidate commit; record the exact SHA and Modal run URL.

## Run Overmind v2 as the control plane

Resolve the Overmind skill root and begin with:

```bash
/home/evan/dev/claude-skills-public/skills/overmind-v2/scripts/om doctor --json
```

Use the provider and billing facts returned by `doctor`. At the time this
prompt was written, both native adapters were installed, but the most recent
Codex failure reported a subscription usage limit until August 1. Do not infer
current capacity from that dated message. Check `doctor`, and either use an
available subscription-native provider or wait. Do not fall back to a metered
backend.

Launch each wave once with `run-many` and an idempotency key:

```bash
OVERMIND_BIN=/home/evan/dev/claude-skills-public/skills/overmind-v2/scripts/om
"$OVERMIND_BIN" run-many --input /tmp/linprogx-close6-wave1.json \
  --idempotency-key linprogx-close6-wave1-v1 --json
"$OVERMIND_BIN" await <group-id> --condition all_terminal \
  --since-cursor <cursor> --timeout 3600 --json
"$OVERMIND_BIN" collect <group-id> --preview-bytes 4000 --json
```

Do not poll `jobs` in a reasoning loop. Resume an interrupted `await` with its
last cursor. Inspect the result artifacts and the files named in each brief.
Terminal state plus a worker narrative is not proof; run the brief's
verification yourself.

Use `reply` for a bounded correction when a worker misunderstood its brief.
Stop obsolete jobs. If a job becomes `unknown`, inspect its requested
artifacts before deciding whether to continue or rerun it. Reuse the original
idempotency key on launch retries.

Each worker brief must use this exact shape:

```text
GOAL:        One observable outcome.
CONTEXT:     Relevant paths, current measurements, and dependencies.
CONSTRAINTS: Scope, invariants, billing class, provenance, and forbidden changes.
DONE WHEN:   Acceptance criteria visible outside the worker's narrative.
VERIFY:      Exact commands or checks.
```

Cap a wave at four active jobs. The parent session owns cross-wave synthesis.
Workers must emit a final report longer than 300 bytes and name every artifact
they created.

## Wave 0 records the baseline

The mission controller does this work before fan-out:

1. Record `git status`, current branch, HEAD, worktrees, dirty files, active
   benchmark processes, and available ports.
2. Run a focused correctness baseline and `just ci`.
3. Reproduce the six routes, statuses, iterations, objectives, and residuals
   from current source.
4. Confirm that the paired and suite artifacts parse through
   `tools/build_lpnetlib_report.py`.
5. Create a campaign ledger at
   `experiments/close_six_campaign_2026_07_31.md`. Record every later job ID,
   hypothesis, gate, result, commit, and Modal URL there.
6. Create dedicated worktrees only when a later wave has a funded writer.

Do not launch implementation work if the baseline differs from the board
without explaining the difference.

## Wave 1 reconciles the evidence

Launch four read-only jobs against the main checkout. Use `dontAsk` for Claude
read-only workers.

### W1-A owns the board truth table

```text
GOAL: Produce a source-linked dossier for the six current losses.
CONTEXT: Read the two c984 Modal artifacts, lpnetlib_39_summary.json, the
current route code, and the final greenbea report.
CONSTRAINTS: Read-only. Latest paired measurements override old wall claims.
No external solver source.
DONE WHEN: The report gives each loss's current route, dimensions after every
presolve stage, iterations, timing distribution, residual, objective delta,
known controls, required speedup, and code paths that can affect it.
VERIFY: Independently recompute every ratio from raw host results and list the
JSON paths used.
```

### W1-B owns the simplex closure map

```text
GOAL: Reconcile every existing simplex mechanism against 25fv47, degen2,
greenbeb, and sierra.
CONTEXT: Read DS2-REWRITE, dse_churn_class_verdict, ds2_chuzr, the two-phase
matrix, the final greenbea campaign, and relevant HANDOFF kills.
CONSTRAINTS: Read-only. Separate current production behavior from historical
gated arms and stale local timing.
DONE WHEN: The report maps Dantzig, churn, DSE, BFRT, two-phase bounds,
controlled aggregation, DS2, fused/unfused FTRAN, scaling, perturbation, and
refactor cadence to SHIPPED, LIVE-UNMEASURED, or KILLED for each loss. Every
kill includes its reopening condition.
VERIFY: Cite repository file paths and exact tables or commits for every claim.
```

### W1-C owns the IPM cost model

```text
GOAL: Identify which exclusive wall slices can fund bnl2 and ganges.
CONTEXT: Read ipm_slice_census, ipm_other_profile, presolve, route logic, and
the current C instrumentation.
CONSTRAINTS: Read-only. Do not infer wall opportunity from iteration counts.
No production changes.
DONE WHEN: The report specifies a measurement plan that attributes at least
98% of each target's wall and names controls with the same factorization and
size regimes. It states what slice would have to disappear to meet 0.362 and
0.906 candidate/shipped.
VERIFY: Give exact worker commands and expected output fields.
```

### W1-D attacks the campaign framing

```text
GOAL: Find omissions, unfair gates, stale assumptions, and genuinely new
mechanism classes.
CONTEXT: Read the loss dossiers, provenance boundary, current solver
architecture, and HANDOFF closure ledger.
CONSTRAINTS: Read-only. No external solver source. Reject renamed dead ideas.
DONE WHEN: Return at most eight ranked hypotheses. Each has a causal mechanism,
target cells, expected affected slice, cheap falsifier, funding invariant,
controls, and the prior kill it differs from. Include a recommendation to
stop funding any hypothesis that cannot meet a target gate.
VERIFY: Cross-reference each proposal against the killed-mechanism ledger.
```

The parent merges these reports into one canonical six-loss dossier. Resolve
contradictions before Wave 2.

## Wave 2 runs cheap falsifiers

Create one worktree per writer. Launch at most four experiment owners in
parallel. Production defaults stay untouched.

### W2-A measures the existing simplex arm matrix

Measure current production, forced Dantzig, exact DSE, two-phase combinations,
DS2, the controlled aggregation gate, and their funded cross-products on all
11 simplex-routed cases. Include the four losses and these controls:
`greenbea`, `agg2`, `agg3`, `cycle`, `fffff800`, `israel`, and `tuff`.

The deliverable is a machine-readable arm matrix with pivots, CPU time, status,
objective, residual, reduced shape, nnz, and phase timing. Kill any arm that
cannot certify or that regresses a control without a global, predeclared gate.

### W2-B measures DSE's full cost on current code

Attribute exact DSE's extra tau solve, denser trajectory, refactor behavior,
and other knock-on costs on `25fv47`, `degen2`, `greenbeb`, and `sierra`.
Exercise the already-gated fused two-RHS research path without shipping it.
Test whether any exact sharing preserves decisions and certificates across the
class. The old greenbea fused-FTRAN result is a regression control, not a
reason to skip the new targets.

The job is killed unless an end-to-end current-code measurement supports the
target's funding gate. A faster CHUZR scan alone is not funded; that lane is
closed.

### W2-C captures exclusive IPM slices

Use `LINPROGX_IPM_SLICE=1` and any additional knob-off instrumentation needed
to cover at least 98% of wall on `bnl2` and `ganges`. Include controls spanning
small and medium IPM routes: `fit1p`, `stocfor2`, `80bau3b`, `cre_a`,
`degen3`, and `woodw`.

Measure direct native IPM, public route, presolve, matrix conversion, setup and
ordering, symbolic work, numeric refactors, triangular solves,
matvec/residuals, certificate cleanup, and result marshalling. Instrumentation
must be absent and bit-identical when its environment flag is off.

### W2-D runs the route and reduction exchange

Force every existing algorithm and presolve mode on all six targets. Cross
normal and aggressive reductions with each compatible solver composition.
Measure current greenbea and all route controls. This is a bounded exchange
matrix, not a tuning sweep.

The report must distinguish:

- a route that cannot certify;
- a route that certifies but misses the wall gate;
- a reduction that helps only with a particular solver composition;
- a structural predicate that generalizes across controls.

After `await all_terminal`, collect bounded results, inspect full artifacts for
surviving mechanisms, and update the campaign ledger. Do not carry unfunded
ideas into implementation.

## Wave 3 builds only funded candidates

The parent selects at most three candidates. Each candidate gets a fresh
worktree, one owner, a characterization test written before behavior changes,
and a predeclared target/control set.

Candidate briefs must state:

1. the measured Wave 2 slice or interaction that funds the work;
2. the target candidate/shipped ratio;
3. the exact global gate, if any;
4. which source files the worker owns;
5. the original-unit certificate checks;
6. the controls that must remain on the same route or improve;
7. a kill command and rollback boundary.

Good candidates can change trajectory, fixed setup cost, factorization cost, or
route composition. They cannot claim that a component-level gain will compose
without an end-to-end measurement.

Every candidate must leave defaults gated until it passes:

- focused characterization tests;
- external-oracle comparison;
- all target and control certificates;
- deterministic repeat;
- local CPU-time sign test;
- `just ci`.

Commit each candidate separately. Do not combine two uncertain mechanisms into
one unreviewable patch.

## Wave 4 reviews and integrates survivors

Create one integration worktree from the current campaign baseline. Launch:

1. one integration owner who may cherry-pick only funded candidate commits;
2. one read-only numerical reviewer focused on certificates, presolve
   reconstruction, tolerances, statuses, and C memory ownership;
3. one read-only performance reviewer who recomputes the whole-wall funding
   math and searches for measurement leakage;
4. one read-only provenance and generalization reviewer who checks global
   gates, controls, source-informed labels, and public claims.

Reviewers do not approve by narrative. They name exact lines, tests, artifacts,
and failed assumptions. Use `reply` to send bounded corrections to the original
candidate owner when stateful context matters.

The integration owner runs the 39-case local correctness sweep and `just ci`.
Any cross-candidate regression returns the relevant candidate to Wave 3.

## Wave 5 certifies code effects on Modal

Only one benchmark operator launches Modal work. This avoids duplicate spend
and conflicting source snapshots.

For each integrated candidate:

1. Commit the exact source state.
2. Upload it with an explicit `--worktree`.
3. Run protocol-v3 `envab`, three hosts by seven pairs, candidate versus
   shipped baseline.
4. Include every target the mechanism can affect and the route controls below.
5. Require original-unit certification in every timed result.

Simplex controls:
`greenbea`, `agg2`, `agg3`, `cycle`, `fffff800`, `israel`, and `tuff`.

IPM controls:
`fit1p`, `stocfor2`, `80bau3b`, `cre_a`, `cre_d`, `degen3`, and `woodw`.

Include PDHG smoke controls if shared sparse kernels changed.

A candidate advances only when its median-of-hosts result meets the
predeclared funding gate and no control has a material regression. Run a fresh
paired comparison with HiGHS for every target that advances. `ganges` needs
three-host evidence with margin; one favorable host does not move the board.

Two independent read-only auditors parse the raw Modal JSON and recompute host
medians, pair wins, statuses, objectives, residuals, and candidate SHA. They
must agree before the parent records a board move.

## Wave 6 certifies the board and ships

After all six targets pass their paired gates:

1. Run the full 39-case, three-solver, three-host suite on the exact candidate
   commit.
2. Re-pair all six former losses against HiGHS.
3. Re-pair any existing win whose route or kernel changed and every prior
   knife-edge cell.
4. Confirm 39/39 linprogx coverage, unchanged `eps`, original-unit
   certificates, and deterministic routes.
5. Run `just ci` under the 98% coverage floor.
6. Regenerate the benchmark table, summary JSON, charts, coverage badge, and
   README from raw artifacts.
7. Update the campaign ledger, `docs/HANDOFF.md`, and provenance framing.
8. Commit in logical units and fast-forward local `main` only after the
   checkout is clean and every gate passes. Do not push unless Evan asks.

The final report gives:

- the 39W-0P-0L board;
- per-host ratios and pair wins for the six closed cells;
- the mechanism and global gate behind each move;
- controls and regressions tested;
- every killed lane and its reopening condition;
- exact commits, Modal URLs, raw artifacts, and CI output;
- clean-room versus source-informed provenance.

If Wave 6 does not reach 39 wins, return to a new Wave 1 with the accumulated
evidence. Do not weaken correctness, provenance, or protocol gates to finish.

---
