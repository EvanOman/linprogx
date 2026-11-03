# GOAL v2 — break the greenbea stalemate

## The goal

**Drive greenbea below 1.0 and make the board 24W-0P-0L.**

You do not stop or declare honorable closure. 23W is already achieved and
documented; it is not an outcome available to you. But note carefully what
changed: the previous goal prompt assumed the way to win was to make our dual
simplex better on greenbea. **That assumption is now the thing most likely to be
wrong.** See "The stalemate" below.

## Current position (measured, do not re-derive)

| | |
|---|---|
| Board | **23W-0P-1L** |
| greenbea cell | **~1.494** (v3 paired, 3 clean hosts, 0/21 pairs) |
| greenbea pivots | **4,283** (Dantzig + shipped churn) |
| HiGHS greenbea | **2,836** |
| Gap to close | **~35% pivots, or equivalent per-pivot cost, or any mix** |

## Facts established 2026-07-25/26 — treat as settled, do not re-measure

1. **greenbea's board cell is 99.5% the pivot loop.** Presolve is 3.0%, route
   and Python glue ~0. There is nothing outside the kernel to reclaim.
2. **Per-pivot cost vs HiGHS is at PARITY** (1.005–1.041x across three hosts).
   The old "linprogx is 1.73x better per pivot" is FALSE — it compared linprogx
   on a quiet local box against HiGHS in another context.
3. Therefore **wall ratio ≈ pivot ratio** (1.527 vs 1.551). Pivot-count work and
   per-pivot work pay at the **same rate**. Neither is privileged.
4. **The board ratio is host-draw dependent.** The 1.215/1.156 records did not
   reproduce. Any greenbea claim must be an `envab` result against a
   **freshly measured** paired baseline. Never compose an envab gain onto an old
   paired ratio.
5. **greenbea and greenbeb are the SAME LP** — A identical (2392x5598, nnz
   31070, elementwise difference 0), b identical (all zeros), c identical. They
   differ in 333 bound values, which *are* the right-hand side. Patching all 333
   reproduces greenbeb exactly (8,919/5,633).
6. **Dantzig's sensitivity to that RHS is 5x DSE's** (+103% vs +20%). So
   **greenbea is not where our solver is worst — it is where our simplest rule
   is luckiest.** 4,399 is an outlier-low draw for a volatile rule.
7. **DSE is the better rule overall**: mean 0.918 vs Dantzig's 1.429 against
   HiGHS, half the variance, wins 7 of 9 simplex cells, and takes 25fv47 *below*
   HiGHS (0.976x). It loses on greenbea alone, on both pivots and per-pivot cost.

## Dead ends — funded, measured, killed. Do not re-open without a new mechanism.

- exact DSE for greenbea — worse in both formulations and both phases
- the bound-swap two-phase for greenbea — 4,829 vs 4,283 single-phase big-M
  (it is worth −56% on 25fv47 and degen2; greenbea is the exception)
- phase-keyed mixed pricing rules — best cell is the DSE/DSE diagonal, no
  interaction to exploit
- `pricing_update` / the DSE FTRAN — irreducible; four attacks, four kills; even
  a *free* pricing_update leaves greenbea at 1.470x Dantzig
- thread oversubscription on constrained hosts — no threads on this path
  (CPU/wall 0.74)
- route / presolve / glue overhead — 99.5% of the cell is the pivot loop
- big-M magnitude tuning — 1e5/1e4/1e3 all ~4,400; 1e2 and below fail
- the "cheap CHUZR" lever — the scan is 3.7% of a pivot; the lever is 21x too small
- anti-cycling tabu windows — worse everywhere, and increase churn

## The stalemate, named precisely

Every remaining idea in the old frame is a variation on *"improve the dual
simplex's pivot selection on greenbea."* That frame has now produced eight
consecutive kills, and fact 6 explains why: **we are trying to beat a rule that
is already having its best day on this instance.** HiGHS does not beat greenbea
by having a better pivot rule for it — it reaches 2,836 with the *same* DSE it
uses everywhere, on an instance where our shortcut is unusually strong.

**So stop trying to win the argument we keep losing. Change the question.**

## Directions — ranked by how much new space they open

You are not required to take these, and inventing a better one is the point of
this prompt. Each is stated with a concrete first experiment so it can be killed
cheaply if wrong.

### A. Exploit Dantzig's variance instead of fighting it (highest novelty)

Fact 6 says Dantzig's greenbea trajectory is a **draw from a high-variance
distribution**. If 4,399 is a lucky draw, **other draws may be far better** —
and we have never looked at the distribution.

*First experiment:* enumerate the spread. Vary only tie-breaking and scan start
(`LINPROGX_DS_ROT_START`, index order, deterministic seeds) across ~30 variants
and plot greenbea's pivot count. If the spread reaches 3,000 the whole game
changes. A **fixed, deterministic portfolio of K variants** is a global
mechanism, not per-problem tuning — and on a 4-vCPU board host, K of them raced
in parallel costs wall = min(), not sum.

*Kill condition:* spread under ~5% means the trajectory is essentially
determined and this is dead.

### B. Are we even solving the same problem? (highest leverage per hour)

The campaign has compared pivot counts for months **without ever checking that
both solvers face the same reduced problem.** Our presolve gives 1525x3868 from
2392x5598. If HiGHS's presolve reduces further, part of the 4,283 vs 2,836 gap
is a **presolve gap**, not a pricing gap — and presolve is a nearly untouched
attack surface with no per-pivot cost at all.

*First experiment:* `highspy` exposes presolved dimensions. Compare rows, cols,
nnz after each solver's presolve. Then: feed HiGHS's presolved problem to *our*
simplex and ours to HiGHS. Four cells, and they separate "better presolve" from
"better pricing" completely.

*Kill condition:* dimensions match within a few percent.

### C. Don't use the dual simplex at all

greenbea reaches the dual simplex because `_ipm_stall_risk` routes it there. The
board measures **wall time**, not algorithmic virtue. There is a known,
unexploited routing bug next door: **greenbeb's IPM is 21% faster than its
simplex but is routed to simplex anyway.**

*First experiment:* force every algorithm on greenbea — IPM with varied
tolerances/regularisation, PDHG, PDHG-then-crossover, IPM-then-crossover — and
measure. The recorded "greenbea's IPM stalls" verdict predates presolve v2 and
the supernodal factor.

*Kill condition:* no route certifies within 2x of 616 ms.

### D. Scaling and conditioning

The dual simplex trajectory is strongly scale-dependent, and DSE is scale-
normalised where Dantzig is not (that is *why* fact 6 holds). greenbea's
one-sided-heavy, ±1-heavy structure may respond to a different equilibration.

*First experiment:* sweep scaling families (Curtis–Reid, geometric-mean,
equilibration, none) x {Dantzig, DSE} on greenbea and greenbeb. **greenbeb is
the control**: same matrix, different RHS. A scaling that helps greenbea but not
greenbeb is suspect (per-problem tuning in disguise); one that helps both is real.

### E. The composition, not the components (the one that never ran)

DS2 core exists and certifies **40/40** LPnetlib instances where the shipped
solver manages 36. It runs on stub components and is 1.13x median DS1 pivots.
The integration of exact DSE into it **was launched and never executed** — the
agent died on the account spend limit, not on the work. HiGHS's 2,836 is the
*composition* of DSE + BFRT + Harris two-pass + perturbation + bound-swap phase
1. We have measured every part separately and never once run them together.

*Note:* fact 6 predicts this will still lose on greenbea. Run it anyway — it is
the board-v2 headline regardless, and a confident prediction is worth testing.

### F. Change the instrument, honestly

`docs/BOARD-V2.md` proposes adding the 8 other simplex-routed cells, taking
simplex representation from 1/24 to 9/32. On that board DSE is a large win and
the class goes 1.843x → 1.115x. **This does not make greenbea win** and must
never be presented as if it did — but it is the structurally correct benchmark,
and the current board demonstrably selects a worse rule on the strength of one
lucky sample.

## Constraints — unchanged and binding

- `eps = 2e-5`, never loosened.
- **Certificate-backed optimality only.** Every accepted answer re-certifies in
  ORIGINAL units: objective vs published optimum, primal residual, bound violation.
- **No per-problem tuning.** Global mechanisms and thresholds only. A constant
  chosen because it suits greenbea is a kill, not a win. Use **greenbeb as the
  control** — same matrix, different RHS — a mechanism that helps greenbea but
  not greenbeb is tuning in disguise.
- **No verbatim copying** from HiGHS. Reading is authorised (owner, 2026-07-25);
  understand and reimplement. Mark every artifact
  `PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.` or
  `PROVENANCE: CLEAN-ROOM (independent)` with justification. Read
  `docs/PROVENANCE.md` — the 23 clean-room cells must never be laundered.
- **Honest reporting.** Kills are first-class results and are the majority of
  this ledger. Do not oversell an inconclusive measurement.
- `just ci` green; characterization tests before touching high-risk areas.
- Work in a **fresh worktree**. `/home/evan/dev/linprogx` belongs to other
  sessions — never touch it. Stage only paths you changed.
- Never kill a process to take a port.

## Measurement doctrine — hard-won, do not relearn

- This box has run at **load average 38–109 on 12 cores**. Cross-process wall
  comparisons drift 4–19% and are **unusable**.
- **Iteration counts and rdtsc cycles are load-invariant. Prefer them.**
- Under load measure **CPU time, not wall**; identical solves spread 46–61% on
  `perf_counter` and 5–6% on `process_time`. Gate verdicts with a **sign test** —
  call a result only when every paired repetition agrees in direction.
- On Modal use `--mode envab`, never paired, for code effects. For a
  dual-simplex change the default 7-cell set gives **six free null controls**
  (only greenbea routes to the DS), which measures the noise floor for you — it
  came out at ±1%.
- `modal_bench.py --action upload-src` **hardcodes a different worktree**. Pass
  `--worktree` explicitly or you will certify the wrong branch.
- Six idealised cost models have already failed to predict cycles. **Fund on
  measurement, never on projection.**

## Operational note

The account hit its **monthly spend limit** on 2026-07-26; two delegated agents
returned that message instead of doing their work. Check `om jobs` results for
rate-limit text before trusting any agent artifact — a job can report
`succeeded` while having produced nothing.

## What a win looks like

An `envab` v3 certification on Modal — 3 hosts x 7 pairs, loadavg 0.00 — showing
greenbea below **1.0** against a freshly measured paired baseline, with the
answer certified in original units, no per-problem constant, greenbeb unharmed
as control, and `just ci` green.

## What an honest non-win looks like

The same rigour, a clear verdict, and a ledger entry saying which of A–F died and
exactly why. That is worth more than a laundered number, and it is how the other
23 cells were earned.
