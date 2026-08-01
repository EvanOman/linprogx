# Why greenbea resists — the answer, and why reading HiGHS did not hand it to us

## The short version

**Reading HiGHS worked.** The mechanisms transplanted cleanly and are worth a
great deal: 25fv47 goes to **0.976x HiGHS on trajectory — below HiGHS** (from
2.74x), degen2 to **1.177x** (from 2.914x). Eight of nine simplex-routed cells
improve.

**greenbea does not improve, and now we know why: on greenbea our existing
simple rule is already anomalously good.** There is nothing for the better
machinery to recover.

## The experiment that showed it

greenbea and greenbeb are **the same linear program**:

| | |
|---|---|
| A | **identical** — 2392 x 5598, nnz 31070, elementwise difference exactly 0 |
| b | **identical** — all zeros |
| c | **identical** — 622 nonzeros |
| bounds | **differ in 333 columns** |

Since `b = 0`, the right-hand side is carried entirely in fixed-variable bounds
(292 of the 333 differences are `boxed -> boxed`; 82 are fixed in both with
different values). They are one model under two demand scenarios.

Patching all 333 bound columns from greenbea to greenbeb reproduces greenbeb
**exactly — 8,919 / 5,633 to the pivot**, which is the validity check that the
patch is real (`experiments/greenbea_bound_patch.py`).

| | Dantzig | exact DSE | DSE/Dantzig |
|---|---:|---:|---:|
| **greenbea** | **4,399** | 4,675 | 1.063 — DSE loses |
| **greenbeb** | **8,919** | 5,633 | 0.632 — DSE wins |
| **change across the RHS swap** | **+103%** | **+20%** | |

**Dantzig's sensitivity to the right-hand side is 5x DSE's.** DSE is the stable
rule. Dantzig is the volatile one — and on greenbea it lands on an unusually
good draw.

*(Intermediate patch fractions return `dual_infeasible`: a half-patched demand
vector is genuinely inconsistent data, so only the two endpoints are valid
solves. The ramp between them is not interpretable and is not used here.)*

## Why this reframes everything

The campaign spent months treating greenbea as **the cell where our dual simplex
is worst**. It is the opposite: it is the cell where our **simplest** rule is
best. Every consequence follows:

- **Why DSE helps 8 of 9 cells but not greenbea** — the greenbea Dantzig
  baseline is anomalously strong, so DSE has nothing to recover.
- **Why removing big-M makes greenbea worse** (4,829 two-phase vs 4,283
  single-phase) while it is worth −56% elsewhere — the bound-swap perturbs the
  favourable configuration.
- **Why greenbeb, on an identical matrix, behaves normally** — a different
  demand vector puts Dantzig back at its typical (poor) performance, and DSE
  duly wins by 1.58x.
- **Why the "conservation law" looked real** — every attempt to build a better
  start was competing against a baseline that was already near this rule's floor.

## So why can't we beat it, even with the source?

Because the remaining gap is **not a missing mechanism**. Reading HiGHS told us
what it does, we reimplemented it, and it delivers exactly what it promised
everywhere the baseline is ordinary. On greenbea:

- HiGHS reaches **2,836**.
- Our best rule reaches **4,399** — and that 4,399 is already a lucky low for it.
- The improved machinery (DSE, bound-swap phase 1, BFRT) reaches **4,675–6,619**.

HiGHS's 2,836 is not the product of any single mechanism we could read off and
port. It is the *composition* — DSE weights maintained correctly from a logical
basis, bound flipping, Harris two-pass, perturbation — operating together on an
instance where our shortcut happens to do unusually well and our replacements do
not yet compose as tightly.

**The honest position:** the source-informed campaign succeeded at the class
level and failed at greenbea, and it failed for a reason that no amount of
further reading would fix. Closing greenbea needs the full composition to work
*better than a rule that is already having a good day* — a materially harder
target than the 1.55x trajectory ratio suggests.

## What this rules out, and what it leaves

**Ruled out** (all measured, this session):
- exact DSE for greenbea — worse in both formulations and both phases
- the bound-swap two-phase for greenbea — worse than single-phase big-M
- a phase-keyed mixed rule — the best cell is the DSE/DSE diagonal, no interaction
- `pricing_update` optimisation — irreducible; four attacks, four kills
- thread oversubscription — no threads on this path (CPU/wall 0.74)
- route/presolve/glue overhead — 99.5% of the cell is the pivot loop

**Left standing:**
- per-pivot cost reduction (pays at the same rate as pivot count, since both are
  ~99.5% of the cell) — the pivot-cost census is live on this
- the churn penalty, shipped and certified, worth −2.6% pivots / −2.17% wall
- a genuinely better-composed dual simplex (DS2 + correctly-initialised DSE),
  which must beat a rule already near its floor on this instance
