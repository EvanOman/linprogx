# Harris dead-work early-outs — bit-identical unit (2026-07-25)

**Verdict: REAL, BIT-IDENTICAL, FUNDED AS A COMPOSITION MEMBER.**
greenbea `ratio_test` −23.63%, whole-wall −2.22% (paired median), zero
trajectory change. Not a flip on its own (17.7013% required); it is the first
member of a stack.

## Why this was reachable at all (the reopening)

The kernel campaign's own master census
(`experiments/k1_census_2026_07_19.md`) ranked **K5 scan+update fusion** as its
#3 angle: *"PRICE (18.9%) + rcost (10.0%) touch the same dense alpha vectors
twice; fusion attacks memory traffic on ~29% of wall."*

`docs/HANDOFF.md` (KERNEL CAMPAIGN CLOSED, 2026-07-20) records its disposition:

> ABANDONED (opencode zombies killed after 24h hung at build; angles' residual
> value negligible against the proven floor): **K5 fusion**, K6 prefetch,
> K10 threaded PRICE, K11 fp32-compare.

So K5 was **never measured**. It died of worker-infrastructure failure and was
then rationalised against "the proven floor". That floor is the **gathered
sparse triangular solve** floor (IPC 0.30–0.60, immune to width/flags/
pipelining/scheduling). K1 measured the pivot-row pipeline at **IPC 1.51
(PRICE), 1.30 (ratio), 1.41 (rcost)** — explicitly *not* on that floor.
Applying the triangular-solve floor to these slices was a category error, and it
left the campaign's largest never-measured slice unexamined.

Re-measured share of that pipeline on today's checkout: **46.05% of wall**
(`pivot_row` 21.97% + `ratio_test` 13.04% + `rcost_update` 11.04%). The
inherited dossier's "solves are 37% of wall" is `btran_rho + ftran_col =
37.64%` and **excludes `pivot_row`, the single largest phase.**

## The dead work (measured, deterministic)

New env-gated instrumentation (`LINPROGX_DS_HARRIS_CENSUS=1`) counting the
AVX2 eligibility masks in `ds_harris_pass1_avx2`. greenbea, all 4,399 pivots,
bit-reproducible across repetitions:

| quantity | per pivot | fraction |
|---|---:|---:|
| columns scanned (`n_total`) | 5,392 | 100% |
| 4-column AVX2 blocks | 1,348 | 100% |
| blocks with **all-zero alpha** | 890 | **66.00%** |
| blocks with **empty eligibility mask** | 1,187 | **88.08%** |
| true numerical support (\|alpha\| ≥ 1e-9) | 1,225.93 | 22.74% |
| admissible columns | 335.26 | 6.22% |

The shipped kernel computes, for **every** block:

```c
__m256d ratio = _mm256_div_pd(_mm256_add_pd(abs_r, tau), abs_alpha);
ratio = _mm256_blendv_pd(_mm256_set1_pd(INFINITY), ratio, eligible_v);
theta_min_v = _mm256_min_pd(theta_min_v, ratio);
```

The blend replaces ineligible lanes with `INFINITY` **after** the divide, and
`min(x, INFINITY) == x`. So for the 88.08% of blocks with an empty mask the
entire tail — including an **unpipelined `vdivpd`** (~8-cycle reciprocal
throughput on Zen 2, not pipelined like mul/add) — is provably dead.

## The change (two early-outs, both bit-identical)

1. **Cheapest-filter-first.** `eligible_mask = nonbasic & alpha & direction`,
   so `alpha_mask == 0` forces `eligible_mask == 0`. Test it immediately after
   the alpha load — *before* the `basis_pos` load, the `bound_status`
   load/widen, and the three status compares. Kills 66.00% of blocks after
   ~5 ops instead of ~15 plus a divide.
2. **Dead-division skip.** For the remaining blocks, `continue` when
   `eligible_mask == 0`, skipping the `r_ext` load, the divide, the blend, the
   min and the candidate-extraction loop.

Both are exact refactorings, not approximations: `theta_min`, `cand_j`,
`cand_alpha` and `n_admissible` are unchanged by construction. This is **not**
an inexact-linear-algebra scheme and therefore does not face the 404/404 Harris
authority standard that killed every Krylov/Jacobi/lookahead variant — there is
no decision to reproduce because no decision changes.

## Measurement methodology (LOADED-BOX doctrine)

This box is shared (load ~11–12 of 12 cores). Cross-process phase minima drift
**4–19%** between runs — larger than the effect under test. An earlier
cross-worktree min-of-15 comparison was therefore **discarded as unsound**: it
showed `ratio_test` −13.4% but every *untouched* control phase moved +4% to
+19% in the same direction.

The sound measurement is an **alternating within-process A/B**, arms strictly
interleaved `B,A,B,A,...` so contention is shared and cancels in the paired
ratio. Both arms compile the same branch (arm A evaluates `fastpath &&` and
falls through), so arm A is if anything slightly penalised — conservative.
Untouched phases are reported as controls and act as the validity gate.

Driver: `experiments/harris_alternating_ab.py`. Arm selector
`LINPROGX_DS_HARRIS_FASTPATH` is refreshed once per solve, never inside the
block loop.

## Result (greenbea, 11 alternating pairs)

Bit-identity asserted on every repetition: **4,399 pivots, objective repr
`-72557668.26492292` (reduced) / `-72555248.12984592` (original units),
residual 1.769e-07** — a single signature across all 22 solves.

| phase | B/A median | min | max | role |
|---|---:|---:|---:|---|
| **ratio_test** | **0.7637** | 0.7465 | 0.7766 | **treatment** |
| btran_rho | 1.0042 | 0.9871 | 1.0334 | control |
| ftran_col | 1.0084 | 0.9975 | 1.0325 | control |
| pivot_row | 1.0197 | 0.9967 | 1.0962 | control |
| refactor | 1.0156 | 0.9826 | 1.0672 | control |
| lu_update | 1.0033 | 0.9863 | 1.0362 | control |
| rcost_update | 1.0035 | 0.9750 | 1.0210 | — |
| **TOTAL** | **0.9778** | 0.9618 | 0.9973 | — |

- Treatment effect on `ratio_test`: **−23.63%**
- Worst control-phase drift: **1.97%** (effect is 12× the drift ⇒ not contention)
- `ratio_test` share of wall (arm A): 14.00%
- Whole-wall saving: **−2.22%** measured directly on TOTAL; −3.31% if controls
  were exactly neutral. **The conservative −2.22% is the number of record.**

## Honest accounting against the board

- Required: **−17.7013%** (whole-wall ratio ≤ 0.822987).
- Delivered here: **−2.22%**, i.e. ~1/8 of the gap.
- This does **not** flip greenbea. It is recorded as a **composition member**:
  bit-identical, global (no per-problem tuning, no threshold), certificate-
  neutral by construction, and on a slice (`ratio_test`) disjoint from the
  triangular-solve slices where most other candidate mechanisms live.

## What this does NOT claim

- It is a **local** measurement. The board is protocol v3 (Modal AWS us-west-2,
  3 hosts × 7 interleaved pairs, median-of-hosts). No v3 recertification has
  been run, so **the board remains 23W-0P-1L**.
- **CORRECTION (route census, 2026-07-25).** The woodw and degen3 A/B numbers
  above were produced by calling `solve_eq_box_dual_simplex` DIRECTLY on the
  presolved model, which bypasses public route selection. A census of all 24
  cells (`experiments/route_census.py`, artifact `/tmp/route_census.json`)
  shows that **only greenbea routes to the dual simplex publicly**; every other
  cell resolves to `native-c-sparse-ipm` or `native-c-sparse-pdhg`, including
  **woodw, which is IPM**. So:
  - The woodw −5.01% and degen3 −0.52% are valid evidence that the kernel
    change is safe and beneficial *wherever the DS runs*, but they are **NOT**
    board-cell improvements. Those cells never execute the changed code.
  - This independently confirms worker D's diagnosis of the previously
    undiagnosed four-column-owner woodw regression: woodw's exposure is the
    IPM/`LINPROGX_CHOL_SCHED` thread pool, not DS arithmetic.
  - Consequently the units touch **exactly one board cell (greenbea)** and
    carry structurally zero regression risk to the other 23 — they cannot
    change a result in code that never runs.

## Considered and rejected (with arithmetic)

**Support-driven scan** — replace the dense contiguous scan of 5,392 columns
with a gathered walk of the 1,226-column support (4.4× fewer elements).
Rejected: `rcost_update` already walks the support and costs ~26.6 cyc/entry
*gathered*, versus `ratio_test`'s ~7.2 cyc/column *contiguous*. Trading 5,392
contiguous for 1,226 gathered is 38.8k → ~33k cycles, ~15% of the phase, and it
would additionally change candidate ordering (a trajectory change requiring
certification) for less benefit than the two bit-identical early-outs already
deliver. This is a miniature of the recorded K3 result that dense sweeps of
sparse storage run 75–87× slower — the contiguous scan is the right structure.

## Reproduction

```bash
git worktree add /home/evan/dev/linprogx-harris-census -b work/harris-census main
cd /home/evan/dev/linprogx-harris-census
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
PYTHONPATH=. LINPROGX_DS_HARRIS_CENSUS=1 uv run python experiments/slice_census_today.py
PYTHONPATH=. uv run python experiments/harris_alternating_ab.py --pairs 11
```

Artifacts: `/tmp/harris_alternating_ab.json`, `/tmp/slice_census_today.json`.

No network, no solver source read, no per-problem tuning, eps=2e-5 untouched,
certificate-backed optimality preserved by bit-identity.
