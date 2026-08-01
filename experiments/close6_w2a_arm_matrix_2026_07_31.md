# W2-A — Simplex arm matrix over the 11 simplex-routed LPnetlib cases

**Date** 2026-07-31 · **Ref** `fc2f86e` · **Worktree** `close6/w2a-arm-matrix`
**Artifacts** `/tmp/linprogx-close6/wave2/w2a/{arm_matrix.json,analysis.json,policy_verdicts.json,route_probe.jsonl}`

137 `(instance × arm)` records over 11 instances × up to 19 arms × 11 interleaved
rounds, every one certificate-gated at `eps=2e-5` in original units after
postsolve.

> **Headline.** No arm in the tree closes any of the four funding gates, and no
> arm that improves a loss survives its controls. The best certifying candidate
> is **1.26×–2.73× short** of its gate. Two structural results are new and both
> are worth more than the timings: `lp_sierra`'s simplex kernel is **4.04%** of
> its route (the cell is a routing problem, not a simplex problem), and the
> shipped aggregation gate **declines strictly fill-negative aggregations** on
> 25fv47 / cycle / fffff800 while accepting fill-positive ones on the green
> twins.

---

## 0. Baseline reproduction (VERIFY gate)

Reproduced exactly at HEAD before any arm was measured — `route_probe.jsonl`:

| cell | pivots | expected | objective | expected | route |
|---|---:|---:|---|---|---|
| lp_25fv47 | **6948** | 6948 ✓ | 5501.845888286745 | ✓ exact | shortcut |
| lp_degen2 | **1453** | 1453 ✓ | −1435.178 | ✓ exact | shortcut |
| lp_greenbeb | **4320** | 4320 ✓ | −4302260.261206587 | ✓ exact | shortcut |
| lp_sierra | **725** | 725 ✓ | 15394362.183631929 | ✓ exact | **ipm-rescue** |
| lp_greenbea | **2424** | 2424 ✓ | −72555248.12984599 | — | shortcut |

**No deviation.** `lp_sierra`'s route is confirmed by `Solution.message` as the
post-IPM-failure rescue (`sparse.py:404`), settling the open inference in W1-A §6
and W1-B §1 — `docs/BOARD-V2.md`'s nine-cell simplex list was the incomplete one.

Nine further historical counts reproduce exactly, which is a strong check that
the kernels are untouched: pre-churn 25fv47 **8300** and degen2 **1447**; exact
DSE 25fv47 **2614** and degen2 **653** (`_csparse.c:10423-10424`); greenbeb DSE
**5633** and Devex-era Dantzig **8919** (`ds2_chuzr_2026_07_26.md:220`); greenbea
pre-churn **4399** — the pivot count of the legacy trace digest
`679168a4baad36d6` — and churn-on **4283**.

## 1. Measurement protocol

The host is shared and was heavily loaded throughout (loadavg 34 → 84 on 12
cores), so **wall time is recorded as context only** and every verdict is CPU
time.

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. **This is load-bearing.** With BLAS
  threads live, `time.process_time` runs ~3.9× wall on every case after the
  first IPM, because OpenBLAS worker threads spin-wait and their spinning is
  charged as process CPU. Unpinned CPU time is not a usable statistic here.
- Arms are **interleaved within each round** and the arm order is **rotated per
  round**, so every arm sees the same load and the sign test is properly paired.
- 11 rounds; statistic = median of per-round `time.process_time`.
- Sign-test thresholds are asymmetric by design — strict about claiming a win,
  lenient about detecting control damage: a **win** needs ≥10/11 (two-sided
  p=0.0117); a **control regression** is called at ≤2/11 (one-sided p=0.0327);
  3–4/11 with a median >5% worse is reported as **suspected**, never passed
  silently.
- **Certificate gate:** status `optimal` **and** original-units max equality
  residual ≤ 2e-5 after postsolve. Anything else is `KILLED-uncertified` and its
  timings are discarded, not reported.

### End-to-end funding arithmetic

Kernel arms are timed in isolation, but gates are against the whole solve, so
each candidate is reconstructed **per round**:

```
est_r = prod_auto_cpu_r − shipped_kernel_cpu_r + candidate_kernel_cpu_r
```

This charges the candidate for presolve, postsolve, the residual check and — on
`lp_sierra` — the discarded IPM prefix. The shipped-kernel arm reproduces
production's pivot count on **all 11 cells**, so the substitution is exact.

## 2. The four losses

| cell | gate | best certifying arm | e2e ratio | sign | pivots (vs shipped) | projected board | shortfall | survives controls? |
|---|---:|---|---:|---:|---|---:|---:|---|
| lp_25fv47 | 0.275 | `fagg-ds-dse` | **0.509** | 11/11 | 2677 (0.385×) | 3.531 → **1.799** | **1.85×** | suspect (fffff800 3/11) |
| lp_degen2 | 0.268 | `ds-dse-phase1` | **0.357** | 11/11 | 653 (0.449×) | 3.620 → **1.292** | **1.33×** | **no** — cycle, fffff800 |
| lp_greenbeb | 0.705 | `agg-ds-dantzig-churn` | **0.885** | 11/11 | 6009 (1.391×) | 1.376 → **1.218** | **1.26×** | **no** — greenbea, tuff |
| lp_sierra | 0.354 | `ds-dantzig-phase1` | 0.966 | 5/11 | 725 (1.000×) | 2.737 → 2.643 | **2.73×** | not significant |

Every loss remains a loss under its own best arm. The two trajectory cells nearly
halve, which is real and large — and still nowhere near the gate.

`lp_greenbeb`'s best arm is a genuine surprise worth recording: the **legacy
Dantzig+churn kernel on the aggregated matrix beats DS2 on CPU (0.885, 11/11)
while taking 39% *more* pivots** (6009 vs 4320). greenbeb's loss is per-pivot
cost, exactly as W1-A §5 predicted, and DS2's per-pivot overhead exceeds its
trajectory advantage there. It is nevertheless killed: the same arm costs
greenbea 1.445 (1/11) and tuff 1.949 (0/11).

### lp_sierra is not a simplex problem

Stage decomposition, 9 rounds, CPU medians:

| stage | CPU (s) | share | status |
|---|---:|---:|---|
| presolve | 0.01346 | 1.27% | — |
| IPM primary | 0.39629 | 37.35% | `iteration_limit` |
| IPM floored retry (presolved) | 0.44243 | 41.70% | `iteration_limit` |
| IPM floored retry (**unpresolved**) | 0.16603 | 15.65% | `iteration_limit` |
| **dual simplex rescue** | **0.04284** | **4.04%** | `optimal` (725 pivots) |

**94.7% of the route is three IPM attempts that all hit the iteration limit and
are discarded.** The simplex that produces the answer is 4% of the cost. A
*free* simplex kernel would leave sierra at e2e ≈ 0.96 — against a 0.354 gate.
No pricing rule, ratio test, phase-1 formulation or aggregation can close this
cell; the only funded lane is the route itself (`_ipm_stall_risk` at
`sparse.py:29-56` does not fire on sierra, so it pays for a full IPM failure
before the basis method it actually needs). This confirms W1-A §8 by measurement
and closes W1-B §6's load-bearing open question.

## 3. Kills

Every global policy verdict is in `policy_verdicts.json`. **Not one closes a
gate.**

| policy | verdict | killed by |
|---|---|---|
| `dse-legacy` (leaving_rule 1→5) | **KILLED** | cycle cannot certify; fffff800 1.201 (1/11); agg3 1.015 (2/11) |
| `dse-churn-legacy` | **KILLED** | cycle; fffff800 1.193 (1/11) |
| `dse-phase1-legacy` | **KILLED** | cycle; fffff800 1.228 (2/11) |
| `dse-bfrt-legacy` | **KILLED** | cycle; fffff800 1.248 (0/11) |
| `churn-off` | **KILLED** | cycle cannot certify; 25fv47 itself regresses to 1.150 (0/11) |
| `phase1-dantzig` | **NO-EFFECT** | 0.966–1.096 everywhere, no significant win |
| `ds2-for-gate-declined` | **KILLED** | agg2 1.622, agg3 1.780, fffff800 1.235, israel 1.345 (all 0–2/11) |
| `aggregation-off` | **KILLED** | greenbea 1.400 (2/11); greenbeb 1.319 |
| `dse-on-aggregated` | **KILLED** | greenbea 2.920 (0/11) |
| `dse-churn-on-aggregated` | **KILLED** | greenbea 3.336 (0/11) |
| `dantzig-on-aggregated` | **KILLED** | greenbea 1.445 (1/11); tuff 1.949 (0/11) |
| `fillneg-gate-then-ds2` | **KILLED** | fffff800 1.204 (2/11) |
| `fillneg-gate-then-dse` | **FUNDED-WITH-SUSPECT-CONTROL** | fffff800 1.230 (3/11) unresolved at n=11 |
| `fillneg-gate-then-dantzig` | **KILLED** | cycle cannot certify under the widened gate |

### The DSE kill has moved, and greenbea is no longer the blocker

W1-B §3 item 3 argued the exact-DSE kill was a greenbea-only fact whose stated
reason no longer applies. **That is confirmed, and it does not help.** Because
the aggregation gate sends greenbea to the DS2 composition, a `leaving_rule`
change on the legacy dual simplex cannot reach greenbea at all — verified:
greenbea `prod-auto` = `agg-ds2` = 2424 pivots, identical, so its 0.986 win is
untouched by every legacy-DS policy in this matrix.

DSE is now blocked by three *different* controls:

- **`lp_cycle` cannot certify under any exact-DSE arm** — `dual_infeasible` at
  560 pivots. In production this does not corrupt the answer (the shortcut
  declines and the route falls through to the IPM), but the realised cost is the
  wasted DSE *plus* the IPM: 0.1222 + 0.3686 = 0.4907 s against the shipped
  kernel's 0.2405 s — **2.04×, 0/9**. Measured, not inferred (`cycle_dse_fallback`).
- **`lp_fffff800`** regresses to 1.193–1.248 at 0–2/11 under every DSE arm.
- **`lp_agg2` / `lp_agg3`** regress 1.01–1.22, mostly in the unresolvable band.

DSE remains a class win on trajectory (7 of 9 cells fewer pivots) and is now a
class *loss* on cost outside 25fv47/degen2/israel. Its recorded reopening
condition — a **global** trigger selecting the rule from observed solver state
(`dse_disposition_2026_07_26.md:81-87`) — is unchanged and still unmet.

### Churn is load-bearing beyond its certified win

`churn-off` does not merely regress 25fv47 (1.150, 0/11): **`lp_cycle` stops
certifying without it** (`dual_infeasible`). The shipped churn penalty is
currently the reason a control certifies at all, which was not previously
recorded.

## 4. The aggregation gate declines fill-negative reductions

W1-B §5 recorded as open *which* of the gate's two tests declines 25fv47 and
degen2. Answered — and the answer inverts the gate's intent:

| cell | presolved (r, c, nnz) | aggregated (r, c, nnz) | row cut | nnz Δ | gate | declines on |
|---|---|---|---:|---:|---|---|
| lp_25fv47 | 726, 1782, 10416 | 681, 1736, 10307 | 6.2% | **−1.0%** | reject | **ROWS** |
| lp_cycle | 892, 2244, 12616 | 826, 2178, 12053 | 7.4% | **−4.5%** | reject | **ROWS** |
| lp_fffff800 | 325, 829, 5170 | 288, 792, 5066 | 11.4% | **−2.0%** | reject | **ROWS** |
| lp_greenbeb | 1523, 3856, 23179 | 1187, 3514, 23937 | 22.1% | +3.3% | accept | — |
| lp_greenbea | 1525, 3868, 23274 | 1188, 3525, 24045 | 22.1% | +3.3% | accept | — |
| lp_tuff | 235, 543, 4049 | 128, 436, 3979 | 45.5% | −1.7% | accept | — |

`lp_degen2`, `lp_sierra`, `lp_agg2`, `lp_agg3`, `lp_israel` have **no** available
aggregation at all, so no gate change can reach them.

All three declined cells are declined by the **≥20% row-reduction** test, and in
all three the aggregation is **strictly fill-negative** — it removes rows *and*
nonzeros. Meanwhile both green twins are admitted *despite* +3.3% nnz growth.
The shipped exchange rate is rejecting free reductions and buying expensive ones.

A gate reformulated as "rows reduce **and** nnz does not grow" is a global
structural rule with no instance predicate, and it does **not** disturb the three
currently-accepted cells (their accepted and forced reductions are byte-identical
objects). Measured consequences of admitting the three new cells:

- **25fv47** → 0.509 with DSE (11/11), 0.560 with DS2 (11/11) — its best result
  anywhere in this matrix, and still 1.85× short of 0.275.
- **cycle** → 0.193 with DS2 (11/11) — a 5× improvement on a control, and the
  aggregation also **repairs** cycle's DSE certificate failure (0.608, 10/11).
- **fffff800** → 1.204–1.230 (2–3/11). This is the blocker, and at n=11 it sits
  between "regressed" and "unresolved".

This is the one lane in the matrix that is not dead. It is **not** a closure for
25fv47 and must not be reported as one; it is a gate-reformulation candidate
whose control exposure is a single cell. Two cautions: (i) composing it with DSE
for newly-admitted cells while keeping DS2 for already-accepted ones is a
*per-cell rule selection* and needs a global trigger that does not exist; (ii)
the gate is global, so cells outside these 11 on the 39-case board may also be
newly admitted — unmeasured here, and it must be swept before any ship.

## 5. What this closes

- **sierra's route** — measured, not inferred: `ipm-rescue`, simplex 4.04%.
- **Which gate test declines 25fv47** — the row test, on a fill-negative reduction.
- **The DSE kill's owner** — no longer greenbea (structurally unreachable); now
  cycle's certificate, fffff800, agg2/agg3.
- **Churn's true role** — cycle's certificate depends on it.
- **greenbeb's per-pivot diagnosis** — confirmed: more pivots, less CPU.

## 6. What this does not close

No gate. `lp_greenbeb` has **no** arm that improves it and survives its controls.
`lp_sierra` cannot be moved by any simplex mechanism that exists. `lp_25fv47` and
`lp_degen2` have large, certificate-clean, statistically unambiguous wins
available that are still 1.3–1.9× short and each killed by a control.

Per AGENTS.md and `docs/PROVENANCE.md`: `lp_greenbeb`, `lp_greenbea` and
`lp_tuff` run the DS2 composition, which is **source-informed**. Nothing here may
be reported as a clean-room result for those three cells.

## 7. Reproduction

```bash
uv sync --extra dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.
uv run python experiments/close6_w2a_route_probe.py
uv run python experiments/close6_w2a_arm_matrix.py --rounds 11 \
    --out /tmp/linprogx-close6/wave2/w2a/arm_matrix.json
uv run python experiments/close6_w2a_supplement.py \
    --matrix /tmp/linprogx-close6/wave2/w2a/arm_matrix.json --rounds 9
uv run python experiments/close6_w2a_analyze.py \
    --matrix /tmp/linprogx-close6/wave2/w2a/arm_matrix.json \
    --out /tmp/linprogx-close6/wave2/w2a/analysis.json
uv run python experiments/close6_w2a_policy.py \
    --analysis /tmp/linprogx-close6/wave2/w2a/analysis.json \
    --out /tmp/linprogx-close6/wave2/w2a/policy_verdicts.json
```

Production defaults are untouched: no file under `src/` was modified, every arm
is an env-scoped or kwarg-scoped call against shipped kernels, and the
gate-bypassed `fagg-*` arms call the existing `_maybe_aggregate` with the shipped
constants and only the accept/reject test omitted.

### Caveats

- All CPU times were taken under loadavg 34–84 on a 12-core shared box.
  Cross-instance absolute comparisons are inflated (`lp_sierra`'s production CPU
  reads 2.73 s here against 0.62 s on an idle probe); within-round paired ratios
  and sign tests are unaffected and are what every verdict rests on.
- Board projections multiply this host's e2e ratio by the Modal paired board
  ratio. They indicate direction and magnitude, not a board result — only
  `tools/modal_bench.py` decides a cell.
- `experiments/close_six_campaign_2026_07_31.md`, cited by the brief and by
  W1-B, **does not exist at `fc2f86e`** (that commit touches only
  `docs/NEXT-GOAL-PROMPT.md`). Wave-1 figures used here come from
  `/tmp/linprogx-close6/wave1/W1-{A,B}.md` and were re-derived by measurement
  rather than taken on trust.
