# Board v2 — broadening the benchmark so the classes are represented

## Why the current board is not representative

The board is 24 LPnetlib instances, and **23 of them route to the IPM or PDHG.
Exactly one (greenbea) routes to the dual simplex.** So the board measures our
IPM almost exclusively — and our IPM is exceptional. The dual simplex, which is
where we are weak, is visible in a single cell, and that cell is our only loss.

That is not a coincidence to be explained away: it is a sampling artefact that
hides a real component deficit.

## What was built

- **Instance pool expanded from 24 to 75** valid LPnetlib fixtures
  (`experiments/download_lpsuite.sh` covers only the original 24; the additional
  fetches are recorded below and should be folded into it).
- **`experiments/route_survey.py`** — classifies every fixture by route
  *without solving*, by presolving and then replicating the public route
  predicates (`AUTO_IPM_MAX_ROWS`, then `_ipm_stall_risk`). Instant and exact as
  a predictor of where the dual simplex is *attempted*.
- **`experiments/category_iters.py`** — compares linprogx and HiGHS by
  **iteration count**, which is load-invariant. This box cannot measure wall
  reliably (a single-shot run reported greenbea at 927 ms where a proper
  median-of-9 gives 377 ms), and iteration count is the actual target anyway.

**Caveat on the survey:** it predicts where the DS is *attempted*. The realised
route also depends on runtime behaviour — `pilotnov` is predicted SIMPLEX but its
DS attempt fails to certify and it falls through to the IPM; `cycle` likewise.
Realised routes come from `category_iters.py`.

## The simplex class, measured (this is the new information)

Nine instances realise the dual-simplex route. **We lose five and win four:**

| instance | lx iters | HiGHS iters | ratio | one-sided | col nnz |
|---|---:|---:|---:|---:|---:|
| 25fv47 | 8,300 | 3,033 | **2.74** | 100.0% | 5.71 |
| degen2 | 1,447 | 537 | **2.69** | 100.0% | 5.55 |
| greenbeb | 8,919 | 4,902 | **1.82** | 92.7% | 5.55 |
| greenbea | 4,399 | 2,836 | **1.55** | 93.0% | 5.55 |
| tuff | 221 | 174 | **1.27** | 95.4% | 7.26 |
| israel | 234 | 240 | 0.97 | 100.0% | 7.73 |
| fffff800 | 345 | 424 | 0.81 | 100.0% | 6.23 |
| agg2 | 274 | 534 | 0.51 | 100.0% | 6.25 |
| agg3 | 272 | 563 | 0.48 | 100.0% | 6.27 |

### Correction to an earlier claim

An earlier five-instance sample suggested losses tracked the **one-sided column
fraction** (the big-M signature). **On nine instances that correlation vanishes**:
the winners average **100%** one-sided and the losers **96.2%**. The earlier
reading was an artefact of `sierra` being the only low-one-sided instance in a
small sample.

### What does separate them: trajectory length

Every loss of real magnitude is on a **long** trajectory —
25fv47 **+5,267** pivots, greenbeb **+4,017**, greenbea **+1,563**,
degen2 **+910** — while every win finishes in **234–345** pivots (tuff, the
smallest loss, is 221 vs 174).

**Our dual simplex degrades as the trajectory lengthens.** On short runs it is
competitive or better than HiGHS; on long runs it falls behind by 1.5–2.7x. That
points at pricing quality decaying over a run — weight drift, degeneracy
handling, or candidate staleness — rather than at a wrong rule chosen up front.
It is a much sharper hypothesis than "our dual simplex is worse", and it is the
one DS2 should be built against.

## Proposed board v2

Keep the existing 24 cells (so the historical board remains comparable) and add
the **eight** additional realised-simplex cells: 25fv47, agg2, agg3, degen2,
fffff800, greenbeb, israel, tuff. That takes simplex representation from
**1/24 (4%)** to **9/32 (28%)** and makes the class measurable rather than
anecdotal.

Rules that must carry over unchanged: protocol v3 (Modal AWS us-west-2, 3 hosts
x 7 interleaved pairs, median-of-hosts), `eps=2e-5`, certificate-backed
optimality, no per-problem tuning, and `lp_qap15` excluded from paired mode
(HiGHS times out at 300 s there).

**Board v2 is not yet certified.** Everything above is local iteration counts.
Standing the board up requires uploading the new fixtures to the Modal volume
(`modal_bench.py --action upload-fixtures`) and a v3 paired run. The existing
board of record remains **23W-0P-1L** until then.

## Fixtures added beyond the original 24

```
lp_25fv47 lp_agg lp_agg2 lp_agg3 lp_bandm lp_bnl1 lp_bnl2 lp_boeing1 lp_boeing2
lp_capri lp_cycle lp_czprob lp_degen2 lp_etamacro lp_fffff800 lp_finnis
lp_forplan lp_ganges lp_gfrd_pnc lp_greenbeb lp_grow7 lp_grow15 lp_grow22
lp_israel lp_maros lp_modszk1 lp_perold lp_pilot lp_pilot4 lp_pilot_we
lp_pilotnov lp_scfxm1 lp_scfxm2 lp_scfxm3 lp_scorpion lp_scrs8 lp_scsd8
lp_sctap1 lp_sctap2 lp_sctap3 lp_seba lp_share1b lp_shell lp_ship04l lp_ship08l
lp_ship12l lp_sierra lp_stair lp_standata lp_standmps lp_stocfor2 lp_tuff
lp_vtp_base lp_wood1p
```

Source: `https://sparse.tamu.edu/mat/LPnetlib/<name>.mat`. Four downloads
returned error pages rather than `.mat` files and were deleted; `lp_nesm` is
among them and is currently missing.
