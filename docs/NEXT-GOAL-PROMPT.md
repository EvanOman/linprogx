# Next /goal prompt (paste into the successor overmind session)

Continue the linprogx campaign. Read
/home/evan/dev/linprogx-perf-worktree/docs/SESSION-HANDOFF.md FIRST —
it is your complete inheritance (mission, reference index, protocol,
science) — then the tail of docs/HANDOFF.md (the dated research
ledger; everything from "PRESOLVE V2 SHIPPED" onward is the current
era, ~55 verdicts). Enter /overmind mode: you orchestrate;
falsifier-first briefs with kill criteria to codex (gpt-5.5 high;
retry gpt-5.6 periodically — it API-400s on this plan, see memory
codex-model-choice) and fresh explicit-model Claude subagents — never
SendMessage-resume a subagent from a Fable session, never run a
worker on Fable.

THE GOAL (standing): beat HiGHS head-to-head on ALL 24 LPnetlib
instances. Board of record (protocol v3, median-of-hosts, Modal AWS
us-west-2): **23W-0P-1L**. The sole loss is greenbea at 1.215
[1.208,1.235] — needs ~-18%.

CONSTRAINTS (inviolable): never read any solver's source code (papers
fine); no per-problem tuning (global mechanisms only); eps=2e-5 fixed;
certificate-backed optimality only; honest reporting; workers get NO
network (audit every log before shipping — grep the event log for
curl/wget/clone/github; a gate-passing unit was once quarantined for
this and the clean room beat it).

KEY REFERENCES (all under the perf worktree; details in the handoff's
index): docs/SESSION-HANDOFF.md (start here) · docs/HANDOFF.md (the
ledger) · experiments/greenbea_dossier_2026_07_18.md + corrections
(closed axes) · dual_phase1_derivation + phase1_predictions (the
confirmed Fenchel auxiliary + the conservation law) · probe_tomography
(the dual-Phase-1 identification) · kernel_campaign_dossier + k1
census (solves at IPC 0.30-0.60 = the hardware floor) + the
lsa/lsb/k3/k12 floor kills · int_kernel_combined (the shipped SIMD
unit) · creative_attack_dossier + c1..c6 reports (the six-for-six
C-wave kills) · pds_mechanism + the netagg/parallel-cols clean-room
ships (the pds arc) · loss_census_2026_07_16 (the census method that
produced the biggest wins) · tools/modal_bench.py (v3 certs; upload-src
after EVERY commit; never lp_qap15 in paired mode).

WHERE MY PATH SUCCEEDED (continue it): the census -> falsifier ->
staged-ship -> v3-certify pipeline flipped 8 cells in 5 days; the
integrity/audit protocol; cumulative host-conditional scoring;
multi-model fan-outs with hypotheses in tension; honest kills recorded
as first-class results. Maintain the ledger, the backup tag, the
chronicle (gh-pages + Claude artifact
https://claude.ai/code/artifact/a1eba80f-2b5d-426f-8527-a2d7f4545d3e),
and Telegram me (nanobot) at board moves.

WHERE I FAILED — greenbea, and the door I could not open: its 4,399
Dantzig pivots are its true trajectory (perturbation/pricing/basis all
optimal); its per-pivot cost sits on a measured hardware latency floor
(gathered sparse solves at 1.5k rows, IPC 0.3-0.6, immune to SIMD
width, flags, pipelining, level scheduling, alternative formats,
threading — every kill is dated); HiGHS's remaining edge is a refined
dual Phase-1 I identified behaviorally and derived mathematically
(exact b-invariance confirmed; native pivot parity achieved at 3,334,
even 2,399 from the K7 basis) but whose CONSTRUCTION COST no method
beats (~0.15-0.2s exact; approximate bases are worthless). Do NOT
re-probe these — every soft angle has a dated verdict.

YOUR OPEN DOOR (new approaches welcome where mine failed): the
reopening condition is a fundamentally different factorization/solve
science or hardware regime. Untouched directions worth YOUR OWN
fresh-eyes census before any building: batched/vectorized multi-pivot
linear algebra that changes the ALGORITHM's arithmetic (not its
schedule); GPU/wide-SIMD offload economics at this size; randomized or
iterative basis-solve substitutes with certificate-grade correction;
literature published AFTER my cutoffs on dual phase-1 construction or
latency-tolerant sparse solves; or an entirely new idea class no
thread of mine named — run your own multi-model ideation fan-out
against the dossier before trusting my framing; I corrected my own
dossier twice, and your fresh eyes may catch what I could not. If
greenbea stays closed, the board rests honestly at 23W-0P-1L —
protect it (re-cert only on code changes), keep the chronicle
current, and spend the campaign's machinery on whatever Evan aims
it at next.

SUCCESSOR AUDIT ADDENDUM (2026-07-21): a fresh Overmind pass tested
the strongest surviving claims and corrected two overbroad closures.
Do not re-probe active-bound sifting (95.5% omniscient live set,
23.9% projected slowdown), next-pivot BTRAN lookahead (4,398/4,398
leaving predictions but 15 floating-point Harris-choice mismatches),
or selected-inverse row caching (16.92% erase-all ceiling, 1.91%
optimistic charged ceiling). C2 forbids only same-pivot BTRAN/FTRAN
overlap; exact-real cross-pivot algebra exists but is not numerically
authoritative. C6 killed the tested dense tableau primal route, not
every hypothetical sparse revised-primal design. Full evidence and
the narrowed reopening condition are in SESSION-HANDOFF section 6
and the three 2026-07-21 falsifier reports.

SECOND SUCCESSOR AUDIT ADDENDUM (2026-07-22): three more fallback classes are
now measured dead; do not re-probe them. Sparse revised primal found a best
7,427-pivot public trajectory versus cold dual's 4,399; linprogx mandatory
kernel accounting is 0.643792s, and even an intentionally favorable
early-sparsity projection is 0.537669s (4.06%), above the 0.461235s board
target. Integrated exact Phase 1 reaches its best boundary at
`2050 + 10 + 2183 = 4243` pivots (3.55%); the best fused-core opportunity is
4.13%, and exact alternation bottoms at 4,328 pivots. Finally, an exact
changed-endpoint simultaneous rank-k audit established fresh-warm authority
and tested no-flip plus flip-aware k=2/3/4 faces at four checkpoints. All 12
flip-aware endpoints made zero flips; the only surviving totals were 4,404,
4,397, and 4,396 pivots versus the 3,519 funding ceiling, with public state
mismatches. This closes the tested fixed-policy face exchange—not every
conceivable rank-k algorithm. Evidence is in SESSION-HANDOFF section 7 and the
three 2026-07-22 falsifier reports. No production code changed; board remains
23W-0P-1L without recertification.

REMAINING FRONTIER CENSUS ADDENDUM (2026-07-22): do not re-probe ordinary
sequential suboptimization, the unfunded joint homogeneous-Phase-1 face rule,
frozen-last-refactor LU Krylov, or the tested matched-Jacobi correction. A
public-API screen on the exact reduced model took 4,669 PAMI pivots versus
3,309 serial pivots and was slower at both one and four threads. Cold reaches
the auxiliary-optimal face at pivot 2,060, but flipping the board still needs
a 29.44% continuation cut and the 20% gate needs 34.01%, far beyond the 3.55%
integrated and 1.61% alternating gains. On four actual greenbea bases, exact
BTRAN rows reproduce 404/404 Harris decisions while true-residual
matched-Jacobi degrees 1/2/3/4/8/16/32 reproduce 0/404 at every degree; every
sample requires exact fallback. Accelerator-resident simplex is outside the
CPU-only v3 protocol. These are scoped funding kills, not impossibility
theorems. Reopening requires a concrete richer approximate-inverse/recycled
construction with complete charge and decision authority, a new Phase-1
invariant with measured 29--34% opportunity, or explicit authority for a new
head-to-head hardware protocol. Evidence is in SESSION-HANDOFF section 8 and
the remaining-frontier and Krylov falsifier reports. No production code
changed; board remains 23W-0P-1L without recertification.
