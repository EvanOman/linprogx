# K7 — Native cheap auxiliary solve (2026-07-19)

## Verdict: KILLED

The native auxiliary solve is floored at **0.157s host-normalized (0.215s raw)** —
roughly 2x the 0.08s kill threshold — because the auxiliary is a genuine
~2,400-pivot LP and our kernels run at ~90us/pivot. The kill criterion `aux
wall >= 0.08s` fires. The other two gates actually PASS (see below), so the
failure is purely the pivot-bound solve cost, not basis quality.

Kill criterion (from K7): "Kill if >=0.08s or B* quality degrades (warm-start
pivots rise >3,500)." The `>=0.08s` clause fires decisively.

## Falsifier stated up front

K7 would be LIVE if our own machinery solved the derived Phase-1 auxiliary in
`<0.08s` (target `<0.05s`) while yielding a dual-feasible `B*` that warm-starts
the main greenbea dual simplex in `<=3,500` pivots. The decisive falsifier: the
auxiliary needs ~2,000 pivots on any simplex method (HiGHS: 1,958; ours: 2,418),
and pivots x our per-pivot cost (~90us) floors the solve near 0.15-0.22s. To hit
the 0.05s target the auxiliary would have to run at **20.7 us/pivot** — a ~4.4x
kernel speedup, the exact thing the campaign has repeatedly failed to find.

## Setup

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`; linprogx presolved shape
  1,525 x 3,868 x 23,274 nnz.
- Column classes (matches P3 exactly): 3,611 lower-only, 257 boxed, 0 free,
  0 upper-only.
- Auxiliary (identical to P3): `min c'x`, `Ax = 0`, `[0,1]` on the 3,611
  lower-only columns, `[0,0]` on the 257 boxed columns.
- Solved with linprogx's OWN `CSRMatrix.solve_eq_box_dual_simplex` (the eq-box
  dual simplex named in the K7 code map), not HiGHS/scipy.
- Global knob sweep only (no per-problem tuning): `leaving_rule in {1,2,3,5}` x
  `bfrt in {0,1}`, best-of-5 wall each.
- `eps = 2e-5`; certificate-backed optimality; foreground; no network, no solver
  source, no git ops.
- Driver: `experiments/k7_native_aux_solve_probe.py`; raw JSON:
  `/tmp/k7-native-aux/results.json`.
- Warm/export hooks: `LINPROGX_DS_WARM_START=1`, `LINPROGX_DS_EXPORT_BASIS=1`.

Host note: this session runs ~1.37x slower than the dossier reference (cold main
measured 0.5455s / 4,399 pivots = 124 us/pivot here vs the dossier's 0.398s /
90.5 us/pivot). Wall numbers below are given raw and normalized by the 0.398/0.5455
= 0.730 cold ratio; pivot counts are host-independent and are the durable result.

## Auxiliary knob sweep (native eq-box dual simplex)

| leaving_rule | bfrt | status | aux pivots | best wall (raw) | norm wall |
|---|---|---|---:|---:|---:|
| 1 (Dantzig) | 0 | optimal | **2,418** | **0.2154s** | **0.157s** |
| 1 (Dantzig) | 1 | optimal | 1,821 | 0.2389s | 0.174s |
| 2 (Devex)   | 0 | optimal | 18,985 | 2.247s | 1.640s |
| 2 (Devex)   | 1 | optimal | 8,524 | 1.628s | 1.188s |
| 3           | 0 | optimal | 17,947 | 3.047s | 2.224s |
| 3           | 1 | optimal | 7,500 | 1.932s | 1.410s |
| 5 (exact DSE)| 0 | optimal | 5,198 | 0.755s | 0.551s |
| 5 (exact DSE)| 1 | optimal | 3,564 | 0.912s | 0.666s |

Best arm: Dantzig, no bound-flip, 2,418 pivots, 0.2154s raw / 0.157s normalized.
`bfrt=1` cuts pivots to 1,821 but the per-pivot bound-flip cost erases the win on
wall. All arms reach objective 0 with equality residual 0.0.

## Dual-feasibility of the native B* (direct linear algebra) — CONFIRMED

Reconstructed `B*` from the exported basis (structural `A[:,j]` for `j<n`,
identity `e_{j-n}` for artificials, artificial cost 0), solved `y = B*^-T c_B*`,
`d = c - A^T y`, checked sign conditions at `eps = 2e-5` on the ORIGINAL reduced LP.

| quantity | value |
|---|---|
| basic structural / basic artificial | 1,464 / 61 |
| lower-only nonbasic sign violations (`d_j >= -eps`) | **0** |
| max lower-only violation | 0.0 |
| max abs reduced cost on a basic column | 1.46e-11 |
| dual feasible | **yes** |

The native auxiliary basis is genuinely dual-feasible for greenbea, exactly as
the P3 HiGHS-produced `B*` was. The mechanism reproduces natively.

## Warm-start quality — PASSES (and beats HiGHS's B*)

| run | pivots | wall (raw) | status | obj (original) | max eq resid |
|---|---:|---:|---|---:|---:|
| cold main (control) | 4,399 | 0.5455s | optimal | -72,555,248.1298459 | 1.77e-7 |
| warm from native aux B* | **2,399** | 0.3652s | optimal | -72,555,248.1298459 | 5.78e-8 |
| (ref) warm from HiGHS B* (P3) | 3,334 | — | optimal | — | 4.77e-7 |

The native aux `B*` warm-starts in **2,399 pivots — below both the 3,500 kill
threshold and the 3,334-pivot HiGHS reference**. Warm start accepted with 0
singular repairs, 0 identity fallback, imported bound status. So the "B* quality
degrades" kill clause does NOT fire — quality actually improves. Objective agrees
with cold to 4e-14 relative; bound violation 4.5e-13.

## Projection arithmetic against the flip targets

- Native aux wall 0.157s (norm) / 0.215s (raw) vs kill 0.08s: **~2.0x / 2.7x over**.
- vs the 0.05s target: **3.1x / 4.3x over**.
- Implied rate to hit 0.05s at 2,418 pivots: **20.7 us/pivot** vs our measured
  ~90 us/pivot floor — a 4.4x kernel speedup would be required.
- HiGHS's OWN auxiliary costs 0.1451s (P3): the 0.08s kill threshold sits BELOW
  even HiGHS's simplex auxiliary. No simplex-based auxiliary — ours or theirs —
  reaches 0.08s, because the auxiliary is an intrinsic ~2,000-pivot LP.
- Honest native pipeline: aux 0.157s + warm 0.267s (norm) = **0.42s** (raw
  0.215 + 0.365 = 0.581s = 1.46x cold, 2.42x HiGHS). Even with the improved
  2,399-pivot warm start, charging the auxiliary keeps the pipeline above cold.

## Why this is a conservation-law consequence, not an engineering miss

The auxiliary solve cost = (pivots to reach an auxiliary-optimal dual-feasible
basis) x (our per-pivot kernel cost). The pivot count is ~2,000 on both HiGHS and
our machinery — it is a property of the auxiliary LP, not the solver. Our
per-pivot cost is the same ~90us that the dossier's 47th settled law fixes. Their
product is ~0.15-0.22s. K7's `<0.05s` demands breaking the same pivots-x-us/pivot
law the whole campaign is stuck on; the alternative "minimal primal loop reusing
the sparse LU" would run the same FTRAN/BTRAN kernels at the same per-pivot cost,
so it cannot escape the floor either.

## Salvageable finding for adjacent angles

The native aux `B*` warm-starts at **2,399 pivots vs the HiGHS B*'s 3,334** — a
28% pivot reduction and a materially better-quality basis. This is a basis-QUALITY
result relevant to K8/K9 (which attack basis quality/density), decoupled from
K7's dead aux-cost question: our own auxiliary machinery yields a better start than
the foreign one — it is only the cost of GETTING there that kills K7.

## Correctness gates

All auxiliary arms optimal at objective 0, aux equality residual 0.0. Cold and
warm main both optimal, agreeing on the original objective to 4e-14 relative,
max original equality residual <= 1.77e-7, max bound violation <= 4.5e-13, all
within `eps = 2e-5`. Dual-feasibility of the native B* verified with 0 violations.

**Verdict: KILLED** — native auxiliary solve 0.157s (norm) / 0.215s (raw) >= the
0.08s kill threshold. Dual-feasibility and the `<=3,500` warm-pivot quality gate
both pass (2,399 pivots, better than HiGHS's B*), but the pivot-bound solve cost
cannot reach target under the settled per-pivot law.
