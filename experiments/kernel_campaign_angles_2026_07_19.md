# Kernel campaign — the twelve angles (2026-07-19)

Each agent executes ONE angle. Read the dossier first
(experiments/kernel_campaign_dossier_2026_07_19.md); its RULES and
DELIVERABLE sections bind every angle.

## K1 — Instruction-throughput census (the master measurement)
Measure IPC, branch-miss rate, and vectorization of the DS hot loops on BOTH
trajectories (cold and B*-start). Try `perf stat`/`perf record` first; if
perf_event_paranoid blocks it, fall back to rdtsc-based cycle counters and
instruction-count estimates via env-gated instrumentation. Attribute per
slice (PRICE scan, BTRAN, FTRAN, ratio, rc-update). VERDICT: per-slice
headroom table (achieved IPC vs ~4 ceiling; branch-miss %; vector ratio).
LIVE if any >=20%-of-wall slice runs under IPC ~2 or >5% branch misses.

## K2 — Branchless SIMD PRICE/pivot-row scan
Rewrite the pivot-row/PRICE scan (24.8% of wall) as branchless AVX2 for the
dense regime: contiguous loads over the dense alpha/row vectors, masked
compares, horizontal max/argmax reductions. Env-gated; trajectory must be
IDENTICAL (same selections, exact fp64 compares — SIMD reorder of compares
must not change argmax ties: define deterministic tie-breaking identical to
scalar). Kill if <25% on the scan slice or any trajectory deviation.

## K3 — Dense-mode SIMD BTRAN/FTRAN bodies
Beyond the killed dense-U (U-solve only): when solve density > global
threshold, run the FULL solve bodies (L and U phases, eta applications) as
contiguous dense sweeps with AVX, skipping index lists. Exact arithmetic
order may change: gate on identical pivot counts + objective 1e-9 + residual
tolerances. Test BOTH trajectories. Kill if <15% on BTRAN+FTRAN combined.

## K4 — Branchless/SIMD Harris ratio test
The two-pass Harris ratio test (14.9% of wall) is branch-heavy. Vectorize:
masked eligibility, SIMD min-reductions for pass-1 bound, vectorized pass-2
argmin with scalar-identical tie-breaking. Trajectory identical required.
Kill if <25% on the ratio slice or any deviation.

## K5 — Scan+update fusion
PRICE scan (24.8%) and reduced-cost update (9.7%) both sweep the same dense
vectors in separate passes. Fuse into one pass (update-then-scan) touching
the data once. Trajectory identical required. Kill if <20% on the combined
slice.

## K6 — Software prefetch + eta-list layout
For the sparse-front paths that remain: software prefetch (tuned distance)
on index-chased loads; restructure FT eta lists to struct-of-arrays with
alignment. Byte-identical results required (pure layout/prefetch). Kill if
<8% end-to-end on either trajectory.

## K7 — Native cheap auxiliary solve
The derived Phase-1 auxiliary (min c'x, Ax=0, unit boxes; see
phase1_predictions_2026_07_18.md) cost 0.145s via throwaway scipy. Engineer
it: solve with our own machinery (the eq-box DS on the auxiliary, or a
minimal primal loop reusing the sparse LU) targeting <0.05s with the SAME
B* (verify dual-feasibility of the result by direct linear algebra as the
probe did). Kill if >=0.08s or B* quality degrades (warm-start pivots rise
>3,500).

## K8 — Auxiliary refinement: boxed columns participate
P2b evidence: HiGHS's DuPh1 responds to boxed-column costs — their
auxiliary likely includes boxed columns with flip-freedom ranges. Extend
the auxiliary (boxed j: x_j in [-1,1] with cost c_j) and variants; measure
whether the refined B* yields FEWER warm-start pivots (<3,334) and/or a
LESS DENSE trajectory (us/pivot <113). This attacks basis QUALITY. Kill if
neither improves.

## K9 — Hybrid start: density shaping
The cold crash gives cheap-but-many pivots; B* gives few-but-dense. Probe
interpolations: slack-biased variants of B* (replace the k densest
structural columns of B* with their slack counterparts, k swept globally)
searching for a start with pivots ~3,600-3,900 at us/pivot ~95-100 ->
0.34-0.37s. Kill if the pivots-x-density product never drops below 0.37s.

## K10 — Threaded PRICE scan
The DS is single-threaded. The PRICE scan over 3,868 columns is
data-parallel. Prototype 2/4-thread scan with per-thread argmax + scalar
merge (deterministic result identical to serial). The a2 lesson: fork-join
overhead kills small kernels — measure the crossover honestly on greenbea's
sizes (persistent thread pool, not per-call spawn). Kill if <15% on the
scan slice at any thread count.

## K11 — fp32-compare PRICE with fp64 confirm
Distinct from the killed fp32-value rounding: keep all VALUES fp64; do the
scan's magnitude COMPARISONS in fp32 (halved compare bandwidth /
double-width SIMD) to select top-k candidates, then re-verify the winner
among top-k in fp64 (exactness preserved if k covers fp32 compare error —
derive the bound). Trajectory identical required. Kill if <15% on the scan
slice or the fp64-confirm overhead eats the gain.

## K12 — Compiler/assembly audit
No algorithm changes: objdump the compiled hot loops; check vectorization,
register spills, redundant loads; then try flag/annotation-only variants
(-O3/-march=native if not already, restrict pointers, alignment/assume
attributes, PGO if practical offline). Results must be bit-identical (flags
that change fp arithmetic, e.g. -ffast-math, are FORBIDDEN). Report the
before/after assembly diagnosis + measured wall. Kill if <5% end-to-end.
