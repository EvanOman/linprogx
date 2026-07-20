# K12 — Compiler / assembly audit (2026-07-19)

**Angle:** No algorithm changes. Disassemble the compiled DS hot loops, diagnose
vectorization / spills / redundant loads, then try flag- and annotation-only
variants (`-march=native`, `-ffp-contract`, PGO). Results must be **bit-identical**
(fp-changing flags forbidden). **Kill if <5% end-to-end.**

**Verdict: KILLED.** The sanctioned build already reaches its ceiling on this
workload. `-O3 -march=native -ffp-contract=off` moves the wall **~0%** (inside
noise); offline PGO **regresses ~10%**. Both preserve the exact trajectory. The
DS hot loop is a memory-latency-bound, branch-heavy sparse index-chase; the
instructions the compiler can improve are not on the critical path.

## Setup

- Host: AMD Ryzen 5 3600 (Zen 2, AVX2+FMA capable), gcc 11.4, Python 3.14.
- Fixture: `/tmp/lpsuite/lp_greenbea.mat`, solved via `algorithm="auto"` sparse
  eq-box dual simplex exactly as `experiments/suite_bench.py` (eps=2e-5,
  max_iter=50k). Harness: `experiments/k12_bench.py`.
- Trajectory reference (all builds identical): **status=optimal, iters=4399,
  obj=-72555248.12984592, residual=1.768892e-07.**
- This host runs greenbea at ~0.53s / **~120 us/pivot** vs the dossier's local
  0.42s / 90.5 us/pivot (~1.25x slower box); relative deltas are what matter.
- Measurement: `.venv/bin/python` directly (uv-run auto-rebuild bypassed),
  `taskset -c 2` pinned, best-of-N, **interleaved A/B** to cancel drift.

## Before: assembly diagnosis of the baseline (`-O3` only)

The shipped `_csparse` ext is compiled `-O3 -pthread -DLINPROGX_HAVE_BLAS` with
**no `-march`**, so gcc targets baseline x86-64 (SSE2 max). Disassembly of
`CSRMatrix_solve_eq_box_dual_simplex`:

| metric (DS loop fn) | baseline `-O3` |
|---|---|
| total instructions | 13,101 |
| branches (jcc/jmp) | 1,462 (11.2%) |
| `%ymm` (AVX) uses | **0** |
| packed-pd (SSE2 vec) | 23 (negligible) |
| FMA (`vfmadd*`) | 0 |
| scalar `movsd` loads | 952 |
| `call` sites | 438 |

Opcode profile is dominated by scalar loads (`movsd` 952), integer address
math (`mov` 3129, `lea` 450, `movslq` 306), and dense branching
(`test`/`cmp`/`je`/`jne`/`jmp`/`jle` in the thousands). The solve bodies
(`lu_ft_ftran`/`lu_ft_btran`) are **100% scalar** (`movsd`/`mulsd`/`subsd`/
`addsd`, zero packed). This is exactly the dossier's "index load -> dependent
load -> branch" sparse-C signature: low ILP, gather/scatter through index
lists that auto-vectorization cannot touch. Zero AVX is emitted despite an
AVX2+FMA host — the single biggest *codegen* headroom on paper.

## Variant 1 — `-O3 -march=native -ffp-contract=off`

Rationale: unlock AVX2/VEX codegen for whatever is vectorizable. `-ffp-contract=off`
is **mandatory for bit-identity**: with FMA available, gcc's default
`-ffp-contract=fast` would fuse `a*b+c` and change results (fp-changing → forbidden).
Reduction vectorization stays disabled without `-ffast-math`, so per-element
IEEE results are preserved.

After (same DS loop fn):

| metric | baseline | march | delta |
|---|---|---|---|
| total instructions | 13,101 | 12,949 | -1.2% |
| branches | 1,462 | 1,553 | +6% |
| `%ymm` (AVX) | 0 | **159** | AVX now emitted |
| packed-pd | 23 | 45 | +2x |
| FMA | 0 | **0** | contraction off (bit-identical) |

The compiler *did* the job: AVX appears, VEX 3-operand scalar encodings replace
2-operand SSE (368 `vmulsd`/`vaddsd`/`vsubsd`), packed work doubles, instruction
count drops. **Zero FMA confirms bit-identity.**

Interleaved A/B (best-of-5 ms per cell, `taskset -c 2`):

```
round  baseline  march
1       526.9    522.3
2       529.2    528.5
3       520.6    529.5
4       534.5    529.6
5       527.7    526.0
6       528.2    539.7
```

baseline min 520.6 / median ~528; march min 522.3 / median ~529. **Delta ~0%,
inside the ~2% run-to-run band.** Trajectory bit-identical (4399 / -72555248.12984592
/ 1.769e-07).

## Variant 2 — Offline PGO (`-fprofile-generate` → greenbea → `-fprofile-use`)

Two-stage on top of the march flags; profile collected on 5 greenbea solves
(the exact target workload). `-Wmissing-profile` clean → profile consumed.
Trajectory bit-identical.

Interleaved A/B (best-of-4 ms per cell):

```
round  march   pgo
1      537.5   590.2
2      533.4   594.7
3      543.9   595.6
4      541.9   590.0
5      540.0   591.8
```

**PGO regresses ~10%** (consistent across all 5 rounds, not noise). Profile-driven
block layout / inlining reorders the branch-heavy loop unfavorably; on a
latency-bound kernel the code-layout win PGO normally buys does not materialize
and the reordering costs.

## Why flags can't move this wall

The wall is not instruction throughput on the vectorizable slices — it is
dependent-load latency and branch resolution in the sparse index-chase. `-march=native`
successfully vectorized the compute-shaped fraction (0 → 159 AVX ops) and still
produced **0%**, because that fraction is negligible against gather/scatter +
per-element branching that no width of SIMD accelerates. This corroborates the
dossier's closed fp32-value result (cache-resident; container gains only ~1x)
and the "instruction throughput was never the bottleneck" seam framing: the
seam is real, but the *compiler* is not the lever that opens it — the loops must
be restructured (K2/K3/K4 branchless-SIMD rewrites), not merely re-flagged.

## Projection against the flip targets

- Flip needs cold-path **~54 us/pivot (-40%)** and ~-40% end-to-end.
- Baseline here: 4,399 pivots @ ~120 us/pivot.
- `-march=native`: **0%** → still ~120 us/pivot. Gap to flip: the entire -40%.
- PGO: **+10%** → ~132 us/pivot. Moves the wrong way.

Kill criterion is <5% end-to-end; best flag variant delivers ~0%. **KILLED.**

## Reproduction

```bash
# baseline vs march (interleaved A/B, pinned, bit-identical trajectory)
bash scratchpad/build_variants.sh   # builds baseline.so and marchv.so
bash scratchpad/ab2.sh
# PGO two-stage
bash scratchpad/pgo_build.sh generate && bash scratchpad/pgo_collect.sh
bash scratchpad/pgo_build.sh use && bash scratchpad/ab.sh
```

pyproject flags were reverted to the committed `-O3` after the audit; the
worktree extension is the clean sanctioned build (0 gcov symbols). No git ops,
no network, foreground only.
