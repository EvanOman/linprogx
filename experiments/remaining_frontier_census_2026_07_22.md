# Remaining greenbea frontier census (2026-07-22)

## Verdict

**NO FUNDED CPU MECHANISM.** After the second successor audit, three independent
clean-room lanes rechecked the remaining algorithm, factorization, and hardware
frontiers. Sequential dual suboptimization, joint Phase-1 merit, multicore
triangular solves, and certificate-corrected iterative basis solves all fail
before production implementation. Accelerator-resident simplex is outside the
current CPU-only v3 hardware protocol.

This is a funding verdict, not a theorem that future algorithms cannot beat the
board. The certified board remains **23W-0P-1L**, with greenbea at 1.2150867 and
a required whole-wall reduction of 17.7013%. The campaign's characterization
gate remains 20%.

## Sequential suboptimization is distinct but unfunded

The 2026-07-18 P-A probe tested scalar-trajectory shadow panels; the 2026-07-22
rank-k probe tested simultaneous changed-endpoint face exchanges. Classical
dual suboptimization is formally different: it performs exact sequential minor
pivots against a frozen major basis, then applies a collective update.

The public [Huangfu-Hall PAMI paper](https://arxiv.org/abs/1503.01889) also
states the central risk: suboptimization is generally an inferior pivot rule
and requires quality cutoffs. Existing black-box evidence on the exact
linprogx-reduced greenbea model already shows 4,669 PAMI pivots versus 3,309
serial pivots. A fresh public-API timing screen with highspy 1.14.0 confirmed
the same deterministic paths:

| public strategy | threads | pivots | median wall, 7 runs |
|---|---:|---:|---:|
| serial dual | 1 | 3,309 | 0.384212s |
| PAMI | 1 | 4,669 | 0.908128s |
| serial dual | 4 | 3,309 | 0.380202s |
| PAMI | 4 | 4,669 | 0.659879s |

All retained runs were `Optimal` with maximum original equality residual
`6.138e-8`. This does not project another implementation's wall onto linprogx;
it is corroborating evidence that the changed path cannot fund a native build.
Together with the measured 1.281-pivot width-four scalar-panel survival and the
failed exact rank-k endpoints, sequential suboptimization has no remaining
funding invariant.

## Joint Phase-1 merit collapses to another row rule

For the exact homogeneous auxiliary, `x=0` is primal feasible for every
nonsingular basis. Its only basis-dependent signal is reduced-cost sign
compatibility:

```text
D_H(B) = sum_{j in auxiliary-at-LO} max(0, -r_j).
```

The cold trajectory reaches `D_H(B)=0` at pivot 2,060. The one genuinely
distinct joint rule would forbid later pivots that leave this
auxiliary-optimal dual face. But the ordinary continuation still takes 2,211
pivots. Merely flipping the board would require at most 1,560 remaining pivots,
a 29.44% continuation cut; the 20% gate allows at most 1,459, a 34.01% cut.

The closest exact measurements improve total pivots only 3.55% for an
integrated boundary and 1.61% for block alternation, while all five tested
local leaving rules lose to Dantzig. Auxiliary-face preservation is
certificate-compatible but has no evidence for the missing 29--34%, so it
does not earn even a diagnostic implementation.

## Factorization and four-vCPU hardware frontier

K1 measures 182.248ms of FTRAN+BTRAN in a 526.448ms cold wall, or 34.62%.
C1's comparable current solve pair is 47.425us. A 20% whole-wall win from
solves alone therefore requires at most 20.03us per pair; the older favorable
36.8% share allows 21.66us.

True Forrest-Tomlin is already the exact recycled-factor algorithm. Ordinary
stale-LU Krylov adds a stale triangular solve and an explicit basis residual
matvec to work the current method already performs exactly. LU update and
refactorization are only 34.362ms and 34.120ms, so deleting either alone cannot
fund the board gap.

Exact multicore level scheduling also has no budget. Two cores have an absolute
20%-gate ceiling of 18.4% whole wall even at perfect 2x solve speed and zero
overhead. The live U factor averages 57.68 levels, reaches 154, and 34.18% of
FTRAN level labels change during updates. A favorable three-core model leaves
only 4.16us for roughly 146 synchronization stages plus dynamic schedule
maintenance.

The Modal protocol allocates four CPU vCPUs and no accelerator. Recent primary
work on [GPU fine-grained domain decomposition](https://arxiv.org/abs/2508.04917),
[48-core elastic triangular scheduling](https://arxiv.org/abs/2607.02324), and
[GPU sparse approximate inverses for SPD systems](https://arxiv.org/abs/2510.27517)
does not supply an admissible four-vCPU method for a nonsymmetric basis that
changes every pivot. Moving both solvers to a new accelerator protocol would be
a new campaign, not a greenbea production change under v3.

## Final arithmetic longshot

The only remaining CPU arithmetic idea was a factor-free sparse approximate
inverse or fixed low-degree polynomial, guarded by FP64 residual bounds and
exact fallback. It was tested on actual greenbea bases and the actual next
FTRAN/BTRAN right-hand sides at pivots 512, 1,536, 3,072, and 4,096.

The detailed evidence is in
`experiments/krylov_basis_solve_falsifier_2026_07_22.md`. The decisive results
are:

- basis nonzeros: 7,022 / 7,138 / 7,116 / 7,522;
- full-call `B*x` plus `B^T*y` residual-pair medians of 25.056--30.695us;
  these are contextual observations only because Python-call subtraction was
  unstable between reruns and supplies no timing lower bound;
- exact BTRAN rows reproduce the native entering choice for **404/404** Harris
  tests, while matched-Jacobi true-residual correction at every recorded
  degree 1/2/3/4/8/16/32 reproduces it in **0/404** tests;
- at degree 32 the four FTRAN/BTRAN relative-residual pairs are
  0.977662/0.742233, 0.336399/0.997509, 0.681554/0.866353, and
  0.569653/0.917206; BTRAN has already stagnated by degree two; and
- every sampled degree therefore requires the current exact solve as fallback,
  making the fully charged path a strict regression independent of timing.

This kills ordinary frozen-refactor LU Krylov and the tested matched-Jacobi
recurrence through degree 32. It does not prove that every learned sparse
approximate inverse or richer recycled space is impossible; any such proposal
still needs a concrete globally fixed construction, complete charge, and
decision-authority evidence. No diagnostic C or production implementation is
justified.

## Scoped closure

The remaining funded frontier is empty under the current facts: fixed
four-vCPU hardware, global policies, fixed `eps=2e-5`, exact status semantics,
and certificate-backed optimality. Reopening requires new evidence for one of:

1. a factor/solve arithmetic method whose complete authoritative pair is below
   20.03us on these changing bases;
2. a new Phase-1 construction or pivot invariant with a demonstrated 29--34%
   continuation-count opportunity; or
3. an explicitly new head-to-head hardware protocol.

No production source changed, so no v3 recertification is warranted.
