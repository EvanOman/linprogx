# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

# The two-phase rule matrix: linprogx BEATS HiGHS on 25fv47's trajectory, and
# greenbea's DSE anomaly is now razor-sharp

Full 3x3 sweep of (phase-1 rule) x (phase-2 rule) over the bound-swap two-phase
structure, using the existing `EXPORT_BASIS`/`WARM_START` hooks, all cells
certified optimal unless marked. A phase-keyed rule is a **global** mechanism --
the same rule applies to every instance -- so this is not per-problem tuning.

## The headline: 25fv47 goes below HiGHS

| cell | shipped big-M Dantzig | **best two-phase** | HiGHS | vs HiGHS | vs shipped |
|---|---:|---:|---:|---:|---:|
| **25fv47** | 6,948 | **2,959** (DSE/DSE) | 3,033 | **0.976x WIN** | **-57.4%** |
| **degen2** | 1,453 | **632** (DSE/DSE) | 537 | 1.177x | **-56.5%** |
| greenbea | **4,283** | 4,829 (Dantzig/Dantzig) | 2,836 | 1.703x | **+12.7% WORSE** |

25fv47's prior loss ratio was **2.74x**. It is now **0.976x on trajectory** --
below HiGHS -- from two mechanisms that both already exist in the shipped C.
degen2 went **2.914x -> 1.177x**.

## The full matrix

### greenbea (HiGHS 2,836) -- the anomaly

| ph1 \ ph2 | dantzig | devex | exactDSE |
|---|---:|---:|---:|
| **dantzig** (2,591) | **4,829** | 4,864 | 5,014 |
| devex (5,603) | 7,855 | 7,866 | 7,503 |
| exactDSE (5,198) | 7,106 | 6,881 | 6,619 |

### 25fv47 (HiGHS 3,033)

| ph1 \ ph2 | dantzig | devex | exactDSE |
|---|---:|---:|---:|
| dantzig (4,200) | 9,418 | 7,300 | 6,244 |
| devex (2,660) | 8,567 | 6,943 | 5,208 |
| **exactDSE (1,176)** | 6,288 | 4,293 *(dual_infeasible)* | **2,959** |

### degen2 (HiGHS 537)

| ph1 \ ph2 | dantzig | devex | exactDSE |
|---|---:|---:|---:|
| dantzig (805) | 1,565 | 1,451 | 1,113 |
| devex (663) | 1,336 | 1,339 | 969 |
| **exactDSE (241)** | 1,009 | 972 | **632** |

## Two hypotheses of mine, both falsified by this sweep

**1. "The phases want different rules."** The premise was that Dantzig wins
phase 1 (2,418 vs DSE's 5,198) and DSE wins phase 2, so a mixed rule should beat
both. **False, and for an interesting reason: the recorded phase-1 numbers were
greenbea's.** On 25fv47 DSE's phase 1 costs **1,176 against Dantzig's 4,200** --
3.6x *better* -- and on degen2 **241 against 805**, 3.3x better. DSE wins phase 1
decisively on both. There is no phase/rule interaction to exploit; the best
diagonal cell (DSE/DSE) wins outright on both cells.

**2. "greenbea's DSE anomaly is an artefact of the big-M starting basis."** This
was the live lead going in. **False.** greenbea dislikes DSE in *both*
formulations and in *both* phases: big-M single-phase 4,675 vs 4,283, and
two-phase 6,619 vs 4,829. Removing big-M does not repair it -- it makes it worse.

## What greenbea's anomaly now is, stated precisely

Exact DSE improves the trajectory on **eight of the nine** simplex-routed cells,
including **greenbea's own sibling greenbeb (8,919 -> 5,633, 1.58x fewer)**. It
degrades exactly one: greenbea. And:

- it is **not a size effect** -- greenbeb is the same dimensions and prefers DSE;
- it is **not the big-M basis** -- the bound-swap formulation makes it worse;
- it is **not a phase effect** -- DSE loses in both of greenbea's phases.

greenbea is also the only cell where the **bound-swap two-phase structure itself
loses** to single-phase big-M (4,829 vs 4,283), while it is worth -56% on 25fv47
and degen2.

That is a much sharper object than "our dual simplex is worse on greenbea". Two
independent, normally-excellent mechanisms both invert on this one instance.

## Board consequence

**None yet, and this must not be overstated.** These are pivot counts on a
prototype two-phase pipeline driven through diagnostic hooks; the phase-1 solve
is a real solve and its per-pivot cost is unmeasured here. 25fv47 and degen2 are
**board-v2 candidate cells, not board cells** -- the board's only simplex-routed
cell is greenbea, and greenbea gets *worse*. Board remains **23W-0P-1L**,
greenbea ~1.494.

What it does establish is that the **class** deficit is largely solvable with
mechanisms already in the tree, and that greenbea is a genuine outlier within its
own category rather than its representative.
