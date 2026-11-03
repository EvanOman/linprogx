# Provenance of the linprogx board — the clean-room boundary

**This document exists so the story cannot be told wrong, by anyone, including
us.**

## The one-sentence version

**linprogx beat HiGHS on 23 of the 24 LPnetlib instances without ever reading
any solver's source code. It could not crack the 24th (greenbea). Only then, on
2026-07-25 and by explicit owner authorisation, did we read the HiGHS
implementation to understand how it solves this problem — and any subsequent win
on greenbea is a source-informed win, not a clean-room one.**

That distinction is permanent. It is not a footnote.

## The boundary

| | |
|---|---|
| Clean-room era | campaign start → commit `4662d63` |
| Boundary tag | `clean-room-boundary-2026-07-25` |
| Source-informed era | branch `greenbea-source-informed`, every commit after `4662d63` |
| Authorised by | Evan Oman, 2026-07-25, explicitly and in writing |

Every commit reachable from `clean-room-boundary-2026-07-25` was produced
**without any solver source being read**. That is verifiable: it is the entire
history up to a named tag.

## What was achieved clean-room, and must always be credited as such

- **23 of 24 LPnetlib cells beaten head-to-head against HiGHS**, protocol v3
  (Modal AWS us-west-2, 3 hosts × 7 interleaved pairs, median-of-hosts).
- greenbea driven from **14x at campaign origin → 1.215 → ~1.156**, the last
  step certified 2026-07-25 (Harris dead-division early-outs + narrow CSR index
  cache, −4.89% on-host, bit-identical).
- The complete research ledger in `docs/HANDOFF.md` — roughly seventy dated
  verdicts, ships and kills alike.

**None of that is diminished by what follows.** It was done blind, against a
mature solver, and it stands.

## What could NOT be achieved clean-room

greenbea. After a full cycle-level audit of every phase of the dual simplex —
twenty closed lanes, six falsified idealised models, a certified −4.89% — the
honest verdict was recorded as **"no identified funded mechanism"** for the
remaining 13.46%. The relevant known fact was that HiGHS reaches optimality in
roughly **3,334 pivots to linprogx's 4,399**, via a refined dual Phase 1 that we
identified *behaviourally* and derived *mathematically* (exact b-invariance
proven by Fenchel duality) but whose **construction cost we could never beat**.

We knew *what* HiGHS was doing. We could not work out how it does it cheaply.

## The rule that was lifted, and the rules that were not

**Lifted, for greenbea only, 2026-07-25, by the owner:**
> *"Never read any solver's source code."*

**NOT lifted, and still binding:**
- No per-problem tuning — global mechanisms and thresholds only.
- `eps=2e-5` fixed, never loosened.
- Certificate-backed optimality only; every accepted answer re-certifies in
  original units.
- Honest reporting; kills are first-class results.
- **No verbatim copying.** HiGHS is MIT and linprogx is MIT, so copying would be
  legally permissible with attribution — we are choosing not to. The standard is
  *understand the algorithm, reimplement it independently*. Anything closely
  derived gets explicit attribution in the source and here.

## How source-informed work is recorded

Every mechanism that used knowledge obtained from reading HiGHS carries, in its
commit message and its `experiments/` report, the line:

```
PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
```

Any mechanism developed on this branch **without** using that knowledge is
marked `PROVENANCE: CLEAN-ROOM (independent)` and says why it qualifies.

## Required framing for any public account

Any README, chronicle, artifact, talk or post describing this campaign must
state both halves. The accurate framing is:

> Evan's AI beat a mature open-source solver on **23 of 24** LPnetlib instances
> under a strict clean-room rule: never read another solver's source. On the
> 24th (greenbea) it could not close the gap — it drove it from 14x to 1.156x
> and then documented, with measurements, that it had no funded mechanism left.
> At that point the clean-room rule was deliberately lifted for that one
> instance, the HiGHS implementation was read, and the remaining gap was closed
> with that understanding.

The following framing is **false** and must never be used:

> ~~Evan's AI beat open-source solvers on all 24 cases.~~

Unqualified, that sentence claims a clean-room result for a cell that is not
one.

## Why we are recording it this way

The campaign's product was never only the board. It was the ledger, and the
ledger's value is that it is honest — it records kills as loudly as ships, and
it has repeatedly corrected itself, including against its own interest. A board
that quietly launders a source-informed win into a clean-room record would
retroactively devalue the twenty-three cells that were genuinely earned blind.

The 23 are worth more than the 24th. This document protects them.
