# Dual Phase-1: independent derivation of the refined formulation (2026-07-18)

Derived from first principles by the orchestrator (Fable), from the campaign's
behavioral evidence only. No solver source consulted. Textbook duality only.

## Setup

Eq-box LP: min c'x s.t. Ax = b, l <= x <= u. For a basis B with duals
y = B^-T c_B, reduced costs d_j = c_j - a_j'y. Dual feasibility requires:
- j one-sided lower ([l_j, inf)):  d_j >= 0
- j one-sided upper ((-inf, u_j]): d_j <= 0
- j free:                          d_j  = 0
- j BOXED (finite l_j, u_j): NO persistent requirement — a bound flip
  (nonbasic at lower <-> at upper) flips the sign requirement, so boxed
  columns can always be made dual-consistent by choosing the right bound.
  Boxed columns contribute ZERO dual infeasibility.

## The Phase-1 objective and its dual

The natural Phase-1 minimizes total dual infeasibility over y:

  Phi(y) =  sum_{j in L} (a_j'y - c_j)_+        [lower-only: penalize d_j < 0]
          + sum_{j in U} (c_j - a_j'y)_+        [upper-only: penalize d_j > 0]
          + sum_{j in F} |c_j - a_j'y|          [free]

Phi is piecewise-linear convex, unconstrained in y. Write each term as
phi_j(a_j'y) and apply Fenchel duality (min_y sum_j phi_j(a_j'y) has dual
max_x { -sum_j phi_j^*(x_j) : A x = 0 }; conjugates of the hinge/abs terms
are linear-plus-box-indicator):

  phi_j = (t - c_j)_+          =>  phi_j^*(x) = c_j x + I[0 <= x <= 1]
  phi_j = (c_j - t)_+          =>  phi_j^*(x) = c_j x + I[-1 <= x <= 0]
  phi_j = |c_j - t|            =>  phi_j^*(x) = c_j x + I[-1 <= x <= 1]
  (boxed j do not appear in Phi) =>  x_j = 0 fixed

## THE RESULT — the auxiliary problem

  minimize   c'x
  subject to A x = 0
             x_j in [0, 1]   for lower-only j
             x_j in [-1, 0]  for upper-only j
             x_j in [-1, 1]  for free j
             x_j = 0         for boxed j            (*)

Strong duality: the optimal basis B* of (*) satisfies the ORIGINAL problem's
dual-feasibility sign conditions exactly (the KKT conditions of (*) ARE those
sign conditions), and min Phi = -min of (*). If min Phi = 0, B* is a
dual-feasible start for Phase-2 on the true problem. The auxiliary is
HOMOGENEOUS (b does not appear) with unit boxes and the ORIGINAL sparse c.

x = 0 is feasible for (*) — so a primal descent from the zero point is
natural, and only columns that can reduce c'x (i.e. columns interacting with
the support of c through Ax = 0 exchanges) ever activate.

## Why this explains every behavioral observation

1. COST SPARSITY (their DuPh1 explodes 5-10x when c densifies): the auxiliary
   objective is c itself; sparse c => tiny active sub-economy => few pivots.
   Dense c => everything participates.
2. dual_feasibility_tolerance moves DuPh1: it is the termination tolerance of
   the auxiliary solve.
3. Scaling moves DuPh1: scaling changes c and A, reshaping the auxiliary.
4. NO pricing/crash/Markowitz knob moves it: the auxiliary is a different LP;
   its difficulty is structural, not pricing-sensitive at this size.
5. Our boxed-bounds attempt (11,377 pivots): the artificial-bounds method
   solves a FULL problem (b != 0, huge boxes, all columns active) — a
   different and vastly harder auxiliary. The derived (*) is homogeneous,
   unit-boxed, with boxed columns FIXED — a small, clean subproblem.
6. Their phase-boundary basis transferring at 3,529 pivots: B* is a genuine
   basis of A (the auxiliary shares A), so it ports; the densification came
   from OUR pricing continuing on a foreign trajectory, not from B* itself.

## Falsifiable predictions (test BEFORE implementing)

P1 (decisive): HiGHS's DuPh1 pivot count is INVARIANT under perturbations of
    b (the auxiliary is homogeneous). DuPh2 may change; DuPh1 must not
    (beyond degenerate-tie noise). No competing hypothesis predicts this.
P2: DuPh1 scales with |support(c) restricted to one-sided/free columns|,
    not with total column count: zeroing cost entries on one-sided columns
    should shrink DuPh1 proportionally; adding cost to BOXED columns only
    should leave DuPh1 unchanged (boxed columns are fixed in (*)).
P3: Solving (*) on presolved greenbea (any correct LP method; the dense
    reference simplex suffices for a probe) and starting our DS Phase-2 from
    B* yields total pivots ~3,300-3,600 WITHOUT the foreign-basis
    densification (B* arises from our own machinery this time).

## Implementation sketch (contingent on P1-P3)

Solve (*) with existing machinery: x = 0 gives a degenerate primal-feasible
start; the auxiliary is small in practice (active economy = cost support).
Options: (a) dense two-phase simplex path as reference/probe; (b) the eq-box
DS applied to (*)'s own dual; (c) a small dedicated primal loop reusing the
sparse factorization. Then Phase-2 = the existing DS from B* with original
bounds/costs — no big-M distortion. Status semantics: min Phi > 0 at
optimality of (*) certifies DUAL INFEASIBILITY of the original (map to the
existing unbounded/infeasible statuses per LP duality; test synthetically).

## Honest scope note

The prior U-P1 kill (44th settled) spent two attempts on the ARTIFICIAL-
BOUNDS family. This derivation defines a DIFFERENT formulation family with
new supporting evidence; the reopening is recorded in the ledger with this
document as its justification. If P1 or P2 fails, the derivation is wrong
and the family closes again with the failed prediction as the tombstone.
