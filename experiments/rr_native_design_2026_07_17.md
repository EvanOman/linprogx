---
date: 2026-07-17
topic: native-ranged-row-pdhg
status: design-before-implementation
---

# Native two-sided row bounds in sparse PDHG (realization B)

## Scope and ship/kill boundary

Realization B solves

\[
  \min_x c^T x \quad\text{s.t.}\quad \ell_r \le Ax \le u_r,
  \quad \ell_x \le x \le u_x
\]

directly in the sparse PDHG kernel. It is initially reachable only when
`LINPROGX_PDHG_RANGED=1`. The existing equality API remains valid and means
`ell_r == u_r == b`.

This design deliberately does not extend the IPM, dual-simplex, or tableau
paths. A ranged contraction must therefore route to PDHG. The kernel has two
acceptance layers which must not be conflated: `kkt_terminated` is the
certificate-backed early exit (primal feasibility, dual feasibility, and
Lagrangian gap), while the existing final status convention promotes a final
primal-feasible point to `optimal` even if the last gap check missed an eval
point. Equality behavior, including that final convention, is preserved
bit-for-bit. Ranged-row diagnostics extend the certificate layer described
below; the Python oracle/original-space gate remains mandatory before a
contracted result is exposed.

Kill rather than weaken those semantics if native ranged rows cannot meet the
certificate or original-space postsolve gates.

## PDHG derivation

Let

* `X = [ell_x, u_x]`,
* `C = [ell_r, u_r]` (a Cartesian product of row intervals),
* `f(x) = c^T x + delta_X(x)`, and
* `g(z) = delta_C(z)`.

The problem is `min_x f(x) + g(Ax)`, with saddle form

\[
  \min_x \max_y f(x) + \langle Ax,y\rangle - g^*(y).
\]

The Chambolle--Pock dual update at extrapolated primal point `x_bar` is

\[
  y^+ = \operatorname{prox}_{\sigma g^*}(y + \sigma A\bar x).
\]

Moreau's identity and `prox_(g/sigma) = Pi_C` give

\[
\begin{aligned}
v   &= y + \sigma A\bar x,\\
y^+ &= v - \sigma\Pi_C(v/\sigma)\\
    &= y + \sigma A\bar x
       - \sigma\Pi_{[\ell_r,u_r]}(y/\sigma + A\bar x).
\end{aligned}
\]

The implementation is componentwise. For
`q_i = (A x_bar)_i` and `t_i = y_i/sigma + q_i`,

\[
y_i^+ =
\begin{cases}
y_i + \sigma(q_i-\ell_{r,i}) & t_i < \ell_{r,i},\\
0                             & \ell_{r,i}\le t_i\le u_{r,i},\\
y_i + \sigma(q_i-u_{r,i})    & t_i > u_{r,i}.
\end{cases}
\]

The explicit zero in the middle branch avoids cancellation. Infinite lower or
upper endpoints work without a separate mathematical case. The existing
kernel uses `q = 2*A*x_trial - A*x`, which is exactly `A*x_bar` for its
extrapolation, so only the row prox changes; the primal box prox, line-search
interaction `dy * A(dx)`, averaging, and restart geometry do not.

### Exact equality compatibility

For an equality row, `ell_r_i == u_r_i == b_i`, projection onto the singleton
is constant:

\[
  \Pi_{\{b_i\}}(y_i/\sigma+q_i)=b_i,
\]

therefore

\[
  y_i^+=y_i+\sigma(q_i-b_i),
\]

which is the shipped update
`y + trial_sigma * (2*ax_trial - ax - scaled_b)`.

Algebraic equivalence is not enough for the compatibility gate. The C hot loop
will have an explicit equality fast path that executes the same expression in
the same evaluation order as the current code. If every row has equal bounds,
the solver will also retain the current `scaled_b` vector, RHS norms, CGLS
cleanup eligibility, and candidate/restart logic. Thus knob-on equality calls
do not merely converge to the same answer: they execute the same floating-point
trajectory. Tests compare complete result dictionaries (including `x`,
objective, iterations, restarts, diagnostics, and step-trial counts) for all
current PDHG fixtures and the full PDHG pytest suite.

## Scaling and data model

The C entry point keeps positional `(c, b, lo, hi, ...)` compatibility and adds
optional keyword-only-equivalent sequences `row_lo` and `row_hi`. Both must be
supplied together, have matrix-row length, and satisfy `row_lo[i] <= row_hi[i]`.
They are rejected unless `LINPROGX_PDHG_RANGED=1`. Omitting them aliases the
logical row bounds to `b,b` and takes the unchanged equality path.

Ruiz scaling is unchanged: `A_tilde = R A D`, `x_original = D x_scaled`, and
`y_original = R y_scaled`. Row bounds scale as

\[
  \tilde\ell_r=R\ell_r,\qquad \tilde u_r=R u_r.
\]

The equality case continues to store and use `scaled_b = Rb`. For genuinely
ranged rows, two arrays `scaled_row_lo` and `scaled_row_hi` are allocated. A
row-bound norm used only for relative progress/restart metrics is

\[
  \|rbound\|_2 = \sqrt{\sum_i
  \max(|\ell_{r,i}|\;\text{if finite},
       |u_{r,i}|\;\text{if finite})^2}.
\]

This is exactly `||b||_2` for equalities, remains finite for one-sided rows,
and does not weaken the absolute certificate gate.

The Python-side contraction result will carry `row_lo` and `row_hi` beside its
CSR matrix. Existing `SparseLPProblem.A_eq/b_eq` is unchanged; equality inputs
continue through the established presolve and solver routes. Ranged rows are an
internal reduced representation until the experiment ships.

## KKT residuals and certificate acceptance

Let `v = Ax` in original units and define signed row violation

\[
r^p_i =
\begin{cases}
v_i-\ell_{r,i} & v_i < \ell_{r,i},\\
0               & \ell_{r,i}\le v_i\le u_{r,i},\\
v_i-u_{r,i}     & v_i > u_{r,i}.
\end{cases}
\]

The reported primal residuals become

* `max_primal_residual = max_i |r^p_i|`, and
* `l2_primal_residual = ||r^p||_2`.

For equality rows these are bit-for-bit the current residuals by an equality
fast path computing `(A_tilde*x - scaled_b)/row_scale` in the same order.

Let `r = c + A^T y`. The column-bound dual violation is unchanged:

* if `r_j > 0`, a finite lower bound contributes `r_j*ell_x_j` to the dual
  objective; otherwise the violation is `r_j`;
* if `r_j < 0`, a finite upper bound contributes `r_j*u_x_j`; otherwise the
  violation is `-r_j`.

For a row interval,

\[
  g_i^*(y_i)=\sup_{z_i\in[\ell_{r,i},u_{r,i}]}y_i z_i
  =\begin{cases}u_{r,i}y_i&y_i>0,\\
                  \ell_{r,i}y_i&y_i<0,\\
                  0&y_i=0.
    \end{cases}
\]

Thus the dual objective changes from
`-sum_i b_i*y_i + sum_j min_(x_j in X_j) r_j*x_j` to

\[
 d(y)=-\sum_i g_i^*(y_i)+
       \sum_j\min_{x_j\in[\ell_{x,j},u_{x,j}]} r_jx_j.
\]

With `y_original = row_scale*y_scaled`, the row contribution is
`-row_hi[i]*y_original[i]` for positive `y`, and
`-row_lo[i]*y_original[i]` for negative `y`. Equality rows execute the old
`-b[i]*y_original[i]` expression exactly.

One-sided rows add a dual-domain check because their support function is
infinite on the wrong sign:

* `u_r_i = +inf` requires `y_i <= 0`; violation is `max(y_i,0)`,
* `ell_r_i = -inf` requires `y_i >= 0`; violation is `max(-y_i,0)`.

These row violations join the existing column violations in
`dual_residual = max(all violations)` and its L2 counterpart. As in the
current column-bound evaluator, an unavailable infinite support term is
omitted from the finite diagnostic dual objective while its nonzero violation
prevents acceptance.

The exact certificate-backed early-acceptance inequalities are therefore:

\[
\begin{aligned}
\max_i \operatorname{dist}((Ax)_i,[\ell_{r,i},u_{r,i}])
  &\le \mathtt{tol},\\
\max(\max_j \rho_j,\max_i \nu_i)
  &\le \mathtt{tol}(1+\|c\|_\infty),\\
|c^Tx-d(y)|
  &\le \mathtt{tol}(1+|c^Tx|+|d(y)|).
\end{aligned}
\]

Only the first inequality's residual definition, the row part of the second,
and the row-support term in the third change. Thresholds do not. The existing
post-loop feasibility promotion remains
`max_i dist((Ax)_i,[ell_r_i,u_r_i]) <= tol`; this is a status-semantic
compatibility requirement, not a new certificate claim. The scale-free KKT
score used for candidate choice/restarts replaces
`||Ax-b||_2/(1+||b||_2)` with
`||dist(Ax,C)||_2/(1+||rbound||_2)`; dual and gap normalizations are unchanged.

The current CGLS feasibility cleanup solves an equality correction and is not
valid for inactive interval interiors. It remains enabled for the all-equality
fast path and is skipped for genuinely ranged problems. Skipping an optional
cleanup can delay acceptance but cannot create a false `optimal` status.

## Degree-2 contraction algebra and postsolve

Consider a pivot equality containing a degree-2 column `x_j`:

\[
 a x_j+p^Tz=b,
\]

and its other incident row

\[
 L\le d x_j+q^Tz\le U.
\]

Substitute `x_j=(b-p^Tz)/a`. The other row becomes

\[
 L-(d/a)b \le (q-(d/a)p)^Tz \le U-(d/a)b.
\]

The eliminated variable bound becomes a ranged row on the pivot remainder:

\[
 \min(b-a\ell_{x,j},b-a u_{x,j})
 \le p^Tz \le
 \max(b-a\ell_{x,j},b-a u_{x,j}),
\]

with the usual extended-real interpretation. This is the primitive used to
derive and test chain composition. When parallel/identical remainder rows are
created, their intervals intersect:

\[
 [L_1,U_1]\cap[L_2,U_2]
   =[\max(L_1,L_2),\min(U_1,U_2)].
\]

If one row is a nonzero scalar multiple `alpha` of another, first divide its
interval by `alpha` (swapping endpoints when `alpha < 0`), then intersect.
An empty intersection is infeasible. A contained interval is redundant.

The pds implementation may specialize this general algebra to unit
coefficients and degree-2 chains, but every contraction record stores enough
to replay

\[
 x_j=(b-p^Tz)/a

in reverse elimination order. Postsolve therefore reconstructs eliminated
columns exactly from the pivot equality, not from a row-bound activity guess.
The final gate recomputes every original equality residual, every original
variable-bound violation, and the original objective at tolerance `2e-5`.

## Validation order

1. Unit tests for the prox (finite interval, equality, both one-sided forms),
   row residual, support-function dual objective, and wrong-sign row dual
   violation.
2. Equality-only knob-off versus knob-on complete-result bit identity on every
   existing PDHG test case and the pds_10, pds_20, qap12, qap15 fixtures.
3. Small contracted-chain property/oracle tests, including negative
   coefficients, finite and one-sided variable bounds, parallel-row interval
   intersection, and exact reverse postsolve.
4. pds_10/pds_20 objective agreement and original-space residual/bound gates at
   `2e-5`.
5. Paired foreground wall measurements. Ship only for pds_10 improvement at
   least 18%, pds_20 regression no more than 1%, and a fully green pytest run.
