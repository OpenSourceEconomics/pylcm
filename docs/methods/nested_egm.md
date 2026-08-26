---
title: Nested endogenous-grid methods
---

# Nested endogenous-grid methods

A model with liquid wealth and a durable or illiquid stock has two continuous choices.
pylcm's nested solvers exploit a particular structure: conditional on the outer
post-decision stock, the remaining problem is a one-dimensional consumption-saving
problem.

Let $d'$ denote the outer post-decision choice. A nested solve computes

$$
W_t(x,d') =
\max_c \left\{
u_t(x,c,d') + \beta\mathbb{E}[V_{t+1}(x')]
\right\}
$$

with a liquid EGM step, then compares the outer candidates:

$$
V_t(x) = \max\left\{
W_t^{\mathrm{keep}}(x),
\max_{d' \in \mathcal D} W_t(x,d')
\right\}.
$$

The keeper branch represents no adjustment; the adjuster branch searches candidate outer
levels. This is nesting, not a coupled two-dimensional Euler inversion
{cite}`druedahl2021`.

## `NEGM`

`NEGM(inner=DCEGM(...), outer_grid=...)` performs one complete inner `DCEGM` solve for
each outer-grid node and compares those candidates with the keeper. The outer solution
is exact relative to that finite candidate set. `outer_batch_size` limits how many
candidate values are evaluated at once. It can reduce temporary evaluation memory, but
it does not cap the candidate bank retained for the exact later comparison. Peak memory
can therefore still grow with the complete outer candidate set.

Use it when the liquid problem has ordinary discrete-continuous non-concavity and the
outer action can be represented by a fixed grid.

## `NNBEGM`

`NNBEGM(inner=NBEGM(...), outer_search=...)` preserves declared kinks, jumps, and hard
constraints inside every outer candidate. Its outer search is itself configurable:

- `FiniteOuterGrid` is exact relative to a fixed candidate grid.
- `AdaptiveOuterMesh` evaluates exact inner solves on a shared mesh, validates the
  interpolant, and refines bracket-local candidates. Without an asserted Lipschitz
  bound, its validation is mesh-relative rather than a proof against arbitrarily narrow
  unseen peaks.

`NNBEGM` can also change how keeper and adjuster values aggregate. The default
`DeterministicOuterMaximum` takes the hard maximum. `UniformObservedFixedCost`
analytically integrates a narrowly specified observed uniform adjustment-cost shock.

Simulation replays that solve-time candidate bank. Current `NNBEGM` therefore requires
every declaration affecting replay to be phase-invariant by object identity: a bare
declaration or identical-object `Phased(solve=f, simulate=f)` is accepted, while genuine
phase variation requires `GridSearch`. This is a candidate-set restriction, not a
general restriction on `Phased`; see
[NNBEGM replay capability](../reference/solvers.md#nnbegm).

## What the model must declare

A `NestedConsumptionSavingsRegime` supplies:

- a `LiquidMargin`: liquid state, consumption action, resources, savings;
- an `OuterContinuousMargin`: outer state, outer action, post-decision state, and
  no-adjustment map;
- a compatible two-margin solver.

Use `outer_unchanged` when the no-adjustment map is literally the identity. Use
`NetOfAdjustmentCost` when resources are the difference between a before-cost node and
an adjustment-cost node. These declarations eliminate identity wrapper functions whose
only purpose would be renaming.

The exact fields and composition rules are in
[Consumption-saving regimes and margins](../reference/consumption_savings.md). Outer
strategy contracts are in
[Outer search and branch aggregation](../reference/outer_search.md).

## Computational implication

If the inner solve costs $C_{\text{inner}}$ and the finite outer search has $N_d$
candidates, the leading work is proportional to $N_d C_{\text{inner}}$. State cells,
discrete branches, stochastic nodes, and envelope candidates live inside
$C_{\text{inner}}$. This multiplication is why streaming and accelerator occupancy
matter, and why a large GPU should be fed concurrent independent work where memory
permits.

See [Scaling, memory, and hardware](performance_scaling.md). Empirical break-even points
remain model- and hardware-dependent; the external benchmark suite is the evolving
evidence source.

## Related pages

- **Guide:**
  [Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md)
- **Methods:** [Solver families](solver_families.md) and
  [The endogenous-grid method](egm_foundations.md)
- **Example:** [Mahler & Yum (2024)](../examples/mahler_yum_2024.md)
- **Reference:**
  [Consumption-saving regimes and margins](../reference/consumption_savings.md),
  [Solvers and capabilities](../reference/solvers.md), and
  [Outer search and branch aggregation](../reference/outer_search.md)
