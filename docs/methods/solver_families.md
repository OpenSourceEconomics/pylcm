---
title: Solver families
---

# Solver families

pylcm exposes one broad solver and a family of structural specializations.

## The map

| Economic problem                                            | Solver       | Continuous maximization                             |
| ----------------------------------------------------------- | ------------ | --------------------------------------------------- |
| General discrete-continuous regime                          | `GridSearch` | Full action-grid product                            |
| Smooth one-margin consumption-saving problem                | `EGM`        | One Euler inversion                                 |
| One liquid margin plus discrete choice                      | `DCEGM`      | EGM per discrete branch, then upper envelope        |
| Liquid inner margin plus finite outer margin                | `NEGM`       | `DCEGM` conditional on each outer candidate         |
| One liquid margin with declared non-convex budget structure | `NBEGM`      | EGM per smooth run/case, then branch-aware envelope |
| Nested outer margin with declared inner budget structure    | `NNBEGM`     | `NBEGM` conditional on outer candidates             |

The table is not a ranking. Each row solves a different declared problem class.

## Grid search is the baseline

If there are $n_j$ nodes for continuous action $j$, grid search evaluates a candidate
count proportional to

$$
N_a = \prod_j n_j
$$

at every state cell. This is expensive as action dimensions accumulate, but it makes few
structural assumptions. Dense candidates also map naturally to accelerators and can be
chunked to control memory. Grid search is exact relative to its action grids, not to the
underlying continuous choice set.

## EGM replaces search with inversion

For a smooth liquid margin, EGM chooses an exogenous post-decision savings grid and
inverts the Euler equation for consumption. That changes the dominant candidate growth
from a current-state-by-action grid to a savings-grid construction plus interpolation.
The gain comes from amortizing one inversion over current liquid states.

The gain disappears if the Euler right-hand side still varies arbitrarily with the
current liquid state after conditioning on the solver's rows. Declared intervals can
sometimes recover amortization; otherwise grid search may be the better representation.

## Envelopes recover non-concave choices

A discrete choice or a non-convex budget can produce several candidate value branches.
`DCEGM` and `NBEGM` construct those branches and take an upper envelope. Envelope
configuration affects accuracy, topology handling, memory, and accelerator suitability;
it is not cosmetic post-processing.

## Nesting avoids a coupled two-dimensional inversion

`NEGM` and `NNBEGM` condition the liquid solve on candidates for an outer durable or
illiquid post-decision state. They do not solve a genuinely coupled two-dimensional
first-order-condition system. The outer candidate count therefore multiplies the cost of
the complete inner solve, and batching or adaptive search can matter as much as the
inner algorithm.

Read the detailed method pages:

- [The endogenous-grid method](egm_foundations.md)
- [Discrete choice and upper envelopes](../explanations/iskhakov_et_al_2017.ipynb)
- [Nested endogenous-grid methods](nested_egm.md)
- [Declared non-convex budgets](nonconvex_budgets.md)
- [Scaling, memory, and hardware](performance_scaling.md)

For exact constructors and prerequisites, see
[Solvers and capabilities](../reference/solvers.md).
