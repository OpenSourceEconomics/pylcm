---
title: Solver families
---

# Solver families

pylcm exposes one broad solver and a family of structural specializations.

## The map

| Economic problem                                                     | Solver       | Continuous maximization                                      |
| -------------------------------------------------------------------- | ------------ | ------------------------------------------------------------ |
| General discrete-continuous regime                                   | `GridSearch` | Full action-grid product                                     |
| Smooth one-margin consumption-saving problem                         | `EGM`        | One Euler inversion                                          |
| One liquid margin with general resources or optional discrete choice | `DCEGM`      | Euler inversion over supported branches, then upper envelope |
| Liquid inner margin plus finite outer margin                         | `NEGM`       | `DCEGM` conditional on each outer candidate                  |
| One liquid margin with declared non-convex budget structure          | `NBEGM`      | EGM per smooth run/case, then branch-aware envelope          |
| Nested outer margin with declared inner budget structure             | `NNBEGM`     | `NBEGM` conditional on outer candidates                      |

The table is not a ranking. Each row solves a different declared problem class.

## EGM-family glossary

| Acronym  | Problem shape                                                         | Required regime                  | Main algorithmic device                                                |
| -------- | --------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `EGM`    | Smooth one-margin consumption-saving                                  | `ConsumptionSavingsRegime`       | Euler inversion on a savings grid                                      |
| `DCEGM`  | One liquid margin with a general resources node or competing branches | `ConsumptionSavingsRegime`       | EGM by branch plus an upper envelope                                   |
| `NBEGM`  | One liquid margin with declared non-convex budget structure           | `ConsumptionSavingsRegime`       | EGM by smooth run/case plus a topology-aware envelope                  |
| `NEGM`   | DCEGM liquid problem conditional on a finite outer choice             | `NestedConsumptionSavingsRegime` | Complete inner DCEGM solves followed by an outer maximum               |
| `NNBEGM` | NBEGM liquid problem inside an outer choice                           | `NestedConsumptionSavingsRegime` | Complete inner NBEGM solves plus configurable outer search/aggregation |

## Grid search is the baseline

If there are $n_j$ nodes for continuous action $j$, grid search evaluates a candidate
count proportional to

$$
N_a = \prod_j n_j
$$

at every state cell. Total work therefore still covers the complete represented action
support. Eligible JIT solve-value routes—ordinary singleton hard max, collective hard
max, and singleton EV1 expected max—evaluate bounded C-order action blocks. Same-period
value references, gated-target continuations, and edge-reference mappings are supported
unchanged inputs to those blocks. Ordinary co-mapped state routes stream while
preserving device-local continuation reads. Eligible singleton folded-state routes
stream the action product at each shock node before the unchanged full-axis quadrature
reduction. Co-map intersections with separate same-period or edge-reference channels
remain dense. The streamed programs publish solve-time values (and collective
dissolution flags), not replay or policy artifacts; all simulation-policy construction
remains dense. The blockwise route does not establish a runtime or peak-memory
improvement without measurement. Grid search is exact relative to its action grids, not
to the underlying continuous choice set.

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
`DCEGM` also supplies the general resources route when plain `EGM` is too narrow. When a
discrete choice or non-convex schedule produces several branches, `DCEGM` and `NBEGM`
construct them and take an upper envelope. Envelope configuration affects accuracy,
topology handling, memory, and accelerator suitability; it is not cosmetic
post-processing.

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

## Related pages

- **Guide:** [Choosing a solver](../user_guide/choosing_a_solver.md) and
  [Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md)
- **Methods:** [The endogenous-grid method](egm_foundations.md),
  [Nested endogenous-grid methods](nested_egm.md), and
  [Declared non-convex budgets](nonconvex_budgets.md)
- **Examples:** [Curated examples](../examples/index.md)
- **Reference:** [Solvers and capabilities](../reference/solvers.md)
