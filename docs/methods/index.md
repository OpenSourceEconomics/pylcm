---
title: Concepts & Methods
---

# Concepts & Methods

This chapter explains why pylcm's representations and algorithms work. It assumes basic
finite-horizon dynamic programming and can use Bellman equations, first-order
conditions, and interpolation notation. It does not document private engine modules.

## Economic and computational representation

- [Dynamic programming and pylcm](dynamic_programming.md)
- [Functions as a dependency graph](../explanations/function_representation.ipynb)
- [Phase-dependent model structure](../explanations/phase_grammar.ipynb)
- [Preference aggregation and certainty equivalents](preferences.md)
- [Beta-delta discounting](../explanations/beta_delta.ipynb)

## Solution methods

- [Solver families](solver_families.md) gives the map.
- [The endogenous-grid method](egm_foundations.md) derives the one-margin foundation.
- [Discrete choice and upper envelopes](../explanations/iskhakov_et_al_2017.ipynb)
- [Nested endogenous-grid methods](nested_egm.md)
- [Declared non-convex budgets](nonconvex_budgets.md)
- [Scaling, memory, and hardware](performance_scaling.md)

## Approximation and uncertainty

- [Interpolation](../explanations/interpolation.ipynb)
- [Approximating continuous shocks](../explanations/approximating_continuous_shocks.ipynb)
- [Stochastic transitions](../explanations/stochastic_transitions.ipynb)

Each method page links to the declarations it requires and to runnable examples. The
[Reference](../reference/index.md) is normative when an exact contract matters.
