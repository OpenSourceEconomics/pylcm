---
title: Examples
---

# Examples

Examples are organized around models, not isolated API calls. Each walkthrough states
the economic question, the pylcm representation, the solver, and the scale of the
computation. Model definitions live in `lcm_examples` so the documented code is
importable and testable.

## Choose by feature

| Example                                                              | Economic feature                          | pylcm feature                                                                 | Solver                                  |
| -------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| [Tiny consumption-saving](tiny.md)                                   | Work, consume, retire                     | Basic regimes and deterministic transitions                                   | Grid search                             |
| [Mortality](mortality.md)                                            | Death risk and borrowing                  | Stochastic regime transition                                                  | Grid search / DCEGM variants            |
| [Epstein–Zin lifecycle](epstein_zin.ipynb)                           | Health and mortality risk                 | Nonlinear certainty equivalent                                                | Grid search                             |
| [Iskhakov et al. (2017)](iskhakov_et_al_2017.md)                     | Retirement and discrete labor             | Discrete-continuous upper envelope                                            | DCEGM                                   |
| [Precautionary savings](precautionary_savings.md)                    | Income risk                               | IID and AR(1) processes                                                       | Grid search                             |
| [Precautionary savings with health](precautionary_savings_health.md) | Joint wealth, health, and exercise        | Multiple states/actions and constraints                                       | Grid search                             |
| [Mahler and Yum (2024)](mahler_yum_2024.md)                          | Quantitative lifecycle health model       | Large state space and empirical inputs                                        | Specialized EGM family; GPU recommended |
| [Collective regimes](collective_regimes.md)                          | Shared choice, participation, dissolution | Stakeholder values, projected same-period values, value-dependent transitions | Grid search                             |

The tiny example is a pedagogical baseline, not the starting declaration for every
model. If your production model needs an EGM-family solver, use the specialized regime
and margin declarations from the beginning. See
[Choose your starting declaration](../getting_started/next_steps.md).

## Beyond the curated examples

The [LCM Zoo](https://github.com/OpenSourceEconomics/lcm-zoo) contains additional model
implementations. The
[LCM solver benchmarks](https://github.com/OpenSourceEconomics/lcm-solver-benchmarks)
track evolving performance and accuracy evidence. These are external destinations:
pylcm's book links to them instead of copying models or benchmark tables that would
quickly become stale.

## Reusing an example

Models export builders or regime declarations that can be imported and modified. Prefer
a builder argument or `Regime.replace(...)` over copying a complete source file. After
changing economic structure, revisit
[Choosing a solver](../user_guide/choosing_a_solver.md); a change in the budget or
continuous-choice topology may require a different declaration rather than only
different parameter values.
