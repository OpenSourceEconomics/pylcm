---
title: pylcm
---

# pylcm

pylcm specifies, solves, and simulates finite-horizon discrete-continuous dynamic
choice models. You write states, actions, preferences, constraints, and transition
laws as ordinary Python functions; pylcm turns that economic description into a
backward-induction and simulation pipeline built on JAX.

Use the route that matches what you are trying to do:

::::{grid} 1 2 2 2

:::{grid-item-card} New to pylcm?
:link: getting_started/index
Install pylcm, solve a tiny model, learn its vocabulary, and decide which solver family
your model must be written for.
:::

:::{grid-item-card} Building a model
:link: user_guide/index
Define regimes, grids, shocks, transition laws, parameters, and outputs through
task-oriented guides.
:::

:::{grid-item-card} Choosing a solution method
:link: user_guide/choosing_a_solver
Start from the economic structure, then compare numerical scaling, memory, and target
hardware.
:::

:::{grid-item-card} Looking for a model?
:link: examples/index
Browse executable examples by economic feature, pylcm feature, and solver family, or
follow links to the wider model zoo and benchmark suite.
:::

:::{grid-item-card} Looking up an exact contract?
:link: reference/index
Find signatures, declarations, capability tables, runtime controls, and every public
export.
:::

::::

The [Concepts & Methods](methods/index.md) chapter explains why the algorithms and
representations work. The [Development](development/index.md) chapter is for people
changing pylcm itself.

## Scope

pylcm is for finite-horizon models with discrete and continuous states and actions,
deterministic or stochastic transitions, and potentially multiple regimes. Brute-force
grid search covers the broadest class. Endogenous-grid solvers cover narrower
consumption-saving structures and require the model to declare those structures from
the outset. GPU acceleration can make large models feasible, but the best solver still
depends on candidate growth, memory, compilation, and hardware.

If finite-horizon dynamic programming is new to you, begin with the
[QuantEcon Dynamic Programming book](https://dp.quantecon.org/) before using the
methods chapter.
