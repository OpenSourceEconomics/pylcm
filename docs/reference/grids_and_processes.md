---
title: Grids and stochastic processes
---

# Grids and stochastic processes

A grid defines a variable's numerical outcome space. A stochastic process defines both
an outcome grid and its transition mechanism.

## Lifecycle and categorical grids

- `AgeGrid(start, stop, step=...)` defines the lifecycle. `exact_values=...` can declare
  irregular ages.
- `@categorical(ordered=True|False)` creates a categorical code class. Every annotated
  field uses `ScalarInt` and receives a consecutive `jnp.int32` code.
- `DiscreteGrid(Category)` turns that category class into a state or action grid.

The explicit `ordered` flag matters: interpolation and comparison may use ordering only
when the economics declares it.

## Continuous grids

| Class                                                                  | Declaration                                           |
| ---------------------------------------------------------------------- | ----------------------------------------------------- |
| `LinSpacedGrid(start, stop, n_points)`                                 | Uniform linear spacing                                |
| `LogSpacedGrid(start, stop, n_points)`                                 | Uniform spacing after log transform                   |
| `IrregSpacedGrid(points)`                                              | Explicit sorted nodes                                 |
| `PiecewiseLinSpacedGrid(start, stop, breakpoints, points_per_segment)` | Linear segments with independently controlled density |
| `PiecewiseLogSpacedGrid(start, stop, breakpoints, points_per_segment)` | Log-spaced segments                                   |
| `GridBreakpoint(value, owner=...)`                                     | Interior boundary and which adjacent segment owns it  |

Piecewise grids control **grid density** at known locations. They do not declare an
economic budget kink to a specialized solver. Use [case pieces](case_pieces.md) or a
[piecewise-affine schedule](piecewise_affine.md) for that structure.

## Stochastic processes

Place process instances in `states` and omit their names from `state_transitions`:

| Process                          | Meaning                                              |
| -------------------------------- | ---------------------------------------------------- |
| `UniformIIDProcess`              | IID uniform draws                                    |
| `NormalIIDProcess`               | IID normal draws                                     |
| `LogNormalIIDProcess`            | IID log-normal draws                                 |
| `NormalMixtureIIDProcess`        | IID normal-mixture draws                             |
| `TauchenAR1Process`              | AR(1) discretized by Tauchen                         |
| `RouwenhorstAR1Process`          | AR(1) discretized by Rouwenhorst                     |
| `TauchenNormalMixtureAR1Process` | Mixture innovation with Tauchen-style discretization |

A process owns its transition mechanism. Adding a second law under `state_transitions`
is an ambiguous declaration and is rejected.

`StateConditioned` lets process parameters depend on a categorical state. Use it when
one named process changes distribution across observable states without changing the
model's state vocabulary. The conditioning categories and parameter leaves must align
with the declared categorical grid.

Some processes support `fold=True` to integrate the shock out of the stored value
function. Folding changes value-array topology and is not supported for collective
regimes.

See [Grids](../user_guide/grids.md) and
[Continuous stochastic processes](../user_guide/continuous_stochastic_processes.md) for
worked construction.
