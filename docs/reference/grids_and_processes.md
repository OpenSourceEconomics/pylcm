---
title: Grids and stochastic processes
---

# Grids and stochastic processes

A grid defines a variable's numerical outcome space. A stochastic process defines both
an outcome grid and its transition mechanism.

(api-lifecycle-and-categorical-grids)=

## Lifecycle and categorical grids

- `AgeGrid(start=..., inclusive_stop=..., step=...)` defines the lifecycle.
  `exact_values=...` can declare irregular ages.
- `@categorical(ordered=True|False)` creates a categorical code class. Every annotated
  field uses `ScalarInt` and receives a consecutive `jnp.int32` code.
- `DiscreteGrid(category_class=Category)` turns that category class into a state or
  action grid.
- `characteristics(category_class=Category, by_category=...)` checks a table of every
  category's Python `int`, `bool` or finite `float` characteristics, keyed by category
  name in declaration order. Columns use one scalar type; integer values fit int32 above
  its minimum. Rows set the same characteristics and remain distinct. The returned
  arrays are read-only and indexed by code, with bool, int32 or float64 dtype.
- `category_with(columns, **characteristics)` returns the int32 code of the category
  with the given characteristics, or the int32 minimum where none has them. The given
  identifying columns must tell all categories apart and have bool or signed-integer
  dtype. Float properties cannot be queried. Query values are Python ints/bools or bool
  and signed-integer arrays. See
  [Categories with characteristics](../user_guide/categories_with_characteristics.md).

The explicit `ordered` flag matters: interpolation and comparison may use ordering only
when the economics declares it.

(api-continuous-grids)=

## Continuous grids

| Class                                                                  | Declaration                                             |
| ---------------------------------------------------------------------- | ------------------------------------------------------- |
| `LinSpacedGrid(start, stop, n_points)`                                 | Uniform linear spacing                                  |
| `LogSpacedGrid(start, stop, n_points)`                                 | Uniform spacing after log transform                     |
| `IrregSpacedGrid(points)`                                              | Explicit sorted nodes                                   |
| `PiecewiseLinSpacedGrid(start, stop, breakpoints, points_per_segment)` | Linear segments with independently controlled density   |
| `PiecewiseLogSpacedGrid(start, stop, breakpoints, points_per_segment)` | Log-spaced segments                                     |
| `GridBreakpoint(value, owner=...)`                                     | Interior boundary and which adjacent segment owns it    |
| `ConditionalAction(grid, active_when, inactive_value)`                 | Action grid used only under some discrete-action values |

Piecewise grids control **grid density** at known locations. They do not declare an
economic budget kink to a specialized solver. Use [case pieces](case_pieces.md) or a
[piecewise-affine schedule](piecewise_affine.md) for that structure.

`ConditionalAction` wraps a continuous action grid. The action ranges over the grid
where every discrete action named in `active_when` takes one of its listed values, and
takes `inactive_value`, which must be a node of the grid, everywhere else. See
[Conditional actions](../user_guide/grids.md#conditional-actions).

(api-stochastic-processes)=

## Stochastic processes

Place process instances in `states` and omit their names from `state_transitions`:

| Process                          | Meaning                                               |
| -------------------------------- | ----------------------------------------------------- |
| `UniformIIDProcess`              | IID uniform draws                                     |
| `NormalIIDProcess`               | IID normal draws                                      |
| `JointNormalIIDProcess`          | Named IID Gaussian shocks sharing a covariance matrix |
| `LogNormalIIDProcess`            | IID log-normal draws                                  |
| `NormalMixtureIIDProcess`        | IID normal-mixture draws                              |
| `TauchenAR1Process`              | AR(1) discretized by Tauchen                          |
| `RouwenhorstAR1Process`          | AR(1) discretized by Rouwenhorst                      |
| `TauchenNormalMixtureAR1Process` | Mixture innovation with Tauchen-style discretization  |

`TauchenAR1Process.on_nodes(nodes=..., rho=None, sigma=None, mu=0.0)` uses supplied
physical nodes with midpoint probability bins and open tails. The axis is fixed;
unspecified law parameters are runtime operands and unit-root persistence is allowed.
`nodes` is a one-dimensional NumPy or JAX array, as are `IrregSpacedGrid(points=...)`
and `DiscreteGrid(values=...)`; lists and tuples are refused. Nodes must be finite and
strictly increasing at the working precision. Ordinary Tauchen construction still places
its own stationary grid.

The factory accepts `transition_rule="midpoint"` (default), `"density"` (row-normalized
normal densities at state nodes, computed in log space), or `"innovation_quadrature"`
(standardized innovation weights mapped linearly to adjacent state nodes, with targets
clipped to endpoints). For innovation quadrature, supply `n_quadrature` (default 9), or
both `innovation_nodes` and `innovation_weights` arrays. Explicit weights are normalized
finite nonnegative masses with positive total. Quadrature configuration is static, not a
process parameter. All rules accept `StateConditioned` scales. Continuous simulation
retains the normal law; node simulation uses the selected matrix.

Density-rule differentiation uses normalized scaled arithmetic without a host
certificate. A private `_lcm` developer/test checker can certify deliberately supplied
rows against a dimensionless absolute-plus-relative budget (`1e-12` at float64, `1e-5`
at float32), or raise `DensityDerivativeAccuracyError`. It adds no check to solve,
simulation or routine AD and is not a public model-validation API. See the
[finite-row guarantee and developer invocation](../user_guide/continuous_stochastic_processes.md).

A process owns its transition mechanism. Adding a second law under `state_transitions`
is an ambiguous declaration and is rejected.

`StateConditioned` lets process parameters depend on a categorical state. Use it when
one named process changes distribution across observable states without changing the
model's state vocabulary. The conditioning categories and parameter leaves must align
with the declared categorical grid.

Every IID process supports `fold=True` when used as a state. The solve still evaluates
utility and the action choice at every shock node, then averages the process axis with
its own weights before storing the period value. Simulation still draws and publishes
the realized shock. Folding is a storage-topology optimization with a deliberately
narrow contract: GridSearch, a singleton regime, fixed distribution parameters, linear
expectation, and no taste shocks or node-dependent downstream transition/value gate. A
state-conditioned folded shock additionally requires its conditioning state to remain
fixed on the regime's own and every incoming edge. See
[Folding an IID shock out of stored values](../user_guide/continuous_stochastic_processes.md#folding-an-iid-shock-out-of-stored-values)
for the complete boundary.

See [Grids](../user_guide/grids.md) and
[Continuous stochastic processes](../user_guide/continuous_stochastic_processes.md) for
worked construction.
