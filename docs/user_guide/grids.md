---
title: Grids
---

# Grids

Grids define the outcome space for state and action variables — what values they can
take. They are passed via the `states` and `actions` mappings on a
[Regime](regimes.ipynb).

A grid is ordinarily the same for every period the regime is active. For a continuous
state, see [Age-specialized functions and grids](age_specialized.md) to let the grid's
bounds or node values vary with age instead (e.g., an age-dependent borrowing limit).

> **Grid breakpoints are numerical, not economic metadata.** A piecewise grid places
> nodes around a known location. It does not tell `NBEGM` that the budget has a kink,
> jump, or hard constraint. Declare that structure separately with
> [case pieces](../reference/case_pieces.md) or a
> [piecewise-affine schedule](../reference/piecewise_affine.md).

## Quick Reference

| Grid Type                | Use Case                  | Key Parameters                                       |
| ------------------------ | ------------------------- | ---------------------------------------------------- |
| `DiscreteGrid`           | Categorical choices       | `category_class` or `values`                         |
| `LinSpacedGrid`          | Evenly spaced continuous  | `start`, `stop`, `n_points`                          |
| `LogSpacedGrid`          | Log-spaced continuous     | `start`, `stop`, `n_points`                          |
| `IrregSpacedGrid`        | Custom point placement    | `points` or `n_points`                               |
| `PiecewiseLinSpacedGrid` | Dense in some regions     | `start`, `stop`, `breakpoints`, `points_per_segment` |
| `PiecewiseLogSpacedGrid` | Log-dense in some regions | `start`, `stop`, `breakpoints`, `points_per_segment` |

All grid classes are imported from `lcm`:

```python
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    IrregSpacedGrid,
    GridBreakpoint,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
```

## Discrete Grids

### DiscreteGrid

For categorical variables. Requires a `@categorical` frozen dataclass defining the
categories:

```python
from lcm import DiscreteGrid, categorical
from lcm.typing import ScalarInt


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


actions = {"labor_supply": DiscreteGrid(category_class=LaborSupply)}
```

Values are integer codes (0, 1, 2, ...) auto-assigned by `@categorical`. In simulation
output, labels are preserved via pandas Categorical.

When each category is a combination of characteristics, such as a labour-market status
that fixes job holding and benefit duration, see
[Categories with characteristics](categories_with_characteristics.md).

When used as an **action**, no further configuration is needed. When used as a
**state**, the transition is specified via `state_transitions` on the `Regime` — see
[Transitions](transitions.ipynb).

### Discrete grids with numeric values

When the outcomes of a discrete variable are numbers, such as hours of work or years of
experience, declare the numbers instead of a category class:

```python
import jax.numpy as jnp

from lcm import DiscreteGrid

actions = {"hours": DiscreteGrid(values=jnp.array([0, 20, 40]))}
states = {
    "experience": DiscreteGrid(values=jnp.arange(31)),
    "productivity": DiscreteGrid(values=jnp.array([0.5, 1.0, 2.0]), tolerance=1e-9),
}
```

`values` is a one-dimensional NumPy or JAX array with an integer or floating dtype, so
the array's dtype decides whether the grid holds integers or floats; a list, tuple or
`range` is refused. Values are strictly ascending and finite. Inside pylcm the variable
is indexed by the codes `0..n-1` of its values, but every function you write receives
the values: `hours` arrives as `0`, `20` or `40`, not as `0`, `1` or `2`, and a
deterministic law of `experience` returns the next year of experience, which pylcm maps
back to its code. Integer values arrive as int32, as codes do, so `jnp.log(hours)` on an
integer grid is float32 even when x64 is enabled; write `jnp.log(hours * 1.0)` or
declare float values if you need the model's float precision.

A function that needs the position of a node, for example to index a parameter array,
reads `<name>_code` next to or instead of `<name>`: `hours` is the value (in pandas'
terms the category) and `hours_code` the int32 position (the code). Index arrays with
the code, `premium[hours_code]`; indexing with the value, `premium[hours]`, is refused
when the model is built. `<name>_code` exists for every discrete variable, so for a
category class it equals `<name>`. It can be read wherever `<name>` can: in functions,
constraints, state transitions and regime-transition laws. No state, action or function
may be named `<name>_code` for a discrete variable `<name>`.

A value matches a node only if it equals it. Integer grids match integers exactly. A
float grid matches within its absolute `tolerance` (zero by default), and
`grid.nearest(x)` snaps a float that arithmetic left between nodes.

Every place where you hand values to pylcm or read them back uses values by default, and
integer codes when the call passes `use_categorical_codes=True`:

- `SimulationResult.to_dataframe()` gives an ordered pandas Categorical whose categories
  are the grid's values, all of them, observed or not. Cast with `.astype(float)` before
  arithmetic, as in `df["hours"].astype(float).mean()`.
- Initial conditions take the values, as a NumPy, JAX or pandas array, or as the numeric
  Categorical `to_dataframe` publishes; its categories must be exactly the grid's values
  in the same order. A value on no node raises an error naming it.
- A `pd.Series` in `params` indexed by a value-grid state uses the values as its index
  labels (for example a level `experience` with labels `0, 1, ..., 30`), matched like
  initial conditions. String spellings such as `"0.5"` are refused.
- `Model.lookup_policy` takes states in values, restricts discrete actions through
  `action_grids` given in values, and returns a value-grid action as a plain array of
  its values.
- `SolutionResult.value_array()` and `value_frame()` carry a value-grid state's values
  as coordinates, so `.sel(experience=4)` selects a node by its value.

The one exception is `Model.simulate(shock_paths=...)`. A shock path holds category
codes even for a value grid, whatever `use_categorical_codes` says, because a path is
replayed exactly and losslessly: a code names exactly one node, while a float value
would need the grid's tolerance to find it. Map codes and values with the grid itself:

```python
import jax.numpy as jnp

grid = DiscreteGrid(values=jnp.array([0.5, 1.0, 2.0]))

grid.values[2]  # 2.0, the value of code 2
grid.encode(jnp.array([0.5, 2.0]))  # the codes 0 and 2, as int32
grid.decode(jnp.array([0, 2]))  # the values 0.5 and 2.0
```

A value-grid column that `to_dataframe()` published becomes a shock path by its codes,
`df["productivity"].cat.codes`.

## Continuous Grids

### Values outside a state grid

Continuous grids accept `off_grid=` with an explicit meaning:

| Rule                      | Continuation or policy lookup         | Economic next state                                |
| ------------------------- | ------------------------------------- | -------------------------------------------------- |
| `"extrapolate"` (default) | Preserve ordinary extrapolation       | Preserve the raw law of motion                     |
| `"clip_lookup"`           | Read at the nearest closed endpoint   | Preserve the raw value in histories and later laws |
| `"clip_state"`            | Read at the nearest closed endpoint   | Cap the next-period handoff to the grid            |
| `"raise"`                 | Refuse a feasible exterior next state | Refuse an exterior realized next state             |

The ambiguous spelling `"clip"` is rejected. `clip_lookup` is appropriate when a finite
numerical grid approximates a state whose economic support is larger; `clip_state`
changes the economic handoff and therefore requires an economic reason. Clipping
preserves non-finite values so a broken law cannot become a finite endpoint.
Stochastic-process support admission remains a separate contract.

The clipped rules are honoured by `GridSearch`. The endogenous-grid solvers (`EGM`,
`DCEGM`, `NBEGM`, `NEGM`, `NNBEGM`) read their continuation and policy on their own
endogenous grids, so a model whose endogenous-grid regime, or any regime it transitions
into, declares `clip_lookup` or `clip_state` is rejected at construction with
`OffGridPolicyError`; declare `"raise"` or `"extrapolate"` there instead.

For an age-dependent grid, the bounds belong to the target period. For a grid whose
points arrive at runtime, those supplied points define the closed endpoints. Interior
piecewise boundary ownership is unchanged. Numerical grid rules do not replace a tax,
benefit or other economic cap declared in the model.

With `runtime_checks=True` and output enabled, diagnostics count raw exterior next
states before clipping, excluding infeasible cells and unreachable lottery support.
Silent solves skip these count-only diagnostic sweeps. The explicit `"raise"` policy
applies at every `log_level`, even with `runtime_checks=False`.

A regime with value-dependent constraints or gated edges only knows which actions and
branches a next state is reached through once values exist. Its solve therefore checks
`"raise"` grids inside the maximization: an exterior next state counts only for an
action every constraint keeps, value constraints included, and through a gated edge only
on the branch the gate selects — the target where it opens, the projected fallback where
it closes. Such a regime emits no solve-phase diagnostic counts for the other rules.

### LinSpacedGrid

Evenly spaced points from `start` to `stop` (inclusive). The most common grid type for
wealth, consumption, and similar variables.

```python
LinSpacedGrid(start=0, stop=100, n_points=50)
```

### LogSpacedGrid

Points concentrated near `start` (logarithmic spacing). Good for variables with
diminishing marginal effects. `start` must be positive.

```python
LogSpacedGrid(start=0.1, stop=100, n_points=50)
```

### IrregSpacedGrid

Explicit point placement. Use when you need specific grid points (e.g., at policy
kinks). `points` is a one-dimensional NumPy or JAX array; a list or tuple is refused:

```python
IrregSpacedGrid(points=jnp.array([0.0, 0.5, 1.0, 5.0, 10.0, 50.0]))
```

You can also defer points to runtime by specifying only `n_points`. The actual points
are then supplied via the params dict:

```python
IrregSpacedGrid(n_points=4)
```

### PiecewiseLinSpacedGrid

Use explicit breakpoints when different parts of one finite domain need different linear
resolutions. Each breakpoint declares which neighboring segment owns the exact boundary
value:

```python
from lcm import GridBreakpoint, PiecewiseLinSpacedGrid

grid = PiecewiseLinSpacedGrid(
    start=0.0,
    stop=100.0,
    breakpoints=(
        GridBreakpoint(value=10.0, owner="right"),
        GridBreakpoint(value=40.0, owner="left"),
    ),
    points_per_segment=(20, 50, 30),
)
```

This declaration forms the nominal segments `[0, 10)`, `[10, 40]`, and `(40, 100]`. The
outer endpoints are always included. A right-owned breakpoint is the first node of the
segment to its right; a left-owned breakpoint is the last node of the segment to its
left. Every breakpoint therefore appears exactly once, and each count is the number of
output nodes its segment contributes:

```python
grid.n_points == 20 + 50 + 30
```

On the open side, the effective endpoint is the representable floating-point value
immediately next to the breakpoint. This keeps equality ownership exact without removing
a full grid spacing.

Use `breakpoints=()` with one entry in `points_per_segment` for a one-segment piecewise
declaration. `LinSpacedGrid` is normally simpler for that case.

### PiecewiseLogSpacedGrid

`PiecewiseLogSpacedGrid` uses the same breakpoint and ownership declarations but
logarithmic spacing within each segment. The complete domain and all breakpoints must be
positive.

```python
from lcm import GridBreakpoint, PiecewiseLogSpacedGrid

PiecewiseLogSpacedGrid(
    start=0.1,
    stop=1_000.0,
    breakpoints=(GridBreakpoint(value=10.0, owner="right"),),
    points_per_segment=(50, 30),
)
```

## Conditional actions

Often a continuous action matters only for some values of a discrete action: hours
worked only when working, a loan amount only when taking a loan. Declare it with
`ConditionalAction` instead of pinning it with a constraint:

```python
from lcm import ConditionalAction, DiscreteGrid, LinSpacedGrid

actions = {
    "labor": DiscreteGrid(category_class=Labor),
    "hours": ConditionalAction(
        grid=LinSpacedGrid(start=0.0, stop=40.0, n_points=5),
        active_when={"labor": Labor.yes},
        inactive_value=0.0,
    ),
    "consumption": LinSpacedGrid(start=0.5, stop=10.0, n_points=100),
}
```

- Where every condition holds, the action ranges over its grid. Elsewhere it takes
  `inactive_value`, and every user function sees that value.
- `active_when` maps discrete actions of the same regime to one value or to several in a
  tuple, such as `{"tenure": (Tenure.purchase, Tenure.refinance)}`. With several
  discrete actions, all conditions must hold.
- `inactive_value` must equal exactly one node of the grid.
- Simulated actions and `lookup_policy` report `inactive_value` where the action is
  inactive.

The declaration means the same as the constraint
`(labor == Labor.yes) | (hours == 0.0)`, and pylcm enforces it that way: results are
identical to that constraint.

## Grid Hierarchy (advanced)

All grids inherit from the `Grid` base class:

- `Grid` — base class, provides `to_jax()`
  - `DiscreteGrid` — categorical
  - `ContinuousGrid` — base for continuous grids, adds `get_coordinate()`
    - `UniformContinuousGrid` — start/stop/n_points base
      - `LinSpacedGrid`
      - `LogSpacedGrid`
    - `IrregSpacedGrid`
    - `PiecewiseLinSpacedGrid`
    - `PiecewiseLogSpacedGrid`
    - `ConditionalAction` — wraps one of the above for a conditional action
    - `_ContinuousStochasticProcess` — base for stochastic continuous grids

The `to_jax()` method converts any grid to a JAX array. `ContinuousGrid` subclasses
provide `get_coordinate()` for mapping values to grid coordinates, used in
[interpolation](../explanations/interpolation.ipynb).

## See Also

- [Regimes](regimes.ipynb) — how grids are used in regime definitions
- [Transitions](transitions.ipynb) — state and regime transitions
- [Continuous stochastic processes](continuous_stochastic_processes.md) — grids with
  built-in transitions
- [Interpolation](../explanations/interpolation.ipynb) — coordinate math for continuous
  grids
- [Age-specialized functions and grids](age_specialized.md) — let a continuous state's
  grid vary with age
