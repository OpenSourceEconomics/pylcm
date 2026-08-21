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

## Quick Reference

| Grid Type                | Use Case                  | Key Parameters                                       |
| ------------------------ | ------------------------- | ---------------------------------------------------- |
| `DiscreteGrid`           | Categorical choices       | `category_class`                                     |
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


actions = {"labor_supply": DiscreteGrid(LaborSupply)}
```

Values are integer codes (0, 1, 2, ...) auto-assigned by `@categorical`. In simulation
output, labels are preserved via pandas Categorical.

When used as an **action**, no further configuration is needed. When used as a
**state**, the transition is specified via `state_transitions` on the `Regime` — see
[Transitions](transitions.ipynb).

## Continuous Grids

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
kinks):

```python
IrregSpacedGrid(points=(0.0, 0.5, 1.0, 5.0, 10.0, 50.0))
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
