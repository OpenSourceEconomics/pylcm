---
title: Grids
---

# Grids

Grids define the outcome space for state and action variables — what values they can
take. They are passed via the `states` and `actions` mappings on a
[Regime](regimes.ipynb).

## Quick Reference

| Grid Type                | Use Case                  | Key Parameters              |
| ------------------------ | ------------------------- | --------------------------- |
| `DiscreteGrid`           | Categorical choices       | `category_class`            |
| `LinSpacedGrid`          | Evenly spaced continuous  | `start`, `stop`, `n_points` |
| `LogSpacedGrid`          | Log-spaced continuous     | `start`, `stop`, `n_points` |
| `IrregSpacedGrid`        | Custom point placement    | `points` or `n_points`      |
| `PiecewiseLinSpacedGrid` | Dense in some regions     | `pieces` (tuple of `Piece`) |
| `PiecewiseLogSpacedGrid` | Log-dense in some regions | `pieces` (tuple of `Piece`) |

All grid classes are imported from `lcm`:

```python
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    IrregSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    Piece,
    categorical,
)
```

## Discrete Grids

### DiscreteGrid

For categorical variables. Requires a `@categorical` frozen dataclass defining the
categories:

```python
from lcm import DiscreteGrid, categorical


@categorical(ordered=True)
class LaborSupply:
    do_not_work: int
    work: int


actions = {"labor_supply": DiscreteGrid(LaborSupply)}
```

Values are integer codes (0, 1, 2, ...) auto-assigned by `@categorical`. In simulation
output, labels are preserved via pandas Categorical.

When used as an **action**, no further configuration is needed. When used as a
**state**, the transition is specified via `state_transitions` on the `Regime` — see
[Transitions](transitions.ipynb).

For state-space dimensions, `DiscreteGrid` accepts a `dispatch=` kwarg that controls how
the dim is compiled into the Bellman kernel — trading memory for parallelism, and
optionally lifting the dim out of the state-action space so a later multi-GPU release
can shard across it. See [Dispatch Strategies](dispatch.md) for the four options, when
to pick each, and the full trade-off table.

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

Multiple linearly spaced segments joined at breakpoints. Dense where you need precision,
sparse elsewhere:

```python
from lcm import Piece, PiecewiseLinSpacedGrid

PiecewiseLinSpacedGrid(
    pieces=(
        Piece(interval="[0, 10)", n_points=20),
        Piece(interval="[10, 100]", n_points=10),
    ),
)
```

Pieces must be adjacent: the upper bound of each piece must equal the lower bound of the
next, with compatible open/closed boundaries (e.g., `[0, 10)` followed by `[10, 100]`).

### PiecewiseLogSpacedGrid

Same structure as `PiecewiseLinSpacedGrid` but with log spacing within each piece. Good
for wealth grids spanning orders of magnitude. All boundary values must be positive.

```python
PiecewiseLogSpacedGrid(
    pieces=(
        Piece(interval="[0.1, 10)", n_points=50),
        Piece(interval="[10, 1000]", n_points=30),
    ),
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
    - `_ShockGrid` — base for stochastic continuous grids

The `to_jax()` method converts any grid to a JAX array. `ContinuousGrid` subclasses
provide `get_coordinate()` for mapping values to grid coordinates, used in
[interpolation](../explanations/interpolation.ipynb).

## See Also

- [Regimes](regimes.ipynb) — how grids are used in regime definitions
- [Transitions](transitions.ipynb) — state and regime transitions
- [Shocks](shocks.md) — stochastic shock grids
- [Interpolation](../explanations/interpolation.ipynb) — coordinate math for continuous
  grids
