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

#### Runtime points driven by `extra_param_names`

When the gridpoints depend on a value that varies across solver iterations — a borrowing
limit, a per-iteration upper bound, a calibrated cap — declare the dependency explicitly
with `extra_param_names`:

```python
IrregSpacedGrid(n_points=64, extra_param_names=("max_consumption",))
```

`extra_param_names` is only valid together with runtime points (`points=None`). Each
name is added to the params template alongside the grid's `points` slot, and
`broadcast_to_template` carries it through even though no DAG function references it.
The names exist so user-side code that *constructs* the gridpoints can read them without
tripping the template validator's `Unknown keys` check.

The typical workflow looks like this. Declare the grid with `extra_param_names`,
populate the extras (e.g. as `model.fixed_params["max_consumption"]`), then inject the
points before solving or simulating:

```python
import jax.numpy as jnp
from lcm import IrregSpacedGrid, Model


def inject_consumption_points(*, params: dict, model: Model) -> dict:
    """Fill the runtime `consumption` gridpoints on every non-terminal regime."""
    max_consumption = jnp.asarray(model.fixed_params["max_consumption"])
    out: dict = dict(params)
    for regime_name, regime in model.regimes.items():
        if regime.terminal:
            continue
        grid = regime.actions["consumption"]
        assert isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime
        points = jnp.geomspace(1e-3, max_consumption, num=grid.n_points)
        regime_entry = dict(out.get(regime_name, {}))
        regime_entry["consumption"] = {"points": points}
        out[regime_name] = regime_entry
    return out


params = inject_consumption_points(params=model.get_params_template(), model=model)
model.solve(params=params)
```

`extra_param_names` carries any number of scalars (`("max_consumption", "floor")`,
etc.). The construction logic in the user-side helper is free to combine them however
the model needs — pylcm only validates that each declared name is present in the
resolved params.

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
