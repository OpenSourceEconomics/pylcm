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
