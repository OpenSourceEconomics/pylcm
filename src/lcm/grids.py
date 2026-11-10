"""User-facing grid classes and the `@categorical` decorator.

Leaf classes users instantiate to declare state and action grids on a
`Regime`, plus `@categorical` for declaring the category enumeration whose
fields become a `DiscreteGrid`'s codes, `characteristics` and
`category_with` for categories that are combinations of characteristics, and the
`ContinuousGrid` base class a solver names when it accepts any continuous
grid, and `ConditionalAction` for a continuous action that is active only under
some discrete-action values. Each constructor normalizes its input to the form
the engine stores; the internal `Grid` / `UniformContinuousGrid` ABCs, the
validators and the coordinate helpers live in `_lcm.grids`.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.axis_boundaries import BoundaryOwner
from _lcm.beartype_conf import GRID_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import coordinates as grid_coordinates
from _lcm.grids.base import UserVector, fail_if_not_a_numeric_vector
from _lcm.grids.categorical import DataclassInstance, categorical
from _lcm.grids.characteristics import category_with, characteristics
from _lcm.grids.conditional_action import (
    UserDiscreteValues,
    find_inactive_index,
    normalize_conditions,
)
from _lcm.grids.continuous import (
    ContinuousGrid,
    UniformContinuousGrid,
    _init_uniform_grid,
    _validate_irreg_spaced_grid,
    _validate_off_grid,
)
from _lcm.grids.discrete import _DiscreteGrid
from _lcm.grids.piecewise import (
    PiecewisePointCounts,
    _init_piecewise_grid,
    piecewise_coordinate,
    piecewise_nodes,
)
from lcm.exceptions import GridInitializationError
from lcm.typing import (
    ActionName,
    DiscreteGridValue,
    Float1D,
    FloatND,
    Int1D,
    OffGrid,
    ScalarFloat,
    ScalarInt,
)

__all__ = [
    "ConditionalAction",
    "ContinuousGrid",
    "DiscreteGrid",
    "GridBreakpoint",
    "IrregSpacedGrid",
    "LinSpacedGrid",
    "LogSpacedGrid",
    "PiecewiseLinSpacedGrid",
    "PiecewiseLogSpacedGrid",
    "categorical",
    "category_with",
    "characteristics",
]


class LinSpacedGrid(UniformContinuousGrid):
    """A linearly spaced grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 50.5, 100]`.

    """

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        n_points: int | ScalarInt,
        off_grid: OffGrid = "extrapolate",
    ) -> None:
        _init_uniform_grid(
            grid=self,
            start=float(start),
            stop=float(stop),
            n_points=n_points,
            requires_positive_start=False,
            off_grid=off_grid,
        )

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_coordinates.linspace(
            start=self.start, stop=self.stop, n_points=self.n_points
        )

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_linspace_coordinate(
            value=value,
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )


class LogSpacedGrid(UniformContinuousGrid):
    """A logarithmically spaced grid of continuous values.

    Requires `start > 0`.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    """

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        n_points: int | ScalarInt,
        off_grid: OffGrid = "extrapolate",
    ) -> None:
        _init_uniform_grid(
            grid=self,
            start=float(start),
            stop=float(stop),
            n_points=n_points,
            requires_positive_start=True,
            off_grid=off_grid,
        )

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array."""
        return grid_coordinates.logspace(
            start=self.start, stop=self.stop, n_points=self.n_points
        )

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        return grid_coordinates.get_logspace_coordinate(
            value=value,
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )


@dataclass(frozen=True, kw_only=True, init=False)
class IrregSpacedGrid(ContinuousGrid):
    """A grid of continuous values at irregular (user-specified) points.

    This grid type is useful for representing non-uniformly spaced points such as
    Gauss-Hermite quadrature nodes.

    `points` is a one-dimensional NumPy or JAX array of integer or floating dtype,
    stored as a JAX array at `canonical_float_dtype()`; lists and tuples are
    refused. When `points` is omitted and only `n_points` is given, the points
    must be supplied at runtime via the params.

    Example:
    --------
    Fixed grid: `IrregSpacedGrid(points=jnp.array([-1.73, -0.58, 0.58, 1.73]))`.
    Grid that is only completed at runtime via params: `IrregSpacedGrid(n_points=4)`.

    """

    points: Float1D | None
    """The grid points in ascending order, or `None` for runtime-supplied points."""

    n_points: int
    """Number of points. Derived from `len(points)` when points are given."""

    off_grid: OffGrid = dataclasses.field(
        default="extrapolate", metadata={"fingerprint_omit_if_default": True}
    )
    """Policy applied to values outside the supplied closed grid."""

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        points: UserVector | None = None,
        n_points: int | None = None,
        off_grid: OffGrid = "extrapolate",
    ) -> None:
        _validate_off_grid(off_grid)
        if points is not None:
            fail_if_not_a_numeric_vector(
                value=points,
                argument="points",
                example="IrregSpacedGrid(points=jnp.array([-1.73, -0.58, 0.58, 1.73]))",
            )
            _validate_irreg_spaced_grid(np.asarray(points, dtype=np.float64).tolist())
            derived_n = len(points)
            if n_points is None:
                n_points = derived_n
            elif n_points != derived_n:
                raise GridInitializationError(
                    f"n_points ({n_points}) does not match len(points) ({derived_n})"
                )
            stored_points: Float1D | None = jnp.asarray(
                points, dtype=canonical_float_dtype()
            )
            # Distinct inputs can round to equal nodes at the canonical precision.
            _validate_irreg_spaced_grid(stored_points.tolist())
        elif n_points is None:
            raise GridInitializationError(
                "Either points or n_points must be specified for IrregSpacedGrid."
            )
        elif n_points < 2:  # noqa: PLR2004
            raise GridInitializationError(
                f"n_points must be at least 2, got {n_points}"
            )
        else:
            stored_points = None
        object.__setattr__(self, "points", stored_points)
        object.__setattr__(self, "n_points", n_points)
        object.__setattr__(self, "off_grid", off_grid)

    @property
    def pass_points_at_runtime(self) -> bool:
        """Whether this grid's points are supplied at runtime via params."""
        return self.points is None

    def to_jax(self) -> Float1D:
        """Convert the grid to a Jax array.

        Raises `GridInitializationError` for runtime-supplied grids
        (`pass_points_at_runtime=True`). To get the substituted points,
        call `regime.solution.state_action_space(regime_params=...)` and
        read from `.states[name]` or `.continuous_actions[name]`.
        """
        if self.points is None:
            raise GridInitializationError(
                f"IrregSpacedGrid declared with n_points={self.n_points} and "
                f"no points; values are supplied at runtime via "
                f"params['<regime>']['<grid_name>']['points']. To get the "
                f"substituted points, call "
                f"`regime.solution.state_action_space(regime_params=...)` and "
                f"read from `.states[name]` or `.continuous_actions[name]`. "
                f"Use `.n_points` if only the shape is needed."
            )
        return self.points

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the generalized coordinate of a value in the grid."""
        if self.points is None:
            raise GridInitializationError(
                "Cannot compute coordinate without points. Pass points at "
                "initialization or use IrregSpacedGrid(n_points=...) and "
                "supply points at runtime via params."
            )
        return grid_coordinates.get_irreg_coordinate(value=value, points=self.points)


@beartype(conf=GRID_CONF)
@dataclass(frozen=True, kw_only=True)
class GridBreakpoint:
    """An interior grid boundary and the segment that owns equality."""

    value: float
    """Interior boundary value."""

    owner: BoundaryOwner = "right"
    """Segment containing the exact value: `"left"` or `"right"`."""


@dataclass(frozen=True, kw_only=True, init=False)
class _PiecewiseGrid(ContinuousGrid):
    """Common storage and coordinate geometry for piecewise continuous grids."""

    start: ScalarFloat
    """Closed lower endpoint at pylcm's canonical floating dtype."""

    stop: ScalarFloat
    """Closed upper endpoint at pylcm's canonical floating dtype."""

    breakpoints: tuple[GridBreakpoint, ...]
    """Strictly increasing interior boundaries and their equality owners."""

    points_per_segment: tuple[ScalarInt, ...]
    """Output-node count contributed by each nominal segment."""

    off_grid: OffGrid = dataclasses.field(
        default="extrapolate", metadata={"fingerprint_omit_if_default": True}
    )
    """Policy at the closed outer endpoints; interior ownership stays unchanged."""

    _breakpoint_values: Float1D = dataclasses.field(init=False, repr=False)
    _segment_selection_thresholds: Float1D = dataclasses.field(init=False, repr=False)
    _segment_starts: Float1D = dataclasses.field(init=False, repr=False)
    _segment_stops: Float1D = dataclasses.field(init=False, repr=False)
    _segment_n_points: Int1D = dataclasses.field(init=False, repr=False)
    _cumulative_offsets: Int1D = dataclasses.field(init=False, repr=False)

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        start: float | ScalarFloat,
        stop: float | ScalarFloat,
        breakpoints: tuple[GridBreakpoint, ...],
        points_per_segment: PiecewisePointCounts,
        off_grid: OffGrid = "extrapolate",
    ) -> None:
        _init_piecewise_grid(
            grid=self,
            start=float(start),
            stop=float(stop),
            breakpoint_values=tuple(float(point.value) for point in breakpoints),
            owners=tuple(point.owner for point in breakpoints),
            points_per_segment=points_per_segment,
            requires_positive_bounds=isinstance(self, PiecewiseLogSpacedGrid),
        )
        _validate_off_grid(off_grid)
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "off_grid", off_grid)

    @property
    def n_points(self) -> ScalarInt:
        """Return the number of output nodes contributed by all segments."""
        return self._segment_n_points.sum(dtype=jnp.int32)

    def _to_jax(self, *, logarithmic: bool) -> Float1D:
        """Construct and concatenate every segment's output nodes."""
        return piecewise_nodes(
            segment_starts=self._segment_starts,
            segment_stops=self._segment_stops,
            segment_n_points=self._segment_n_points,
            logarithmic=logarithmic,
        )

    def _get_coordinate(self, *, value: FloatND, logarithmic: bool) -> FloatND:
        """Return the ownership-aware generalized coordinate of `value`."""
        return piecewise_coordinate(
            value=value,
            segment_selection_thresholds=self._segment_selection_thresholds,
            segment_starts=self._segment_starts,
            segment_stops=self._segment_stops,
            segment_n_points=self._segment_n_points,
            cumulative_offsets=self._cumulative_offsets,
            logarithmic=logarithmic,
        )


class PiecewiseLinSpacedGrid(_PiecewiseGrid):
    """A linearly spaced grid split at explicitly owned breakpoints.

    Each entry of `points_per_segment` is the number of output nodes contributed
    by that nominal segment. Every breakpoint appears exactly once, in the segment
    selected by its `GridBreakpoint.owner`.
    """

    def to_jax(self) -> Float1D:
        """Return all segment nodes in ascending order."""
        return self._to_jax(logarithmic=False)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the ownership-aware generalized coordinate of a value."""
        return self._get_coordinate(value=value, logarithmic=False)


class PiecewiseLogSpacedGrid(_PiecewiseGrid):
    """A logarithmically spaced grid split at explicitly owned breakpoints.

    The complete domain and every breakpoint must be strictly positive. Ownership
    and contributed-node semantics are identical to `PiecewiseLinSpacedGrid`.
    """

    def to_jax(self) -> Float1D:
        """Return all segment nodes in ascending order."""
        return self._to_jax(logarithmic=True)

    def get_coordinate(self, value: FloatND) -> FloatND:
        """Return the ownership-aware generalized coordinate of a value."""
        return self._get_coordinate(value=value, logarithmic=True)


class DiscreteGrid(_DiscreteGrid):
    """A discrete grid defining the outcome space of a discrete variable.

    A grid is declared in one of two ways:

    - `category_class`: named categories. User functions receive the code of a
      category.
    - `values`: a one-dimensional NumPy or JAX array of numbers, e.g.
      `values=jnp.array([0, 20, 40])` or `values=jnp.arange(31)`. The codes are
      the positions `0..n-1` of the values.

    A value returned for a value grid names a node when it equals that node. For
    float values, `tolerance` widens the match to any value within that absolute
    distance of a node; `nearest` snaps a float to its closest node instead.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values. The legacy
            single-positional-argument form remains supported for compatibility.
        values: The numbers the variable takes, as a one-dimensional NumPy or JAX
            array of integer or floating dtype: strictly ascending and finite.
            Integers are stored as int32, floats at the canonical float dtype.
            Lists, tuples and ranges are refused.
        tolerance: The absolute distance within which a float matches a node. Zero
            (exact match) by default. Must be below half the smallest gap between
            values, and at least the float spacing at every value when positive.
            Integer values always match exactly.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields, or if `values` or `tolerance` are invalid.

    """

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *legacy_category_class: type[DataclassInstance],
        category_class: type[DataclassInstance] | None = None,
        values: UserVector | None = None,
        tolerance: float = 0.0,
    ) -> None:
        if len(legacy_category_class) > 1:
            msg = "DiscreteGrid accepts at most one positional argument."
            raise TypeError(msg)
        if legacy_category_class:
            if category_class is not None:
                msg = "DiscreteGrid got multiple values for 'category_class'."
                raise TypeError(msg)
            category_class = legacy_category_class[0]

        if values is not None:
            if category_class is not None:
                raise GridInitializationError(
                    "DiscreteGrid takes either `category_class` or `values`, not both."
                )
            self._init_from_values(values=values, tolerance=float(tolerance))
            return
        if category_class is None:
            msg = "DiscreteGrid requires either `category_class` or `values`."
            raise TypeError(msg)
        if tolerance != 0.0:
            raise GridInitializationError(
                "`tolerance` applies only to a grid declared with float `values`."
            )
        self._init_from_category_class(category_class)


@dataclass(frozen=True, kw_only=True, init=False)
class ConditionalAction(ContinuousGrid):
    """A continuous action that is active only under some discrete-action values.

    Where every condition in `active_when` holds, the action ranges over its grid.
    Elsewhere it takes `inactive_value`, and user functions see that value. The
    declaration means the same as a constraint pinning the action to its inactive
    node unless the conditions hold.

    Example:
        ```python
        "hours": ConditionalAction(
            grid=LinSpacedGrid(start=0.0, stop=40.0, n_points=5),
            active_when={"labor": Labor.yes},
            inactive_value=0.0,
        )
        ```

    """

    grid: ContinuousGrid
    """The grid the action ranges over where it is active."""

    active_when: MappingProxyType[ActionName, frozenset[DiscreteGridValue]]
    """Mapping of discrete action names to the values that activate the action.

    The action is active where every named discrete action takes one of its listed
    values. Values are what user functions see for that action: category codes for a
    categorical grid, numbers for a numeric one.
    """

    inactive_value: float
    """The node of `grid` the action takes where it is inactive."""

    inactive_index: int
    """Position of `inactive_value` among the nodes of `grid`."""

    def __init__(
        self,
        *,
        grid: ContinuousGrid,
        active_when: Mapping[ActionName, UserDiscreteValues],
        inactive_value: float,
    ) -> None:
        _fail_if_grid_is_not_plain(grid=grid)
        conditions = normalize_conditions(active_when=active_when)
        inactive_index = find_inactive_index(
            grid=grid, inactive_value=float(inactive_value)
        )
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "active_when", conditions)
        object.__setattr__(self, "inactive_value", float(inactive_value))
        object.__setattr__(self, "inactive_index", inactive_index)

    @property
    def off_grid(self) -> OffGrid:
        """Return the lookup policy of the wrapped grid."""
        return self.grid.off_grid

    def to_jax(self) -> Float1D:
        """Return the nodes of the wrapped grid."""
        return self.grid.to_jax()

    def get_coordinate(self, value: ScalarFloat) -> ScalarFloat:
        """Return the coordinate of `value` on the wrapped grid."""
        return self.grid.get_coordinate(value)


def _fail_if_grid_is_not_plain(*, grid: ContinuousGrid) -> None:
    if isinstance(grid, ConditionalAction):
        raise GridInitializationError(
            "ConditionalAction must wrap a plain continuous grid such as "
            f"LinSpacedGrid; got {type(grid).__name__}."
        )
    if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
        raise GridInitializationError(
            "ConditionalAction cannot wrap an IrregSpacedGrid that receives its "
            "points at runtime: its inactive node must be known when the model is "
            "built."
        )
