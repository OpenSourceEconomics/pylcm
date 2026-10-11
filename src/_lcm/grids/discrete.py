"""`DiscreteGrid`: the outcome space of one discrete variable.

A grid holds either a category class, whose codes index the variable's axis, or
a strictly ascending sequence of numbers, whose positions are the codes. Stochastic
movement over the codes is declared with `StochasticTransition`, not here.
"""

import itertools
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Int

from _lcm.beartype_conf import GRID_CONF
from _lcm.dtypes import canonical_float_dtype
from _lcm.grids.base import Grid
from _lcm.grids.categorical import (
    DataclassInstance,
    _validate_discrete_grid,
    get_category_codes,
)
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, FloatND, Int1D, IntND, ValueND

# The code of a value off a value grid, equal to the simulation's sentinel for
# a categorical state missing from initial conditions. int32 min stays
# out of bounds after JAX normalizes a negative index by adding the axis length,
# so a fill-mode read cannot wrap it onto the last node the way `-1` would.
MISSING_CAT_CODE = jnp.iinfo(jnp.int32).min

# A code array of any integer width; with x64 enabled, `jnp.array([0])` is int64.
type _CodeIndex = Int[Array, "..."]

# A declared value, before it is normalized to a Python scalar.
type _UserValue = int | float | np.integer | np.floating


class DiscreteGrid(Grid):
    """A discrete grid defining the outcome space of a discrete variable.

    A grid is declared in one of two ways:

    - `category_class`: named categories. User functions receive the code of a
      category.
    - `values`: numbers, e.g. `values=[0, 20, 40]` or `values=range(31)`. The
      codes are the positions `0..n-1` of the values.

    A value returned for a value grid names a node when it equals that node. For
    float values, `tolerance` widens the match to any value within that absolute
    distance of a node; `nearest` snaps a float to its closest node instead.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values. The legacy
            single-positional-argument form remains supported for compatibility.
        values: The numbers the variable takes, as a sequence or a one-dimensional
            array: strictly ascending, finite, and all integers or all floats.
            Integers are stored as int32, floats at the canonical float dtype.
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
        values: Sequence[_UserValue] | np.ndarray | ValueND | None = None,
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
            self.__init_values(values=values, tolerance=tolerance)
            return
        if category_class is None:
            msg = "DiscreteGrid requires either `category_class` or `values`."
            raise TypeError(msg)
        if tolerance != 0.0:
            raise GridInitializationError(
                "`tolerance` applies only to a grid declared with float `values`."
            )

        _validate_discrete_grid(category_class)
        category_codes = get_category_codes(category_class)
        self.__categories = tuple(category_codes)
        # Coerce `ScalarInt` codes to Python `int` for the `codes` property.
        # `codes` is the Python-side API (the tuple flows into dict/set
        # operations that need hashable members); the JAX-side representation
        # comes from `to_jax()`.
        self.__codes = tuple(int(code) for code in category_codes.values())
        self.__ordered: bool = getattr(category_class, "_ordered", False)
        self.__values: tuple[int | float, ...] | None = None
        self.__value_array: Int1D | Float1D | None = None
        self.__tolerance = 0.0
        self.__holds_integers = False

    def __init_values(
        self,
        *,
        values: Sequence[_UserValue] | np.ndarray | ValueND,
        tolerance: float,
    ) -> None:
        """Validate and store a value grid's numbers and tolerance."""
        if isinstance(values, np.ndarray | jax.Array):
            if values.ndim != 1:
                raise GridInitializationError(
                    f"`values` must be one-dimensional, got shape {values.shape}."
                )
            declared = _validate_values(np.asarray(values).tolist())
        else:
            declared = _validate_values(values)
        is_integer = isinstance(declared[0], int)
        value_array = jnp.asarray(
            declared, dtype=jnp.int32 if is_integer else canonical_float_dtype()
        )
        # Distinct floats can round to equal nodes at the canonical precision.
        collision = _value_order_error(tuple(value_array.tolist()))
        if collision is not None:
            raise GridInitializationError(
                f"{collision} The values collide after rounding to {value_array.dtype}."
            )
        _validate_tolerance(
            tolerance=tolerance, value_array=value_array, is_integer=is_integer
        )
        self.__categories = tuple(repr(value) for value in declared)
        self.__codes = tuple(range(len(declared)))
        self.__ordered = True
        self.__values = declared
        self.__value_array = value_array
        self.__tolerance = float(tolerance)
        self.__holds_integers = is_integer

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the category names; the `repr` of each value for a value grid."""
        return self.__categories

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the list of category codes."""
        return self.__codes

    @property
    def ordered(self) -> bool:
        """Return whether the categories have a meaningful ordering."""
        return self.__ordered

    @property
    def values(self) -> tuple[int | float, ...] | None:
        """Return the declared values, or `None` for a category-class grid."""
        return self.__values

    @property
    def tolerance(self) -> float:
        """Return the absolute distance within which a float matches a node."""
        return self.__tolerance

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array.

        Discrete state/action codes are pinned to `int32` regardless of the
        ambient `jax_enable_x64` setting. A single integer dtype across
        transitions, V-array indexing, and action lookups keeps the JIT cache
        unsplit and lets AOT-compiled programs ship one signature. `int32`
        accommodates any realistic category count and matches the
        `MISSING_CAT_CODE` sentinel.
        """
        return jnp.array(self.codes, dtype=jnp.int32)

    def values_to_jax(self) -> Int1D | Float1D:
        """Return the values of a value grid, int32 or at the canonical float dtype.

        Raises:
            GridInitializationError: If the grid is declared with a category class.

        """
        return self.__require_value_array()

    def encode(self, value: ArrayLike) -> IntND:
        """Return the code of the node each value names, else `MISSING_CAT_CODE`.

        An integer grid takes integers and matches by equality. A float grid
        compares at its own dtype and matches the nearest node within
        `tolerance`. NaN never matches.

        Raises:
            TypeError: If an integer grid receives non-integer values.

        """
        nodes = self.__require_value_array()
        value = jnp.asarray(value)
        if self.__holds_integers:
            if value.dtype.kind not in "iu":
                msg = (
                    "An integer value grid matches integers exactly; got "
                    f"{value.dtype} values. Return integers from the law."
                )
                raise TypeError(msg)
            index = jnp.clip(
                jnp.searchsorted(nodes, value, side="left"), 0, nodes.shape[0] - 1
            )
            matches = nodes[index] == value
        else:
            value = value.astype(nodes.dtype)
            index = _nearest_index(nodes=nodes, value=value)
            matches = jnp.abs(value - nodes[index]) <= self.__tolerance
        return jnp.where(matches, index, MISSING_CAT_CODE).astype(jnp.int32)

    def decode(self, code: ArrayLike) -> ValueND:
        """Return the value each code indexes.

        A code outside `0..n-1`, including `MISSING_CAT_CODE` and `-1`, reads NaN
        on a float grid and `MISSING_CAT_CODE` on an integer grid. Nothing is
        clamped or wrapped onto a node.
        """
        nodes = self.__require_value_array()
        return nodes.at[jnp.asarray(code)].get(
            mode="fill",
            # Integers have no NaN, so an off-grid code reads the sentinel there.
            fill_value=MISSING_CAT_CODE if self.__holds_integers else math.nan,
            wrap_negative_indices=False,
        )

    def nearest(self, value: ArrayLike) -> FloatND:
        """Return the node of a float grid closest to each value, the lower on a tie.

        Use it in a law that computes a value by arithmetic and means the node it
        lands next to. NaN stays NaN, so `encode` still flags it.

        Raises:
            TypeError: If the grid holds integer values, which laws return exactly.

        """
        nodes = self.__require_value_array()
        if self.__holds_integers:
            msg = (
                "`nearest` snaps floats onto a float value grid. An integer value "
                "grid matches the integers a law returns exactly."
            )
            raise TypeError(msg)
        value = jnp.asarray(value).astype(nodes.dtype)
        snapped = nodes[_nearest_index(nodes=nodes, value=value)]
        return jnp.where(jnp.isnan(value), jnp.nan, snapped)

    def __require_value_array(self) -> Int1D | Float1D:
        if self.__value_array is None:
            raise GridInitializationError(
                "Only a DiscreteGrid declared with `values` has values; this grid "
                f"has the categories {self.categories}."
            )
        return self.__value_array


def _validate_values(values: Sequence[_UserValue]) -> tuple[int | float, ...]:
    """Return the values as Python scalars, or raise if they are no value grid."""
    if len(values) == 0:
        raise GridInitializationError("`values` must hold at least one value.")
    bools = [i for i, value in enumerate(values) if isinstance(value, bool | np.bool_)]
    if bools:
        raise GridInitializationError(
            f"`values` must be numbers; the entries at {bools} are bool."
        )
    scalars = tuple(
        value.item() if isinstance(value, np.generic) else value for value in values
    )
    message = _value_kind_error(scalars) or _value_order_error(scalars)
    if message is not None:
        raise GridInitializationError(message)
    return scalars


def _value_kind_error(scalars: tuple[int | float, ...]) -> str | None:
    """Report values that are mixed, outside int32, or not finite."""
    if all(isinstance(value, int) for value in scalars):
        out_of_range = [
            value for value in scalars if not MISSING_CAT_CODE < value < 2**31
        ]
        if out_of_range:
            return (
                "Integer `values` must lie in the int32 range above its minimum, "
                f"[{MISSING_CAT_CODE + 1}, {2**31 - 1}]; got {out_of_range}."
            )
        return None
    if all(isinstance(value, float) for value in scalars):
        non_finite = [value for value in scalars if not math.isfinite(value)]
        if non_finite:
            return f"`values` must be finite; got {non_finite}."
        return None
    return (
        "`values` must be all integers or all floats; got "
        f"{[type(value).__name__ for value in scalars]}."
    )


def _value_order_error(scalars: tuple[int | float, ...]) -> str | None:
    """Report the first pair of values that is not strictly ascending."""
    for position, (left, right) in enumerate(itertools.pairwise(scalars)):
        if not left < right:
            return (
                "`values` must be strictly ascending; "
                f"values[{position}]={left} >= values[{position + 1}]={right}."
            )
    return None


def _validate_tolerance(
    *, tolerance: float, value_array: Int1D | Float1D, is_integer: bool
) -> None:
    """Raise unless `tolerance` matches each value to at most one node."""
    if is_integer:
        if tolerance != 0.0:
            raise GridInitializationError(
                "Integer `values` always match exactly; `tolerance` applies only "
                f"to float values, got tolerance={tolerance}."
            )
        return
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise GridInitializationError(
            f"`tolerance` must be finite and >= 0, got {tolerance}."
        )
    nodes = np.asarray(value_array)
    if nodes.size > 1:
        smallest_gap = float(np.min(np.diff(nodes)))
        if not tolerance < smallest_gap / 2:
            raise GridInitializationError(
                f"`tolerance` must be below half the smallest gap between values "
                f"({smallest_gap / 2}), so a value matches at most one node; got "
                f"{tolerance}."
            )
    if tolerance > 0:
        spacing = np.spacing(np.abs(nodes))
        widest = int(np.argmax(spacing))
        if tolerance < spacing[widest]:
            raise GridInitializationError(
                f"`tolerance` {tolerance} is below the float spacing "
                f"{float(spacing[widest])} at the value {float(nodes[widest])} in "
                f"{nodes.dtype}, so it admits no value near that node. Use 0 for an "
                f"exact match or a tolerance of at least that spacing."
            )


def _nearest_index(*, nodes: Float1D, value: FloatND) -> IntND:
    """Return the index of the float node closest to each value, the lower on a tie.

    A value outside the nodes, infinity included, takes the end node. NaN takes an
    arbitrary node; callers decide what NaN means.
    """
    n_nodes = nodes.shape[0]
    upper = jnp.clip(jnp.searchsorted(nodes, value, side="left"), 0, n_nodes - 1)
    lower = jnp.clip(upper - 1, 0, n_nodes - 1)
    lower_value = nodes[lower]
    upper_value = nodes[upper]
    # Inside the nodes `lower_value < value <= upper_value`, so both distances
    # are nonnegative; at or above the last node the upper node is the closest.
    upper_is_closer = (value >= upper_value) | (
        (value > lower_value) & (value - lower_value > upper_value - value)
    )
    return jnp.where(upper_is_closer, upper, lower).astype(jnp.int32)


def read_at_codes(
    *, array: FloatND, positions: tuple[_CodeIndex | slice, ...]
) -> FloatND:
    """Return `array[positions]`, NaN wherever a code lies off its axis.

    A code outside `0..n-1`, including `MISSING_CAT_CODE` and `-1`, reads NaN.
    Nothing is clamped onto an end node or wrapped onto the last one, so a read
    of an off-grid next state cannot pass for a value.
    """
    return array.at[positions].get(
        mode="fill", fill_value=jnp.nan, wrap_negative_indices=False
    )
