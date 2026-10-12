"""`DiscreteGrid`: the outcome space of one discrete variable.

A grid holds either a category class, whose codes index the variable's axis, or
a strictly ascending sequence of numbers, whose positions are the codes. Stochastic
movement over the codes is declared with `StochasticTransition`, not here.
"""

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax.typing import ArrayLike

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
_INT32_MAX = jnp.iinfo(jnp.int32).max

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
        self.__midpoints = None
        self.__tolerance = 0.0

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
        stored = tuple(value_array.tolist())
        # A finite declared float can overflow the canonical precision.
        if not is_integer and not all(math.isfinite(value) for value in stored):
            raise GridInitializationError(
                f"`values` must remain finite at {value_array.dtype}; got {stored}."
            )
        # Distinct floats can round to equal nodes at the canonical precision.
        collision = _value_order_error(stored)
        if collision is not None:
            raise GridInitializationError(
                f"{collision} The values collide after rounding to {value_array.dtype}."
            )
        _validate_tolerance(
            tolerance=tolerance, value_array=value_array, is_integer=is_integer
        )
        self.__midpoints: Float1D | None = (
            None
            if is_integer
            else jnp.asarray(
                _downward_midpoints(nodes=np.asarray(value_array)),
                dtype=value_array.dtype,
            )
        )
        self.__categories = tuple(repr(value) for value in declared)
        self.__codes = tuple(range(len(declared)))
        self.__ordered = True
        self.__values = declared
        self.__value_array = value_array
        self.__tolerance = float(tolerance)

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
        if self.__midpoints is None:
            value = _classify_integers(value)
            index = jnp.clip(
                jnp.searchsorted(nodes, value, side="left"), 0, nodes.shape[0] - 1
            )
            matches = nodes[index] == value
        else:
            value = _as_float_query(value=value, dtype=nodes.dtype)
            index = _nearest_index(midpoints=self.__midpoints, value=value)
            matches = jnp.abs(value - nodes[index]) <= self.__tolerance
        return jnp.where(matches, index, MISSING_CAT_CODE).astype(jnp.int32)

    def decode(self, code: ArrayLike) -> ValueND:
        """Return the value each code indexes.

        A code outside `0..n-1`, including `MISSING_CAT_CODE` and `-1`, reads NaN
        on a float grid and `MISSING_CAT_CODE` on an integer grid. Nothing is
        clamped or wrapped onto a node.
        """
        nodes = self.__require_value_array()
        codes = _classify_integers(code)
        n_nodes = nodes.shape[0]
        index = jnp.where((codes >= 0) & (codes < n_nodes), codes, n_nodes)
        return nodes.at[index].get(
            mode="fill",
            fill_value=_off_grid_value(nodes),
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
        if self.__midpoints is None:
            msg = (
                "`nearest` snaps floats onto a float value grid. An integer value "
                "grid matches the integers a law returns exactly."
            )
            raise TypeError(msg)
        value = _as_float_query(value=value, dtype=nodes.dtype)
        snapped = nodes[_nearest_index(midpoints=self.__midpoints, value=value)]
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
    # Compared as the dtype it is applied at, against exact rational gaps: a
    # float subtraction of distant nodes can overflow.
    stored_tolerance = Fraction(float(nodes.dtype.type(tolerance)))
    if nodes.size > 1:
        half_gap = (
            min(
                Fraction(float(right)) - Fraction(float(left))
                for left, right in itertools.pairwise(nodes)
            )
            / 2
        )
        if not stored_tolerance < half_gap:
            raise GridInitializationError(
                f"`tolerance` must be below half the smallest gap between values "
                f"({float(half_gap)}), so a value matches at most one node; got "
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


def _nearest_index(*, midpoints: Float1D, value: FloatND) -> IntND:
    """Return the index of the float node closest to each value, the lower on a tie.

    `midpoints` holds, for each pair of adjacent nodes, the largest float at or
    below their exact midpoint, so a value owns the upper node exactly when it
    lies above that float. A value outside the nodes, infinity included, takes
    the end node. NaN takes an arbitrary node; callers decide what NaN means.
    """
    return jnp.searchsorted(midpoints, value, side="left").astype(jnp.int32)


def _downward_midpoints(*, nodes: np.ndarray) -> np.ndarray:
    """Return the largest float of the nodes' dtype at or below each exact midpoint.

    The midpoints are compared as exact rationals, so no rounding of a distance or
    a midpoint can move a value to the wrong side.
    """
    dtype = nodes.dtype.type
    cuts = []
    for left, right in itertools.pairwise(nodes):
        midpoint = (Fraction(float(left)) + Fraction(float(right))) / 2
        cut = dtype(float(midpoint))
        while Fraction(float(cut)) > midpoint:
            cut = np.nextafter(cut, dtype(-math.inf))
        while Fraction(float(np.nextafter(cut, dtype(math.inf)))) <= midpoint:
            cut = np.nextafter(cut, dtype(math.inf))
        cuts.append(cut)
    return np.asarray(cuts, dtype=nodes.dtype)


def _classify_integers(value: ArrayLike) -> IntND:
    """Return each integer as int32 if int32 holds it, else `MISSING_CAT_CODE`.

    The range test runs on the original integers, in their own dtype, before any
    conversion, so no integer of any width or signedness can alias a different
    int32. `MISSING_CAT_CODE` is never a node and never a valid code, so callers
    need no separate validity mask:

    - int8, int16, uint8 and uint16 widen to int32 first, which is exact; a
      bound wider than the dtype would otherwise wrap into it.
    - int32, int64, uint32 and uint64 compare against bounds built in their own
      dtype, so no comparison promotes; unsigned integers are never negative, so
      their lower bound is 0.
    - Host values are classified in NumPy before they reach the device, so a
      64-bit host integer is never truncated on upload when x64 is off.

    Raises:
        TypeError: If the values are not integers.

    """
    library = jnp if isinstance(value, jax.Array) else np
    integers = library.asarray(value)
    if integers.dtype.kind not in "iu":
        msg = f"Value-grid integers and codes must be integers; got {integers.dtype}."
        raise TypeError(msg)
    if integers.dtype.itemsize < np.dtype(np.int32).itemsize:
        integers = integers.astype(library.int32)
    bound = integers.dtype.type
    lowest = 0 if integers.dtype.kind == "u" else MISSING_CAT_CODE + 1
    in_range = (integers >= bound(lowest)) & (integers <= bound(_INT32_MAX))
    classified = library.where(
        in_range, integers.astype(library.int32), library.int32(MISSING_CAT_CODE)
    )
    return jnp.asarray(classified, dtype=jnp.int32)


def _as_float_query(*, value: ArrayLike, dtype: np.dtype) -> FloatND:
    """Return queries at the grid's float dtype, converting host values directly."""
    if isinstance(value, jax.Array):
        return value.astype(dtype)
    return jnp.asarray(np.asarray(value).astype(dtype))


def _off_grid_value(nodes: Int1D | Float1D) -> float | int:
    """Return the value an off-grid code reads: NaN, or the int32 sentinel."""
    if jnp.issubdtype(nodes.dtype, jnp.integer):
        return MISSING_CAT_CODE
    return math.nan
