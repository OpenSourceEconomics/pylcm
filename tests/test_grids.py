import dataclasses
import functools
import itertools
import math
from dataclasses import make_dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
    validate_category_class,
)
from _lcm.grids import discrete as discrete_module
from _lcm.grids.categorical import _validate_discrete_grid, get_category_codes
from _lcm.grids.continuous import _validate_continuous_grid
from _lcm.grids.discrete import MISSING_CAT_CODE
from _lcm.simulation import initial_conditions
from _lcm.typing import DataclassInstance
from lcm.exceptions import CategoricalDefinitionError, GridInitializationError
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED


def _make_dc(
    name: str, *fields: tuple[str, jax.Array | str | None]
) -> type[DataclassInstance]:
    """Build a dataclass with `ScalarInt` class attrs.

    Sidesteps `@dataclass(frozen=True)`'s rejection of `jax.Array`
    defaults: we make the dataclass with no defaults, then write the
    `ScalarInt` scalars onto the class via `type.__setattr__`.
    `validate_category_class` reads field values via `getattr(cls, name)`,
    so the class attrs are what get checked.
    """
    # `make_dataclass` is typed as returning a bare `type`.
    cls = cast(
        "type[DataclassInstance]",
        make_dataclass(name, [(fname, ScalarInt) for fname, _ in fields]),
    )
    for fname, fval in fields:
        type.__setattr__(cls, fname, fval)
    return cls


@categorical(ordered=False)
class _Category:
    a: ScalarInt
    b: ScalarInt


def test_get_category_codes_returns_each_category_code():
    """Each category of a validated class maps to its integer code, in order."""
    codes = get_category_codes(_Category)
    assert [(name, int(code)) for name, code in codes.items()] == [("a", 0), ("b", 1)]


def test_get_category_codes_returns_a_read_only_mapping():
    """The category codes come back read-only."""
    assert type(get_category_codes(_Category)) is MappingProxyType


def test_discrete_grid_reports_a_field_without_a_value():
    """A category field without a value is reported like any non-`ScalarInt`."""
    category_class = cast(
        "type[DataclassInstance]", make_dataclass("Category", [("a", ScalarInt)])
    )
    error_msg = (
        "Field values of the category_class must be `ScalarInt` "
        r"\(0-d int32 jax scalars\). The values to the following "
        r"fields are not: \['a'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        DiscreteGrid(category_class=category_class)


def test_validate_discrete_grid_empty():
    category_class = _make_dc("Category")
    error_msg = "category_class must have at least one field"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_scalar_input():
    category_class = _make_dc("Category", ("a", jnp.int32(1)), ("b", "s"))
    error_msg = (
        "Field values of the category_class must be `ScalarInt` "
        r"\(0-d int32 jax scalars\). The values to the following "
        r"fields are not: \['b'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_int32_input():
    """A 0-d integer scalar of another width is no `ScalarInt` code."""
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", jnp.int16(1)))
    error_msg = (
        "Field values of the category_class must be `ScalarInt` "
        r"\(0-d int32 jax scalars\). The values to the following "
        r"fields are not: \['b'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_none_input():
    category_class = _make_dc("Category", ("a", None), ("b", jnp.int32(1)))
    error_msg = (
        "Field values of the category_class must be `ScalarInt` "
        r"\(0-d int32 jax scalars\). The values to the following "
        r"fields are not: \['a'\]"
    )
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_unique():
    category_class = _make_dc("Category", ("a", jnp.int32(1)), ("b", jnp.int32(1)))
    error_msg = "Field values of the category_class must be unique."
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_consecutive_unordered():
    category_class = _make_dc("Category", ("a", jnp.int32(1)), ("b", jnp.int32(0)))
    error_msg = "Field values of the category_class must be consecutive integers"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_discrete_grid_non_consecutive_jumps():
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", jnp.int32(2)))
    error_msg = "Field values of the category_class must be consecutive integers"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_discrete_grid(category_class)


def test_validate_category_class_valid():
    """A valid category class yields no error messages."""
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", jnp.int32(1)))
    errors = validate_category_class(category_class)
    assert errors == ()


def test_validate_category_class_not_dataclass():
    """Non-dataclass should return error."""

    class NotDataclass:
        a = 0
        b = 1

    errors = validate_category_class(NotDataclass)  # ty: ignore[invalid-argument-type]
    assert len(errors) == 1
    assert "must be a dataclass" in errors[0]


def test_validate_category_class_non_consecutive():
    """Non-consecutive values should return error."""
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", jnp.int32(2)))
    errors = validate_category_class(category_class)
    assert len(errors) == 1
    assert "consecutive integers" in errors[0]


def test_validate_category_class_not_starting_at_zero():
    """Values not starting at 0 should return error."""
    category_class = _make_dc("Category", ("a", jnp.int32(1)), ("b", jnp.int32(2)))
    errors = validate_category_class(category_class)
    assert len(errors) == 1
    assert "consecutive integers starting from 0" in errors[0]


def test_discrete_grid_creation():
    category_class = _make_dc(
        "Category",
        ("a", jnp.int32(0)),
        ("b", jnp.int32(1)),
        ("c", jnp.int32(2)),
    )
    grid = DiscreteGrid(category_class=category_class)
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_discrete_grid_accepts_category_class_as_legacy_positional_argument():
    """The long-standing public constructor form remains supported."""
    category_class = _make_dc(
        "Category",
        ("a", jnp.int32(0)),
        ("b", jnp.int32(1)),
    )

    grid = DiscreteGrid(category_class)

    assert grid.categories == ("a", "b")


def test_discrete_grid_invalid_category_class():
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", "wrong_type"))
    with pytest.raises(
        GridInitializationError,
        match="Field values of the category_class must be `ScalarInt`",
    ):
        DiscreteGrid(category_class=category_class)


def test_discrete_grid_ordered_true():
    @categorical(ordered=True)
    class OrderedCat:
        low: ScalarInt
        medium: ScalarInt
        high: ScalarInt

    grid = DiscreteGrid(category_class=OrderedCat)
    assert grid.ordered is True


def test_discrete_grid_ordered_false():
    @categorical(ordered=False)
    class UnorderedCat:
        a: ScalarInt
        b: ScalarInt

    grid = DiscreteGrid(category_class=UnorderedCat)
    assert grid.ordered is False


# @categorical: ScalarInt annotation contract


def test_categorical_rejects_int_annotation():
    """Plain `int` annotations are rejected at decoration time."""
    with pytest.raises(
        CategoricalDefinitionError,
        match=r"must annotate every field as `ScalarInt`",
    ):
        # `a: int` MUST stay — this is the rejected case under test.
        @categorical(ordered=False)
        class _Bad:
            a: int


def test_categorical_rejects_str_annotation():
    """Non-`ScalarInt` annotations of any kind are rejected."""
    with pytest.raises(
        CategoricalDefinitionError,
        match=r"must annotate every field as `ScalarInt`",
    ):

        @categorical(ordered=False)
        class _Bad:
            a: str


def test_categorical_error_lists_all_offending_fields():
    """The error message names every field with a wrong annotation."""
    with pytest.raises(CategoricalDefinitionError, match=r"`x: int`.*`y: str`"):
        # `x: int` / `y: str` MUST stay — both are rejected cases under test.
        @categorical(ordered=False)
        class _Bad:
            x: int
            y: str


@categorical(ordered=False)
class _LaborSupply:
    work: ScalarInt
    retire: ScalarInt


def test_categorical_class_is_a_dataclass_of_its_categories():
    """A `@categorical` class is a dataclass whose fields are its categories."""
    names = [field.name for field in dataclasses.fields(_LaborSupply)]
    assert names == ["work", "retire"]


def test_categorical_class_attr_is_scalar_int():
    """Class-level access returns a 0-d int32 jax scalar."""

    @categorical(ordered=False)
    class Cat:
        first: ScalarInt
        second: ScalarInt

    assert Cat.first.shape == ()
    assert Cat.first.dtype == jnp.int32
    assert int(Cat.first) == 0
    assert int(Cat.second) == 1


def test_categorical_instance_attr_is_scalar_int():
    """Instance-level access also returns a 0-d int32 jax scalar."""

    @categorical(ordered=False)
    class Cat:
        first: ScalarInt
        second: ScalarInt

    # `dataclass_transform` cannot express the decorator's `init=False` fields.
    instance = Cat()  # ty: ignore[missing-argument]
    assert instance.first.shape == ()
    assert instance.first.dtype == jnp.int32
    assert int(instance.second) == 1


def test_lin_spaced_grid_rejects_non_numeric_start():
    """Non-numeric `start` raises `GridInitializationError`."""
    with pytest.raises(GridInitializationError, match="start"):
        LinSpacedGrid(start="a", stop=1, n_points=10)  # ty: ignore[invalid-argument-type]


def test_lin_spaced_grid_rejects_non_numeric_stop():
    """Non-numeric `stop` raises `GridInitializationError`."""
    with pytest.raises(GridInitializationError, match="stop"):
        LinSpacedGrid(start=1, stop="a", n_points=10)  # ty: ignore[invalid-argument-type]


def test_lin_spaced_grid_rejects_non_numeric_n_points():
    """Non-numeric `n_points` raises `GridInitializationError`."""
    with pytest.raises(GridInitializationError, match="n_points"):
        LinSpacedGrid(start=1, stop=2, n_points="a")  # ty: ignore[invalid-argument-type]


def test_validate_continuous_grid_negative_n_points():
    error_msg = "n_points must be an int greater than 0 but is -1"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(
            start=jnp.asarray(1.0), stop=jnp.asarray(2.0), n_points=jnp.int32(-1)
        )


def test_validate_continuous_grid_start_greater_than_stop():
    error_msg = "start must be less than stop"
    with pytest.raises(GridInitializationError, match=error_msg):
        _validate_continuous_grid(
            start=jnp.asarray(2.0), stop=jnp.asarray(1.0), n_points=jnp.int32(10)
        )


def test_linspace_grid_creation():
    grid = LinSpacedGrid(start=1, stop=5, n_points=5)
    assert np.allclose(grid.to_jax(), np.linspace(1, 5, 5))


def test_linspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LinSpacedGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_creation():
    grid = LogSpacedGrid(start=1, stop=10, n_points=3)
    assert np.allclose(grid.to_jax(), np.logspace(np.log10(1), np.log10(10), 3))


_SINGLETON_LOG_QUERIES = (1.0, 1.5, 2.0, 4.0)


@pytest.mark.parametrize("jit", [False, True])
def test_singleton_logspace_grid_maps_every_value_to_its_only_node(
    *, jit: bool
) -> None:
    grid = LogSpacedGrid(start=1, stop=2, n_points=1)
    coordinate = jax.jit(grid.get_coordinate) if jit else grid.get_coordinate
    np.testing.assert_array_equal(
        coordinate(jnp.asarray(_SINGLETON_LOG_QUERIES)), np.zeros(4)
    )


def test_singleton_logspace_grid_coordinate_has_zero_derivative() -> None:
    grid = LogSpacedGrid(start=1, stop=2, n_points=1)
    derivative = jax.vmap(jax.grad(grid.get_coordinate))(
        jnp.asarray(_SINGLETON_LOG_QUERIES)
    )
    np.testing.assert_array_equal(derivative, np.zeros(4))


@pytest.mark.parametrize("start", [1e3, 1e6, 1e9])
def test_singleton_logspace_grid_has_zero_derivative_when_bounds_share_a_log(
    *, start: float
) -> None:
    """Adjacent representable bounds can share one log; the derivative stays zero."""
    dtype = _canonical_float()
    lower = dtype(start)
    upper = np.nextafter(lower, dtype(np.inf))
    assert jnp.log(jnp.asarray(lower)) == jnp.log(jnp.asarray(upper))
    grid = LogSpacedGrid(start=float(lower), stop=float(upper), n_points=1)
    queries = jnp.asarray([lower, upper, 2 * upper], dtype=dtype)
    derivative = jax.vmap(jax.grad(grid.get_coordinate))(queries)
    np.testing.assert_array_equal(derivative, np.zeros(3))


def test_logspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LogSpacedGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_rejects_zero_start():
    with pytest.raises(GridInitializationError, match="log-spaced grid"):
        LogSpacedGrid(start=0, stop=10, n_points=5)


def test_logspace_grid_rejects_negative_start():
    with pytest.raises(GridInitializationError, match="log-spaced grid"):
        LogSpacedGrid(start=-1.0, stop=10, n_points=5)


def test_validate_continuous_grid_rejects_nan_start():
    with pytest.raises(GridInitializationError, match="start must be finite"):
        _validate_continuous_grid(
            start=jnp.asarray(float("nan")),
            stop=jnp.asarray(10.0),
            n_points=jnp.int32(5),
        )


def test_validate_continuous_grid_rejects_inf_stop():
    with pytest.raises(GridInitializationError, match="stop must be finite"):
        _validate_continuous_grid(
            start=jnp.asarray(1.0),
            stop=jnp.asarray(float("inf")),
            n_points=jnp.int32(5),
        )


def _canonical_float() -> type[np.floating]:
    return np.float64 if X64_ENABLED else np.float32


def test_irreg_spaced_grid_rejects_points_that_round_to_one_node() -> None:
    """Distinct integers past the float mantissa round to one stored node."""
    lower = 2 ** (np.finfo(_canonical_float()).nmant + 1)
    with pytest.raises(GridInitializationError, match="strictly ascending"):
        IrregSpacedGrid(points=(lower, lower + 1))


def test_irreg_spaced_grid_keeps_adjacent_representable_points() -> None:
    dtype = _canonical_float()
    upper = np.nextafter(dtype(1.0), dtype(np.inf))
    grid = IrregSpacedGrid(points=(1.0, float(upper)))
    np.testing.assert_array_equal(grid.get_coordinate(grid.to_jax()), [0.0, 1.0])


def test_irreg_spaced_grid_rejects_nan_points():
    with pytest.raises(GridInitializationError, match="must be finite"):
        IrregSpacedGrid(points=(1.0, float("nan"), 3.0))


def test_irreg_spaced_grid_rejects_inf_points():
    with pytest.raises(GridInitializationError, match="must be finite"):
        IrregSpacedGrid(points=(1.0, 2.0, float("inf")))


def test_replace_mixin():
    grid = LinSpacedGrid(start=1, stop=5, n_points=5)
    new_grid = grid.replace(start=0)
    assert new_grid.start == 0
    assert new_grid.stop == 5
    assert new_grid.n_points == 5


def test_irreg_spaced_grid_creation():
    grid = IrregSpacedGrid(points=(1.0, 2.0, 5.0, 10.0))
    assert np.allclose(grid.to_jax(), np.array([1.0, 2.0, 5.0, 10.0]))
    assert grid.n_points == 4


def test_irreg_spaced_grid_invalid_too_few_points():
    with pytest.raises(GridInitializationError, match="at least 2 elements"):
        IrregSpacedGrid(points=(1.0,))


def test_irreg_spaced_grid_invalid_non_numeric():
    with pytest.raises(GridInitializationError, match="points"):
        IrregSpacedGrid(points=(1.0, "a", 3.0))  # ty: ignore[invalid-argument-type]


def test_irreg_spaced_grid_invalid_not_ascending():
    with pytest.raises(GridInitializationError, match="strictly ascending order"):
        IrregSpacedGrid(points=(1.0, 3.0, 2.0))


def _create_equivalent_grid(
    *, grid_type: str, lin_grid: LinSpacedGrid
) -> IrregSpacedGrid | PiecewiseLinSpacedGrid:
    """Create a grid equivalent to the given LinSpacedGrid."""
    if grid_type == "IrregSpacedGrid":
        return IrregSpacedGrid(points=tuple(float(x) for x in lin_grid.to_jax()))
    if grid_type == "PiecewiseLinSpacedGrid":
        return PiecewiseLinSpacedGrid(
            start=lin_grid.start,
            stop=lin_grid.stop,
            breakpoints=(),
            points_per_segment=(lin_grid.n_points,),
        )
    msg = f"Unknown grid type: {grid_type}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("start", "stop", "n_points"),
    [
        (0.0, 1.0, 5),
        (1.0, 100.0, 10),
        (-10.0, 10.0, 21),
        (0.5, 2.5, 3),
    ],
)
@pytest.mark.parametrize("grid_type", ["IrregSpacedGrid", "PiecewiseLinSpacedGrid"])
def test_linspaced_coordinates_match_other_grid_types(
    *, start: float, stop: float, n_points: int, grid_type: str
):
    """LinSpacedGrid coordinates should match equivalent grids of other types."""
    lin_grid = LinSpacedGrid(start=start, stop=stop, n_points=n_points)
    other_grid = _create_equivalent_grid(grid_type=grid_type, lin_grid=lin_grid)

    # Generate test values: grid points, interpolation, and extrapolation
    gridpoints = [float(x) for x in lin_grid.to_jax()]
    step = (stop - start) / (n_points - 1)

    # Interpolation: midpoints between consecutive grid points
    interpolation_values = [
        (gridpoints[i] + gridpoints[i + 1]) / 2 for i in range(n_points - 1)
    ]

    # Extrapolation: outside the grid range
    extrapolation_values = [start - step, start - 0.1, stop + 0.1, stop + step]

    all_test_values = gridpoints + interpolation_values + extrapolation_values

    # Tolerance depends on precision and grid value magnitude
    base_rtol = 1e-6 if X64_ENABLED else 1e-4
    max_magnitude = max(abs(start), abs(stop), 1.0)
    rtol = base_rtol * max_magnitude

    for value in all_test_values:
        value_jax = jnp.asarray(value)
        lin_coord = float(lin_grid.get_coordinate(value_jax))
        other_coord = float(other_grid.get_coordinate(value_jax))
        assert np.isclose(lin_coord, other_coord, rtol=rtol), (
            f"Mismatch at value {value} for {grid_type} vs LinSpacedGrid "
            f"({start}, {stop}, {n_points}): "
            f"LinSpacedGrid={lin_coord}, {grid_type}={other_coord}"
        )


def test_lin_spaced_grid_get_coordinate_with_array():
    grid = LinSpacedGrid(start=1, stop=2, n_points=6)
    values = jnp.array([1.0, 1.2, 1.5])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.0, 1.0, 2.5])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_log_spaced_grid_get_coordinate_with_array():
    grid = LogSpacedGrid(start=1, stop=100, n_points=7)
    points = grid.to_jax()
    mid = (float(points[1]) + float(points[2])) / 2
    coords = grid.get_coordinate(jnp.array([mid]))
    aaae(coords, jnp.array([1.5]), decimal=DECIMAL_PRECISION)


def test_irreg_spaced_grid_get_coordinate_with_array():
    grid = IrregSpacedGrid(points=[0.0, 1.0, 3.0, 6.0])
    values = jnp.array([0.5, 3.0, 4.5])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.5, 2.0, 2.5])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_piecewise_lin_spaced_grid_get_coordinate_with_array():
    grid = PiecewiseLinSpacedGrid(
        start=0,
        stop=10,
        breakpoints=(),
        points_per_segment=(11,),
    )
    values = jnp.array([0.0, 5.0, 10.0])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.0, 5.0, 10.0])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_piecewise_log_spaced_grid_get_coordinate_with_array():
    grid = PiecewiseLogSpacedGrid(
        start=1,
        stop=100,
        breakpoints=(),
        points_per_segment=(3,),
    )
    points = grid.to_jax()
    values = jnp.array([float(points[0]), float(points[1]), float(points[2])])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.0, 1.0, 2.0])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_discrete_grid_rejects_batch_size_combined_with_distributed():
    """Retired execution keywords cannot enter a discrete grid declaration."""
    with pytest.raises(TypeError, match="batch_size"):
        DiscreteGrid(
            category_class=_make_dc("_BS", ("a", jnp.int32(0)), ("b", jnp.int32(1))),
            batch_size=1,  # ty: ignore[unknown-argument]
            distributed=True,  # ty: ignore[unknown-argument]
        )


def test_discrete_grid_has_no_execution_properties():
    """The outcome definition carries no private or public execution settings."""
    grid = DiscreteGrid(
        category_class=_make_dc("_OK", ("a", jnp.int32(0)), ("b", jnp.int32(1))),
    )
    assert not any("batch_size" in name or "distributed" in name for name in vars(grid))


@pytest.mark.parametrize(
    "make_grid",
    [
        pytest.param(
            lambda: LinSpacedGrid(start=1, stop=10, n_points=4, distributed=True),  # ty: ignore[unknown-argument]
            id="LinSpacedGrid",
        ),
        pytest.param(
            lambda: LogSpacedGrid(start=1, stop=10, n_points=4, distributed=True),  # ty: ignore[unknown-argument]
            id="LogSpacedGrid",
        ),
        pytest.param(
            lambda: IrregSpacedGrid(points=[1.0, 2.0, 3.0, 4.0], distributed=True),  # ty: ignore[unknown-argument]
            id="IrregSpacedGrid",
        ),
        pytest.param(
            lambda: PiecewiseLinSpacedGrid(
                start=0,
                stop=10,
                breakpoints=(),
                points_per_segment=(11,),
                distributed=True,  # ty: ignore[unknown-argument]
            ),
            id="PiecewiseLinSpacedGrid",
        ),
        pytest.param(
            lambda: PiecewiseLogSpacedGrid(
                start=1,
                stop=100,
                breakpoints=(),
                points_per_segment=(3,),
                distributed=True,  # ty: ignore[unknown-argument]
            ),
            id="PiecewiseLogSpacedGrid",
        ),
    ],
)
def test_continuous_grid_rejects_distributed_true(make_grid):
    """Every continuous grid constructor refuses the retired placement keyword."""
    with pytest.raises(TypeError, match="distributed"):
        make_grid()


def test_value_grid_codes_index_the_declared_values():
    """A value grid's codes are `0..n-1` in the order of its values."""
    grid = DiscreteGrid(values=[0, 20, 40])
    assert grid.codes == (0, 1, 2)


def test_value_grid_keeps_the_declared_values():
    """`values` returns the declared numbers as Python scalars."""
    grid = DiscreteGrid(values=[np.int64(0), 20, 40])
    assert grid.values == (0, 20, 40)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param([0, 20, 40], ("0", "20", "40"), id="int"),
        pytest.param([np.int64(0), np.int32(20)], ("0", "20"), id="numpy-int"),
        pytest.param([0.0, 0.25], ("0.0", "0.25"), id="float"),
        pytest.param([np.float64(0.5), 1.0], ("0.5", "1.0"), id="numpy-float"),
        pytest.param(range(3), ("0", "1", "2"), id="range"),
        pytest.param(np.linspace(0, 1, 3), ("0.0", "0.5", "1.0"), id="numpy-array"),
        pytest.param(jnp.arange(3), ("0", "1", "2"), id="jax-array"),
    ],
)
def test_value_grid_labels_its_categories_with_the_python_scalar_repr(
    *, values, expected
):
    """Each category label is the `repr` of the declared Python scalar."""
    assert DiscreteGrid(values=values).categories == expected


def test_value_grid_is_ordered():
    """Strictly ascending values give an ordered grid."""
    assert DiscreteGrid(values=[0.0, 0.5]).ordered is True


def test_category_grid_has_no_values():
    """A grid declared with a category class carries no values."""
    assert DiscreteGrid(category_class=_Category).values is None


def test_value_grid_to_jax_returns_int32_codes():
    """The engine representation of a value grid stays its int32 codes."""
    codes = DiscreteGrid(values=[0, 20, 40]).to_jax()
    assert (codes.dtype, codes.tolist()) == (jnp.int32, [0, 1, 2])


def test_integer_value_grid_values_to_jax_returns_int32_values():
    """Integer values are published as int32."""
    values = DiscreteGrid(values=[0, 20, 40]).values_to_jax()
    assert (values.dtype, values.tolist()) == (jnp.int32, [0, 20, 40])


def test_float_value_grid_values_to_jax_uses_the_canonical_float_dtype():
    """Float values are published at the canonical float dtype."""
    values = DiscreteGrid(values=[0.0, 0.25]).values_to_jax()
    assert values.dtype == canonical_float_dtype()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"values": []}, "at least one value", id="empty"),
        pytest.param({"values": np.zeros((2, 2))}, "one-dimensional", id="2d-array"),
        pytest.param({"values": [0, 40, 20]}, "strictly ascending", id="unsorted"),
        pytest.param({"values": [0, 20, 20]}, "strictly ascending", id="duplicate"),
        pytest.param({"values": [0, 0.5]}, "all integers or all floats", id="mixed"),
        pytest.param({"values": [True, False]}, "bool", id="bool"),
        pytest.param({"values": [0.0, float("nan")]}, "finite", id="nan"),
        pytest.param({"values": [0.0, float("inf")]}, "finite", id="inf"),
        pytest.param({"values": [0, 2**31]}, "int32", id="int32-overflow"),
        pytest.param(
            {"values": [0, 1], "category_class": _Category},
            "either `category_class` or `values`",
            id="with-category-class",
        ),
        pytest.param({"values": [0, 1], "tolerance": 0.1}, "Integer", id="int-tol"),
        pytest.param({"values": [0.0, 1.0], "tolerance": -0.1}, ">= 0", id="neg-tol"),
        pytest.param(
            {"values": [0.0, 1.0], "tolerance": float("nan")}, ">= 0", id="nan-tol"
        ),
        pytest.param(
            {"values": [0.0, 1.0], "tolerance": 0.5}, "half the smallest", id="wide-tol"
        ),
        pytest.param(
            {"values": [0.0, 1e6], "tolerance": 1e-12}, "spacing", id="sub-ulp-tol"
        ),
    ],
)
def test_value_grid_rejects_invalid_declarations(*, kwargs, match):
    """Invalid values or tolerances raise `GridInitializationError`."""
    with pytest.raises(GridInitializationError, match=match):
        DiscreteGrid(**kwargs)


@pytest.mark.skipif(X64_ENABLED, reason="the two values are distinct at float64")
def test_value_grid_rejects_values_that_collide_at_the_canonical_dtype():
    """Values that round to one float32 node are refused at float32."""
    with pytest.raises(GridInitializationError, match="strictly ascending"):
        DiscreteGrid(values=[1e8, 1e8 + 1])


def test_discrete_grid_requires_category_class_or_values():
    """A grid without a category class or values is refused."""
    with pytest.raises(TypeError, match="either `category_class` or `values`"):
        DiscreteGrid()


def test_integer_value_grid_encodes_exact_members_and_flags_others():
    """Integers map to their code by exact membership; others get the sentinel."""
    codes = DiscreteGrid(values=[0, 20, 40]).encode(jnp.array([40, 20, 7, -20]))
    assert codes.tolist() == [2, 1, MISSING_CAT_CODE, MISSING_CAT_CODE]


def test_integer_value_grid_refuses_float_input():
    """An integer grid matches integers only; a float input is a type error."""
    with pytest.raises(TypeError, match="integer"):
        DiscreteGrid(values=[0, 20, 40]).encode(jnp.array([20.0]))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(20, id="python-int"),
        pytest.param(np.array([20]), id="numpy-array"),
        pytest.param(np.int64(20), id="numpy-scalar"),
    ],
)
def test_integer_value_grid_encodes_host_input(value):
    """Python and NumPy integers encode like JAX arrays."""
    code = DiscreteGrid(values=[0, 20, 40]).encode(value)
    assert np.ravel(code).tolist() == [1]


def test_float_value_grid_encodes_exact_nodes():
    """A float equal to a node at the canonical dtype maps to its code."""
    grid = DiscreteGrid(values=[0.0, 0.7])
    codes = grid.encode(jnp.array([0.7, 0.0], dtype=canonical_float_dtype()))
    assert codes.tolist() == [1, 0]


def test_float_value_grid_without_tolerance_flags_one_ulp_off():
    """With the default tolerance a value one ULP above a node is off the grid."""
    node = jnp.asarray(0.7, dtype=canonical_float_dtype())
    code = DiscreteGrid(values=[0.0, 0.7]).encode(jnp.nextafter(node, jnp.inf))
    assert int(code) == MISSING_CAT_CODE


def test_float_value_grid_tolerance_admits_a_value_near_a_node():
    """A value within the declared tolerance maps to the nearest node."""
    node = jnp.asarray(0.7, dtype=canonical_float_dtype())
    grid = DiscreteGrid(values=[0.0, 0.7], tolerance=1e-6)
    assert int(grid.encode(jnp.nextafter(node, jnp.inf))) == 1


def test_float_value_grid_without_tolerance_flags_one_ulp_above_the_smallest_normal():
    """The difference to the smallest normal node is subnormal and must not flush."""
    finfo = np.finfo(_canonical_float())
    grid = DiscreteGrid(values=[float(finfo.tiny), 1.0])
    node = jnp.asarray(finfo.tiny, dtype=canonical_float_dtype())
    assert grid.encode(jnp.stack([node, jnp.nextafter(node, jnp.inf)])).tolist() == [
        0,
        MISSING_CAT_CODE,
    ]


def test_float_value_grid_tolerance_holds_at_the_smallest_normal():
    """A subnormal distance is compared exactly against a subnormal tolerance.

    The nodes are the smallest normal and twice it; their spacing allows a
    tolerance of two subnormal steps. Two steps above the lower node matches it,
    three steps do not.
    """
    finfo = np.finfo(_canonical_float())
    step = float(finfo.smallest_subnormal)
    grid = DiscreteGrid(
        values=[float(finfo.tiny), 2 * float(finfo.tiny)], tolerance=2 * step
    )
    queries = jnp.asarray(
        [float(finfo.tiny) + 2 * step, float(finfo.tiny) + 3 * step],
        dtype=canonical_float_dtype(),
    )
    assert grid.encode(queries).tolist() == [0, MISSING_CAT_CODE]


@pytest.mark.parametrize("value", [0.3, -0.5, 2.0, float("nan")])
def test_float_value_grid_tolerance_flags_values_between_nodes(value):
    """Values farther than the tolerance from every node get the sentinel."""
    grid = DiscreteGrid(values=[0.0, 0.25, 0.5, 1.0], tolerance=1e-6)
    code = grid.encode(jnp.asarray(value, dtype=canonical_float_dtype()))
    assert int(code) == MISSING_CAT_CODE


def test_float_value_grid_decodes_codes_to_values():
    """Codes decode to the values they index."""
    values = DiscreteGrid(values=[0.0, 0.25, 0.5]).decode(jnp.array([2, 0]))
    aaae(values, [0.5, 0.0], decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("code", [MISSING_CAT_CODE, -1, 3])
def test_float_value_grid_decodes_off_grid_codes_to_nan(code):
    """A code outside `0..n-1` reads NaN, never a clamped or wrapped node."""
    value = DiscreteGrid(values=[0.0, 0.25, 0.5]).decode(jnp.int32(code))
    assert bool(jnp.isnan(value))


@pytest.mark.parametrize("code", [MISSING_CAT_CODE, -1, 3])
def test_integer_value_grid_decodes_off_grid_codes_to_the_sentinel(code):
    """An integer grid reads the int32 sentinel for a code outside `0..n-1`."""
    value = DiscreteGrid(values=[0, 20, 40]).decode(jnp.int32(code))
    assert int(value) == MISSING_CAT_CODE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.3, 0.25, id="interior"),
        pytest.param(0.375, 0.25, id="tie-to-lower"),
        pytest.param(0.9, 1.0, id="upper"),
        pytest.param(-5.0, 0.0, id="below"),
        pytest.param(7.0, 1.0, id="above"),
    ],
)
def test_float_value_grid_nearest_snaps_to_the_closest_node(*, value, expected):
    """`nearest` returns the closest node, the lower one on a tie."""
    grid = DiscreteGrid(values=[0.0, 0.25, 0.5, 1.0])
    snapped = grid.nearest(jnp.asarray(value, dtype=canonical_float_dtype()))
    assert float(snapped) == expected


def test_float_value_grid_nearest_keeps_nan():
    """`nearest` passes NaN through, so the encoder still flags it."""
    grid = DiscreteGrid(values=[0.0, 1.0])
    snapped = grid.nearest(jnp.asarray(jnp.nan, dtype=canonical_float_dtype()))
    assert bool(jnp.isnan(snapped))


def test_integer_value_grid_refuses_nearest():
    """`nearest` serves float grids; an integer law returns exact integers."""
    with pytest.raises(TypeError, match="float"):
        DiscreteGrid(values=[0, 20, 40]).nearest(jnp.int32(25))


def test_float_value_grid_nearest_maps_infinity_to_the_end_nodes():
    """`nearest` sends `-inf` to the first node and `inf` to the last."""
    grid = DiscreteGrid(values=[0.0, 0.25, 1.0])
    values = jnp.asarray([-jnp.inf, jnp.inf], dtype=canonical_float_dtype())
    assert grid.nearest(values).tolist() == [0.0, 1.0]


def test_value_grid_nearest_result_encodes_to_its_node():
    """A snapped value always encodes to a valid code."""
    grid = DiscreteGrid(values=[0.0, 0.25, 0.5, 1.0])
    values = jnp.asarray([0.3, 0.9, -1.0], dtype=canonical_float_dtype())
    assert grid.encode(grid.nearest(values)).tolist() == [1, 3, 0]


def test_value_grid_encode_agrees_under_jit():
    """The compiled encoder gives the eager result."""
    grid = DiscreteGrid(values=[0.0, 0.25, 0.5, 1.0], tolerance=1e-6)
    values = jnp.asarray([0.25, 0.3, 1.0, -1.0], dtype=canonical_float_dtype())
    assert jax.jit(grid.encode)(values).tolist() == grid.encode(values).tolist()


def test_value_grid_decode_and_nearest_agree_under_jit():
    """The compiled decoder and snapper give the eager results."""
    grid = DiscreteGrid(values=[0.0, 0.25, 0.5, 1.0])
    codes = jnp.array([3, -1, 0], dtype=jnp.int32)
    values = jnp.asarray([0.3, 0.9], dtype=canonical_float_dtype())
    got = (jax.jit(grid.decode)(codes), jax.jit(grid.nearest)(values))
    want = (grid.decode(codes), grid.nearest(values))
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
    np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))


def test_value_grid_sentinel_is_the_simulation_missing_code():
    """Off-grid values and missing initial conditions share one sentinel code."""
    assert MISSING_CAT_CODE == initial_conditions.MISSING_CAT_CODE


def _exact_nearest_value(*, nodes: tuple[float, ...], value: float) -> float:
    """Return the nearest node by exact rational distance, the lower on a tie."""
    if math.isnan(value):
        return math.nan
    if math.isinf(value):
        return nodes[0] if value < 0 else nodes[-1]
    exact = Fraction(value)
    distances = [abs(exact - Fraction(node)) for node in nodes]
    return nodes[min(range(len(nodes)), key=lambda index: (distances[index], index))]


def _exact_downward_midpoints(
    *, nodes: tuple[float, ...], dtype: type[np.floating]
) -> tuple[float, ...]:
    """Return the largest float of `dtype` at or below each exact midpoint."""
    cuts = []
    for left, right in itertools.pairwise(nodes):
        midpoint = (Fraction(left) + Fraction(right)) / 2
        cut = float(dtype(float(midpoint)))
        while Fraction(cut) > midpoint:
            cut = float(np.nextafter(dtype(cut), dtype(-math.inf)))
        while Fraction(float(np.nextafter(dtype(cut), dtype(math.inf)))) <= midpoint:
            cut = float(np.nextafter(dtype(cut), dtype(math.inf)))
        cuts.append(cut)
    return tuple(cuts)


def _canonical_np_float() -> type[np.floating]:
    return np.float64 if X64_ENABLED else np.float32


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize(
    ("integer_dtype", "value"),
    [(np.int64, 2**32 + 20), (np.int64, -(2**32) + 20), (np.uint64, 2**32 + 20)],
)
@pytest.mark.parametrize("route", ["eager", "jit_closure"])
def test_integer_value_grid_keeps_wide_host_integers_off_the_grid(
    *, shape, integer_dtype, value, route
):
    """A wide host integer whose low int32 word is a node does not match it."""
    grid = DiscreteGrid(values=[0, 20, 40])
    query = np.asarray(value, dtype=integer_dtype).reshape(shape)
    observed = np.asarray(
        jax.jit(lambda: grid.encode(query))()
        if route == "jit_closure"
        else grid.encode(query)
    )
    np.testing.assert_array_equal(observed, np.full(shape, MISSING_CAT_CODE))


def test_integer_value_grid_matches_across_int32_boundaries():
    """Membership follows the original integer on both sides of the int32 range."""
    nodes = (-(2**31) + 1, -1, 0, 20, 2**31 - 1)
    query = np.asarray(
        [-(2**31) - 1, -(2**31), -(2**31) + 1, -1, 0, 20, 2**31 - 1, 2**31, 2**32 + 20],
        dtype=np.int64,
    )
    expected = [nodes.index(v) if v in nodes else MISSING_CAT_CODE for v in query]
    observed = DiscreteGrid(values=nodes).encode(query)
    assert np.asarray(observed).tolist() == expected


@pytest.mark.parametrize("integer_dtype", [np.int64, np.uint64])
def test_float_value_grid_reads_wide_host_integers_as_floats(integer_dtype):
    """A wide host integer reaches a float grid as its own float, not its low word."""
    grid = DiscreteGrid(values=[0.0, 20.0, 40.0])
    query = np.asarray([2**32 + 20], dtype=integer_dtype)
    assert (
        np.asarray(grid.encode(query)).tolist(),
        np.asarray(grid.nearest(query)).tolist(),
    ) == ([MISSING_CAT_CODE], [40.0])


@pytest.mark.parametrize("float_grid", [False, True])
@pytest.mark.parametrize("integer_dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("route", ["eager", "jit_closure"])
def test_value_grid_decodes_wide_host_codes_as_off_grid(
    *, float_grid, integer_dtype, route
):
    """A host code outside `0..n-1` reads off-grid even when its low word is valid."""
    alias = -(2**32) if integer_dtype == np.int64 else 2**33
    codes = np.asarray([[0, 2**32, alias], [1, 2**32 + 1, alias + 1]], integer_dtype)
    nodes = (0.0, 0.25, 0.5) if float_grid else (0, 20, 40)
    off = math.nan if float_grid else MISSING_CAT_CODE
    grid = DiscreteGrid(values=nodes)
    observed = np.asarray(
        jax.jit(lambda: grid.decode(codes))()
        if route == "jit_closure"
        else grid.decode(codes)
    )
    expected = np.asarray(
        [nodes[int(c)] if 0 <= int(c) < 3 else off for c in codes.flat]
    ).reshape(codes.shape)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(X64_ENABLED, reason="these floats are finite at float64")
@pytest.mark.parametrize("values", [(1e39,), (-1e39,), (0.0, 1e39), (-1e39, 0.0, 1e39)])
def test_value_grid_refuses_finite_values_that_overflow_storage(values):
    """Finite declared values that round to infinity at the canonical dtype raise."""
    with pytest.raises(GridInitializationError, match="finite"):
        DiscreteGrid(values=values)


def test_value_grid_keeps_finite_values_at_both_dtype_extremes():
    """The largest finite values of the canonical dtype remain valid nodes."""
    largest = float(np.finfo(_canonical_np_float()).max)
    grid = DiscreteGrid(values=[-largest, 0.0, largest])
    assert np.asarray(grid.values_to_jax()).tolist() == [-largest, 0.0, largest]


def test_value_grid_refuses_tolerance_above_an_overflowing_half_gap():
    """A gap that overflows when subtracted still bounds the tolerance exactly."""
    magnitude, tolerance = (1e308, 1.5e308) if X64_ENABLED else (3e38, 3.1e38)
    with pytest.raises(GridInitializationError, match="half the smallest gap"):
        DiscreteGrid(values=[-magnitude, magnitude], tolerance=tolerance)


@pytest.mark.parametrize(
    "declared",
    [(0.0, 1.0), (1024.0, 1028.0, 1040.0), (0.125, 0.5, 2.0), "wide"],
)
@pytest.mark.parametrize("side", ["below", "equal", "above"])
def test_value_grid_tolerance_stays_strictly_below_the_exact_half_gap(
    *, declared, side
):
    """Only a tolerance strictly below the exact half gap is accepted."""
    dtype = _canonical_np_float()
    if declared == "wide":
        magnitude = 1e308 if X64_ENABLED else 3e38
        declared = (-magnitude, magnitude)
    nodes = tuple(float(dtype(node)) for node in declared)
    half_gap = float(
        min(
            Fraction(right) - Fraction(left)
            for left, right in itertools.pairwise(nodes)
        )
        / 2
    )
    tolerance = (
        half_gap
        if side == "equal"
        else float(
            np.nextafter(
                dtype(half_gap), dtype(-math.inf if side == "below" else math.inf)
            )
        )
    )
    if side == "below":
        assert DiscreteGrid(values=nodes, tolerance=tolerance).tolerance == tolerance
    else:
        with pytest.raises(GridInitializationError, match="half the smallest gap"):
            DiscreteGrid(values=nodes, tolerance=tolerance)


@pytest.mark.parametrize("route", ["eager", "jit", "vmap"])
def test_float_value_grid_nearest_tells_apart_both_sides_of_an_exact_midpoint(route):
    """A query just above the midpoint of two nodes snaps up, just below snaps down."""
    dtype = _canonical_np_float()
    tiny = 2.0 ** (-55 if X64_ENABLED else -26)
    queries = jnp.asarray([[-tiny, 0.0, tiny], [-math.inf, math.inf, math.nan]], dtype)
    grid = DiscreteGrid(values=[-1.0, 1.0])
    observed = np.asarray(
        _apply_route(function=grid.nearest, values=queries, route=route)
    )
    expected = np.asarray(
        [
            _exact_nearest_value(nodes=(-1.0, 1.0), value=float(q))
            for q in queries.ravel()
        ],
        dtype,
    ).reshape(queries.shape)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "case",
    [
        "scaled_small",
        "scaled_large",
        "translated_negative",
        "translated_positive",
        "nonuniform",
        "adjacent",
        "wide_signed",
    ],
)
@pytest.mark.parametrize("route", ["eager", "jit", "vmap"])
def test_float_value_grid_nearest_matches_exact_distances_at_every_cutoff(
    *, case, route
):
    """Each represented neighbour of every midpoint snaps to its exact nearest node."""
    dtype = _canonical_np_float()
    nodes = _differential_nodes(case=case, dtype=dtype)
    queries = list(nodes)
    for cut in _exact_downward_midpoints(nodes=nodes, dtype=dtype):
        queries += [
            float(np.nextafter(dtype(cut), dtype(-math.inf))),
            cut,
            float(np.nextafter(dtype(cut), dtype(math.inf))),
        ]
    queries += [-math.inf, math.inf, math.nan]
    grid = DiscreteGrid(values=nodes)
    observed = np.asarray(
        _apply_route(
            function=grid.nearest, values=jnp.asarray(queries, dtype), route=route
        )
    )
    expected = np.asarray(
        [_exact_nearest_value(nodes=nodes, value=q) for q in queries], dtype
    )
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("route", ["eager", "jit", "vmap"])
def test_singleton_float_value_grid_nearest_keeps_endpoints_and_nan(route):
    """A one-node grid snaps every finite or infinite query to it and keeps NaN."""
    dtype = _canonical_np_float()
    queries = jnp.asarray([-math.inf, -1.0, 2.0, 8.0, math.inf, math.nan], dtype)
    observed = np.asarray(
        _apply_route(
            function=DiscreteGrid(values=[2.0]).nearest, values=queries, route=route
        )
    )
    np.testing.assert_array_equal(observed, [2.0, 2.0, 2.0, 2.0, 2.0, math.nan])


def _differential_nodes(*, case, dtype):
    """Finite normal grids with different scales and gap geometries."""
    if case in ("scaled_small", "scaled_large"):
        exponent = 400 if dtype == np.float64 else 80
        scale = 2.0 ** (-exponent if case == "scaled_small" else exponent)
        declared = tuple(scale * value for value in (0.5, 1.0, 1.75, 4.0))
    elif case in ("translated_negative", "translated_positive"):
        shift = -1024.0 if case == "translated_negative" else 1024.0
        declared = tuple(shift + value for value in (0.0, 1.0, 3.0, 7.0))
    elif case == "nonuniform":
        declared = (0.125, 0.5, 1.0, 16.0, 1024.0)
    elif case == "adjacent":
        steps = [dtype(1.0)]
        for _ in range(6):
            steps.append(np.nextafter(steps[-1], dtype(math.inf)))
        declared = tuple(float(steps[index]) for index in (0, 1, 2, 6))
    else:
        scale = 2.0 ** (1022 if dtype == np.float64 else 126)
        declared = (-3.0 * scale, 2.0 * scale)
    return tuple(float(dtype(node)) for node in declared)


def _apply_route(*, function, values, route):
    if route == "jit":
        return jax.jit(function)(values)
    if route == "vmap":
        return jax.vmap(function)(values)
    return function(values)


_NARROW_NODE_COUNTS = (127, 128, 255, 256, 257, 383, 384, 511, 512)


def _range_grid(*, n_nodes, kind):
    values = tuple(range(n_nodes))
    return DiscreteGrid(
        values=tuple(float(value) for value in values) if kind == "float" else values
    )


def _assert_decodes_like_python_integers(*, decoded, codes, n_nodes, kind):
    """Compare with the original codes' Python integers, before any narrowing."""
    expected = [
        float(int(code))
        if 0 <= int(code) < n_nodes
        else (math.nan if kind == "float" else MISSING_CAT_CODE)
        for code in codes.flat
    ]
    np.testing.assert_array_equal(
        np.asarray(decoded, dtype=float).ravel(), np.asarray(expected, dtype=float)
    )


@pytest.mark.parametrize("n_nodes", _NARROW_NODE_COUNTS)
@pytest.mark.parametrize("kind", ["integer", "float"])
@pytest.mark.parametrize("route", ["eager", "jit_closure"])
def test_value_grid_decodes_narrow_host_codes_off_the_axis_as_off_grid(
    *, n_nodes, kind, route
):
    """An int8 code off the axis stays off-grid even when `n` does not fit int8."""
    grid = _range_grid(n_nodes=n_nodes, kind=kind)
    for codes in (
        np.asarray(-1, dtype=np.int8),
        np.asarray([[-128, -1, 0], [1, 126, 127]], dtype=np.int8),
    ):
        decoded = (
            jax.jit(lambda codes=codes: grid.decode(codes))()
            if route == "jit_closure"
            else grid.decode(codes)
        )
        _assert_decodes_like_python_integers(
            decoded=decoded, codes=codes, n_nodes=n_nodes, kind=kind
        )


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
@pytest.mark.parametrize("n_nodes", [127, 256, 257])
@pytest.mark.parametrize("kind", ["integer", "float"])
@pytest.mark.parametrize("route", ["eager", "jit", "vmap"])
def test_value_grid_decodes_narrow_device_codes_by_their_integer_bounds(
    *, dtype, n_nodes, kind, route
):
    """Narrow device codes are bounded as integers, not in their own dtype."""
    info = np.iinfo(dtype)
    codes = np.asarray(
        [int(info.min), 0, 1, min(126, int(info.max)), int(info.max)], dtype=dtype
    )
    grid = _range_grid(n_nodes=n_nodes, kind=kind)
    decoded = _apply_route(function=grid.decode, values=jnp.asarray(codes), route=route)
    _assert_decodes_like_python_integers(
        decoded=decoded, codes=codes, n_nodes=n_nodes, kind=kind
    )


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("kind", ["integer", "float"])
@pytest.mark.parametrize("route", ["eager", "jit_closure"])
def test_value_grid_decodes_wide_host_codes_by_their_original_bounds(
    *, dtype, kind, route
):
    """Wide host codes are bounded before any narrowing, around both int32 edges."""
    values = [0, 1, 2, 3, 2**31, 2**32, 2**32 + 1, 2**32 + 2]
    values += [-1, -(2**31), -(2**32)] if dtype is np.int64 else [2**64 - 1]
    codes = np.asarray(values, dtype=dtype)
    grid = _range_grid(n_nodes=3, kind=kind)
    decoded = (
        jax.jit(lambda: grid.decode(codes))()
        if route == "jit_closure"
        else grid.decode(codes)
    )
    _assert_decodes_like_python_integers(
        decoded=decoded, codes=codes, n_nodes=3, kind=kind
    )


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16])
def test_integer_value_grid_encodes_narrow_host_integers_by_membership(dtype):
    """Narrow host integers match exactly the nodes they equal."""
    info = np.iinfo(dtype)
    raw = [int(info.min), 0, 1, 20, int(info.max)]
    nodes = (0, 1, 20, 300)
    expected = [nodes.index(v) if v in nodes else MISSING_CAT_CODE for v in raw]
    encoded = DiscreteGrid(values=nodes).encode(np.asarray(raw, dtype=dtype))
    assert np.asarray(encoded).tolist() == expected


_MAX_NODE = 2**31 - 1
_NEGATIVE_NODES = (MISSING_CAT_CODE + 1, -65537, -65536, -257, -256, -129, -128, -2, -1)
_INTEGER_GRIDS = {
    "singleton": (-1,),
    "negative_family": _NEGATIVE_NODES,
    "mixed_boundaries": (*_NEGATIVE_NODES, 0, 1, 20, _MAX_NODE),
}
_INTEGER_DTYPES = [
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
]


def _reference_encode(*, nodes, query):
    """The code of the node equal to the original integer, else the missing code."""
    return nodes.index(query) if query in nodes else MISSING_CAT_CODE


def _reference_decode(*, nodes, code, floating):
    """The node a code indexes, bounds checked on the original integer."""
    if 0 <= code < len(nodes):
        return nodes[code]
    return math.nan if floating else MISSING_CAT_CODE


def _representable(*, dtype, candidates):
    info = np.iinfo(dtype)
    return sorted({c for c in candidates if info.min <= c <= info.max})


def _integer_routes(*, function, host, dtype):
    """Evaluate `function` on host and device routes the precision can represent."""
    results = {
        "host_eager": function(host),
        "host_jit_closure": jax.jit(functools.partial(function, host))(),
    }
    if np.dtype(dtype).itemsize < 8 or X64_ENABLED:
        device = jnp.asarray(host)
        assert np.asarray(device).tolist() == host.tolist()
        results |= {
            "device_eager": function(device),
            "device_jit": jax.jit(function)(device),
            "device_vmap": jax.vmap(function)(device),
        }
    return {route: np.asarray(result) for route, result in results.items()}


@pytest.mark.parametrize("grid_name", list(_INTEGER_GRIDS))
@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_integer_value_grid_encodes_every_integer_dtype_like_python_integers(
    *, grid_name, dtype
):
    """Each input integer type matches a node exactly when its integer equals it."""
    nodes = _INTEGER_GRIDS[grid_name]
    queries = _representable(
        dtype=dtype,
        candidates=(
            *nodes,
            *(2**32 + node for node in nodes),
            *(2**64 + node for node in nodes),
            MISSING_CAT_CODE,
            2**31,
            2**32 - 1,
            int(np.iinfo(dtype).min),
            int(np.iinfo(dtype).max),
            0,
            1,
        ),
    )
    host = np.asarray(queries, dtype=dtype)
    expected = [_reference_encode(nodes=nodes, query=q) for q in queries]
    grid = DiscreteGrid(values=nodes)
    observed = _integer_routes(function=grid.encode, host=host, dtype=dtype)
    assert {
        route: codes.tolist() for route, codes in observed.items()
    } == dict.fromkeys(observed, expected)


@pytest.mark.parametrize("kind", ["integer", "float"])
@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_value_grid_decodes_every_integer_dtype_like_python_integers(*, kind, dtype):
    """Each code type reads its node exactly when its integer lies in `0..n-1`."""
    nodes = (0, 20, 40, 60, 80) if kind == "integer" else (0.0, 0.5, 1.0, 1.5, 2.0)
    codes = _representable(
        dtype=dtype,
        candidates=(
            int(np.iinfo(dtype).min),
            -1,
            0,
            4,
            5,
            int(np.iinfo(dtype).max),
            2**32,
            2**32 + 4,
            2**64 - 1,
            -(2**32) + 4,
        ),
    )
    host = np.asarray(codes, dtype=dtype)
    expected = np.asarray(
        [
            _reference_decode(nodes=nodes, code=c, floating=kind == "float")
            for c in codes
        ],
        dtype=float,
    )
    grid = DiscreteGrid(values=nodes)
    for route, decoded in _integer_routes(
        function=grid.decode, host=host, dtype=dtype
    ).items():
        np.testing.assert_array_equal(
            np.asarray(decoded, dtype=float), expected, err_msg=route
        )


_CLASSIFY_INTEGERS = discrete_module._classify_integers


def _classify_bypassing_device_inputs(value):
    if isinstance(value, jax.Array):
        return jnp.asarray(value).astype(jnp.int32)
    return _CLASSIFY_INTEGERS(value)


def _classify_after_casting_to_int32(value):
    return _CLASSIFY_INTEGERS(jnp.asarray(value).astype(jnp.int32))


def _classify_with_a_zero_fill(value):
    classified = _CLASSIFY_INTEGERS(value)
    return jnp.where(classified == MISSING_CAT_CODE, 0, classified)


@pytest.mark.parametrize(
    "mutant",
    [
        _classify_bypassing_device_inputs,
        _classify_after_casting_to_int32,
        _classify_with_a_zero_fill,
    ],
)
def test_integer_differential_detects_each_broken_classifier(*, monkeypatch, mutant):
    """Each way of losing the original integer makes the encode differential fail."""
    monkeypatch.setattr(discrete_module, "_classify_integers", mutant)
    with pytest.raises(AssertionError):
        test_integer_value_grid_encodes_every_integer_dtype_like_python_integers(
            grid_name="mixed_boundaries", dtype=np.uint32
        )
