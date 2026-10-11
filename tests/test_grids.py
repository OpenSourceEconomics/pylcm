import dataclasses
from dataclasses import make_dataclass
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
