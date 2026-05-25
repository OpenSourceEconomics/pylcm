from dataclasses import make_dataclass

import jax.numpy as jnp
import numpy as np
import portion
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.grids import (
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseGridSegment,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
    validate_category_class,
)
from _lcm.grids.categorical import _validate_discrete_grid
from _lcm.grids.continuous import _validate_continuous_grid
from _lcm.utils.containers import get_field_names_and_values
from lcm.exceptions import CategoricalDefinitionError, GridInitializationError
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED


def _make_dc(name: str, *fields: tuple[str, object]) -> type:
    """Build a dataclass with `ScalarInt` class attrs.

    Sidesteps `@dataclass(frozen=True)`'s rejection of `jax.Array`
    defaults: we make the dataclass with no defaults, then write the
    `ScalarInt` scalars onto the class via `type.__setattr__`.
    `validate_category_class` reads field values via `getattr(cls, name)`,
    so the class attrs are what get checked.
    """
    cls = make_dataclass(name, [(fname, ScalarInt) for fname, _ in fields])
    for fname, fval in fields:
        type.__setattr__(cls, fname, fval)
    return cls


def test_get_fields_with_defaults():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 2)])
    assert get_field_names_and_values(category_class) == {"a": 1, "b": 2}


def test_get_fields_no_defaults():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert get_field_names_and_values(category_class) == {"a": None, "b": None}


def test_get_fields_instance():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert get_field_names_and_values(category_class(a=1, b=2)) == {"a": 1, "b": 2}


def test_get_fields_empty():
    category_class = make_dataclass("Category", [])
    assert get_field_names_and_values(category_class) == {}


def test_validate_discrete_grid_empty():
    category_class = make_dataclass("Category", [])
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
    """Valid category class should return empty error list."""
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", jnp.int32(1)))
    errors = validate_category_class(category_class)
    assert errors == []


def test_validate_category_class_not_dataclass():
    """Non-dataclass should return error."""

    class NotDataclass:
        a = 0
        b = 1

    errors = validate_category_class(NotDataclass)
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
    grid = DiscreteGrid(category_class)
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_discrete_grid_invalid_category_class():
    category_class = _make_dc("Category", ("a", jnp.int32(0)), ("b", "wrong_type"))
    with pytest.raises(
        GridInitializationError,
        match="Field values of the category_class must be `ScalarInt`",
    ):
        DiscreteGrid(category_class)


def test_discrete_grid_ordered_true():
    @categorical(ordered=True)
    class OrderedCat:
        low: ScalarInt
        medium: ScalarInt
        high: ScalarInt

    grid = DiscreteGrid(OrderedCat)
    assert grid.ordered is True


def test_discrete_grid_ordered_false():
    @categorical(ordered=False)
    class UnorderedCat:
        a: ScalarInt
        b: ScalarInt

    grid = DiscreteGrid(UnorderedCat)
    assert grid.ordered is False


# --- @categorical: ScalarInt annotation contract ---


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

    instance = Cat()
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
    grid_type: str, lin_grid: LinSpacedGrid
) -> IrregSpacedGrid | PiecewiseLinSpacedGrid:
    """Create a grid equivalent to the given LinSpacedGrid."""
    if grid_type == "IrregSpacedGrid":
        return IrregSpacedGrid(points=tuple(float(x) for x in lin_grid.to_jax()))
    if grid_type == "PiecewiseLinSpacedGrid":
        return PiecewiseLinSpacedGrid(
            segments=(
                PiecewiseGridSegment(
                    interval=f"[{lin_grid.start}, {lin_grid.stop}]",
                    n_points=lin_grid.n_points,
                ),
            )
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
    start: float, stop: float, n_points: int, grid_type: str
):
    """LinSpacedGrid coordinates should match equivalent grids of other types."""
    lin_grid = LinSpacedGrid(start=start, stop=stop, n_points=n_points)
    other_grid = _create_equivalent_grid(grid_type, lin_grid)

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


def test_piecewise_lin_spaced_grid_creation_with_strings():
    """PiecewiseLinSpacedGrid can be created with string intervals."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[1, 4)", n_points=3),
            PiecewiseGridSegment(interval="[4, 10]", n_points=7),
        )
    )
    assert grid.n_points == 10


def test_piecewise_lin_spaced_grid_creation_with_portion_objects():
    """PiecewiseLinSpacedGrid can be created with portion.Interval objects."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval=portion.closedopen(0, 5), n_points=5),
            PiecewiseGridSegment(interval=portion.closed(5, 10), n_points=6),
        )
    )
    assert grid.n_points == 11


def test_piecewise_lin_spaced_grid_closed_boundary_is_exact():
    """Closed boundaries should produce exact endpoint values."""
    grid = PiecewiseLinSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[0, 5]", n_points=6),)
    )
    points = grid.to_jax()
    assert float(points[0]) == 0.0
    assert float(points[-1]) == 5.0


def test_piecewise_lin_spaced_grid_open_boundary_excludes_endpoint():
    """Open boundaries should not include the exact endpoint value."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0, 5)", n_points=5),
            PiecewiseGridSegment(interval="[5, 10]", n_points=6),
        )
    )
    points = grid.to_jax()
    # First segment should end just before 5
    assert float(points[4]) < 5.0
    # Second segment should start exactly at 5
    assert float(points[5]) == 5.0


def test_piecewise_lin_spaced_grid_no_representable_value_between_segments():
    """No representable float between adjacent open/closed boundaries."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0, 5)", n_points=5),
            PiecewiseGridSegment(interval="[5, 10]", n_points=6),
        )
    )
    points = grid.to_jax()

    # The last point of the first segment (open boundary)
    last_of_first = points[4]
    # The first point of the second segment (closed boundary)
    first_of_second = points[5]

    # nextafter should give us exactly the first point of the second segment
    assert jnp.nextafter(last_of_first, jnp.inf) == first_of_second


def test_piecewise_lin_spaced_grid_adjacent_closedopen_closedclosed():
    """[a, x) followed by [x, b] should be valid (adjacent)."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[1, 4)", n_points=3),
            PiecewiseGridSegment(interval="[4, 7]", n_points=4),
        )
    )
    assert grid.n_points == 7


def test_piecewise_lin_spaced_grid_adjacent_closed_openclosed():
    """[a, x] followed by (x, b] should be valid (adjacent)."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[1, 4]", n_points=4),
            PiecewiseGridSegment(interval="(4, 7]", n_points=3),
        )
    )
    assert grid.n_points == 7


def test_piecewise_lin_spaced_grid_invalid_gap():
    """[a, x) followed by (x, b] should be invalid (gap at x)."""
    with pytest.raises(GridInitializationError, match="not adjacent"):
        PiecewiseLinSpacedGrid(
            segments=(
                PiecewiseGridSegment(interval="[1, 4)", n_points=3),
                PiecewiseGridSegment(interval="(4, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_overlap():
    """[a, x] followed by [x, b] should be invalid (overlap at x)."""
    with pytest.raises(GridInitializationError, match="not adjacent"):
        PiecewiseLinSpacedGrid(
            segments=(
                PiecewiseGridSegment(interval="[1, 4]", n_points=3),
                PiecewiseGridSegment(interval="[4, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_numeric_gap():
    """Segments with numeric gap between them should be invalid."""
    with pytest.raises(GridInitializationError, match="gap"):
        PiecewiseLinSpacedGrid(
            segments=(
                PiecewiseGridSegment(interval="[1, 3]", n_points=3),
                PiecewiseGridSegment(interval="[5, 7]", n_points=3),
            )
        )


def test_piecewise_lin_spaced_grid_invalid_not_tuple():
    """Segments must be a tuple."""
    with pytest.raises(GridInitializationError, match="segments"):
        PiecewiseLinSpacedGrid(
            segments=[PiecewiseGridSegment(interval="[1, 4]", n_points=3)]  # ty: ignore[invalid-argument-type]
        )


def test_piecewise_lin_spaced_grid_invalid_not_segment():
    """Each element in segments must be a PiecewiseGridSegment object."""
    with pytest.raises(GridInitializationError, match="segments"):
        PiecewiseLinSpacedGrid(
            segments=({"interval": "[1, 4]", "n_points": 3},)  # ty: ignore[invalid-argument-type]
        )


def test_piecewise_lin_spaced_grid_invalid_n_points():
    """n_points must be >= 2."""
    with pytest.raises(GridInitializationError, match="n_points must be an int >= 2"):
        PiecewiseLinSpacedGrid(
            segments=(PiecewiseGridSegment(interval="[1, 4]", n_points=1),)
        )


def test_piecewise_lin_spaced_grid_invalid_interval_string():
    """Invalid interval string should raise error."""
    with pytest.raises(GridInitializationError, match="invalid"):
        PiecewiseLinSpacedGrid(
            segments=(PiecewiseGridSegment(interval="invalid", n_points=3),)
        )


def test_piecewise_lin_spaced_grid_three_segments():
    """Grid with three adjacent segments should work."""
    grid = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0, 3)", n_points=3),
            PiecewiseGridSegment(interval="[3, 7)", n_points=4),
            PiecewiseGridSegment(interval="[7, 10]", n_points=4),
        )
    )
    assert grid.n_points == 11
    points = grid.to_jax()
    assert float(points[0]) == 0.0
    assert float(points[-1]) == 10.0


def test_piecewise_log_spaced_grid_creation():
    """PiecewiseLogSpacedGrid can be created with valid positive intervals."""
    grid = PiecewiseLogSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0.1, 10)", n_points=5),
            PiecewiseGridSegment(interval="[10, 1000]", n_points=6),
        )
    )
    assert grid.n_points == 11


def test_piecewise_log_spaced_gridpoints_are_log_spaced():
    """Points within each segment should be logarithmically spaced."""
    grid = PiecewiseLogSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[1, 100]", n_points=3),)
    )
    points = grid.to_jax()
    assert np.allclose(points, [1.0, 10.0, 100.0])


def test_piecewise_log_spaced_grid_closed_boundary_is_exact():
    """Closed boundaries should produce exact endpoint values."""
    grid = PiecewiseLogSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[1, 1000]", n_points=4),)
    )
    points = grid.to_jax()
    assert float(points[0]) == pytest.approx(1.0)
    assert float(points[-1]) == pytest.approx(1000.0)


def test_piecewise_log_spaced_grid_open_boundary_excludes_endpoint():
    """Open boundaries should not include the exact endpoint value."""
    grid = PiecewiseLogSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[1, 10)", n_points=3),
            PiecewiseGridSegment(interval="[10, 100]", n_points=3),
        )
    )
    points = grid.to_jax()
    assert float(points[2]) < 10.0
    assert float(points[3]) == pytest.approx(10.0)


def test_piecewise_log_spaced_grid_invalid_negative_lower():
    """Lower bound must be positive for logspace."""
    with pytest.raises(GridInitializationError, match="must be positive"):
        PiecewiseLogSpacedGrid(
            segments=(PiecewiseGridSegment(interval="[-1, 10]", n_points=3),)
        )


def test_piecewise_log_spaced_grid_invalid_zero_lower():
    """Lower bound must be strictly positive for logspace."""
    with pytest.raises(GridInitializationError, match="must be positive"):
        PiecewiseLogSpacedGrid(
            segments=(PiecewiseGridSegment(interval="[0, 10]", n_points=3),)
        )


def test_piecewise_log_spaced_grid_invalid_gap():
    """Segments with gap should be invalid."""
    with pytest.raises(GridInitializationError, match="not adjacent"):
        PiecewiseLogSpacedGrid(
            segments=(
                PiecewiseGridSegment(interval="[1, 10)", n_points=3),
                PiecewiseGridSegment(interval="(10, 100]", n_points=3),
            )
        )


def test_piecewise_log_spaced_grid_three_segments():
    """Grid with three adjacent segments should work."""
    grid = PiecewiseLogSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0.01, 1)", n_points=3),
            PiecewiseGridSegment(interval="[1, 100)", n_points=3),
            PiecewiseGridSegment(interval="[100, 10000]", n_points=3),
        )
    )
    assert grid.n_points == 9
    points = grid.to_jax()
    assert float(points[0]) == pytest.approx(0.01)
    assert float(points[-1]) == pytest.approx(10000.0)


def test_piecewise_log_spaced_grid_coordinate_at_gridpoints():
    """Coordinates at exact grid points should match indices."""
    grid = PiecewiseLogSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[1, 100]", n_points=3),)
    )
    points = grid.to_jax()
    for i, p in enumerate(points):
        coord = float(grid.get_coordinate(p))
        assert coord == pytest.approx(i)


def test_piecewise_log_spaced_grid_coordinate_multi_segment():
    """Coordinates should work across multiple segments."""
    grid = PiecewiseLogSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[1, 10)", n_points=2),
            PiecewiseGridSegment(interval="[10, 100]", n_points=2),
        )
    )
    assert float(grid.get_coordinate(jnp.asarray(10.0))) == pytest.approx(2.0)
    assert float(grid.get_coordinate(jnp.asarray(100.0))) == pytest.approx(3.0)


def _create_boundary_test_grid(grid_cls, boundary_style: str):
    """Create a piecewise grid for boundary testing."""
    if grid_cls == PiecewiseLinSpacedGrid:
        if boundary_style == "closedopen_closed":
            return grid_cls(
                segments=(
                    PiecewiseGridSegment(interval="[0, 5)", n_points=5),
                    PiecewiseGridSegment(interval="[5, 10]", n_points=6),
                )
            )
        return grid_cls(
            segments=(
                PiecewiseGridSegment(interval="[0, 5]", n_points=6),
                PiecewiseGridSegment(interval="(5, 10]", n_points=5),
            )
        )
    # PiecewiseLogSpacedGrid
    if boundary_style == "closedopen_closed":
        return grid_cls(
            segments=(
                PiecewiseGridSegment(interval="[1, 10)", n_points=3),
                PiecewiseGridSegment(interval="[10, 100]", n_points=3),
            )
        )
    return grid_cls(
        segments=(
            PiecewiseGridSegment(interval="[1, 10]", n_points=3),
            PiecewiseGridSegment(interval="(10, 100]", n_points=3),
        )
    )


@pytest.mark.parametrize("grid_cls", [PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid])
@pytest.mark.parametrize("boundary_style", ["closedopen_closed", "closed_openclosed"])
def test_piecewise_boundary_conditions(grid_cls, boundary_style: str):
    """Test boundary conditions for piecewise grids.

    - closedopen_closed: [a, x) + [x, b] - value at x goes to segment 1
    - closed_openclosed: [a, x] + (x, b] - value at x goes to segment 0
    """
    grid = _create_boundary_test_grid(grid_cls, boundary_style)

    # Determine boundary value and expected coordinates
    is_lin = grid_cls == PiecewiseLinSpacedGrid
    boundary = 5.0 if is_lin else 10.0
    below_boundary = 4.99 if is_lin else 9.9
    above_boundary = 5.01 if is_lin else 10.1

    # Coordinate at boundary depends on boundary style
    if boundary_style == "closedopen_closed":
        # [a, x) + [x, b]: boundary belongs to segment 1
        expected_coord_at = 5.0 if is_lin else 3.0
    else:
        # [a, x] + (x, b]: boundary belongs to segment 0
        expected_coord_at = 5.0 if is_lin else 2.0

    # Test value just below boundary -> segment 0
    coord_below = float(grid.get_coordinate(jnp.asarray(below_boundary)))
    assert coord_below < expected_coord_at

    # Test value exactly at boundary
    coord_at = float(grid.get_coordinate(jnp.asarray(boundary)))
    assert coord_at == pytest.approx(expected_coord_at)

    # Test value just above boundary -> segment 1
    coord_above = float(grid.get_coordinate(jnp.asarray(above_boundary)))
    assert coord_above > expected_coord_at


def test_piecewise_single_segment():
    """Test piecewise grid with a single segment works correctly."""
    grid = PiecewiseLinSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[0, 10]", n_points=11),)
    )
    assert float(grid.get_coordinate(jnp.asarray(0.0))) == pytest.approx(0.0)
    assert float(grid.get_coordinate(jnp.asarray(5.0))) == pytest.approx(5.0)
    assert float(grid.get_coordinate(jnp.asarray(10.0))) == pytest.approx(10.0)


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
        segments=(PiecewiseGridSegment(interval="[0, 10]", n_points=11),)
    )
    values = jnp.array([0.0, 5.0, 10.0])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.0, 5.0, 10.0])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


def test_piecewise_log_spaced_grid_get_coordinate_with_array():
    grid = PiecewiseLogSpacedGrid(
        segments=(PiecewiseGridSegment(interval="[1, 100]", n_points=3),)
    )
    points = grid.to_jax()
    values = jnp.array([float(points[0]), float(points[1]), float(points[2])])
    coords = grid.get_coordinate(values)
    expected = jnp.array([0.0, 1.0, 2.0])
    aaae(coords, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "make_grid",
    [
        pytest.param(
            lambda **kw: LinSpacedGrid(start=1, stop=10, n_points=4, **kw),
            id="LinSpacedGrid",
        ),
        pytest.param(
            lambda **kw: LogSpacedGrid(start=1, stop=10, n_points=4, **kw),
            id="LogSpacedGrid",
        ),
        pytest.param(
            lambda **kw: IrregSpacedGrid(points=[1.0, 2.0, 3.0, 4.0], **kw),
            id="IrregSpacedGrid",
        ),
        pytest.param(
            lambda **kw: DiscreteGrid(
                _make_dc("_BS", ("a", jnp.int32(0)), ("b", jnp.int32(1))), **kw
            ),
            id="DiscreteGrid",
        ),
    ],
)
def test_grid_rejects_batch_size_combined_with_distributed(make_grid):
    """`batch_size > 0` and `distributed=True` on one axis is rejected at init.

    Each Python-level batch triggers its own per-period cross-device
    collective in the sharded solve, so the combination multiplies the
    sync count by `ceil(n_per_device / batch_size)` and inverts the
    compute/communication ratio. Construction-time rejection prevents
    the foot-gun.
    """
    with pytest.raises(GridInitializationError, match="distributed=True"):
        make_grid(batch_size=1, distributed=True)


@pytest.mark.parametrize(
    "make_grid",
    [
        pytest.param(
            lambda: LinSpacedGrid(
                start=1, stop=10, n_points=4, batch_size=0, distributed=True
            ),
            id="LinSpacedGrid",
        ),
        pytest.param(
            lambda: DiscreteGrid(
                _make_dc("_OK", ("a", jnp.int32(0)), ("b", jnp.int32(1))),
                batch_size=0,
                distributed=True,
            ),
            id="DiscreteGrid",
        ),
    ],
)
def test_grid_accepts_batch_size_zero_with_distributed(make_grid):
    """`batch_size=0` with `distributed=True` is the canonical sharded setting."""
    grid = make_grid()
    assert grid.distributed is True
    assert grid.batch_size == 0
