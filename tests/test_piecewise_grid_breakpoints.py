"""Piecewise grids expose breakpoint ownership directly."""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import lcm.grids as public_grids
from _lcm.dtypes import canonical_float_dtype
from lcm import (
    GridBreakpoint,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from lcm.exceptions import GridInitializationError


def test_piecewise_grid_has_only_breakpoint_interface() -> None:
    """Piecewise grids accept endpoints, breakpoints, and per-segment counts."""
    parameters = inspect.signature(PiecewiseLinSpacedGrid).parameters

    assert {
        "start",
        "stop",
        "breakpoints",
        "points_per_segment",
    } <= parameters.keys()
    assert "segments" not in parameters


def test_piecewise_grid_segment_is_not_public() -> None:
    """The interval-segment declaration is absent from public namespaces."""
    assert not hasattr(lcm, "PiecewiseGridSegment")
    assert not hasattr(public_grids, "PiecewiseGridSegment")


def test_piecewise_grid_rejects_the_segment_interface() -> None:
    """The removed segment keyword is an ordinary signature error."""
    with pytest.raises(TypeError):
        PiecewiseLinSpacedGrid(segments=())  # ty: ignore[missing-argument, unknown-argument]


def test_right_owned_breakpoint_contributes_to_the_right_segment() -> None:
    """A right-owned breakpoint appears once at the next segment's first node."""
    grid = PiecewiseLinSpacedGrid(
        start=1.0,
        stop=10.0,
        breakpoints=(GridBreakpoint(value=4.0, owner="right"),),
        points_per_segment=(3, 6),
    )

    points = grid.to_jax()

    assert int(grid.n_points) == 9
    assert float(points[0]) == 1.0
    assert float(points[-1]) == 10.0
    assert float(points[2]) < 4.0
    assert float(points[3]) == 4.0
    assert jnp.nextafter(points[2], jnp.inf) == points[3]


def test_left_owned_breakpoint_contributes_to_the_left_segment() -> None:
    """A left-owned breakpoint appears once at the preceding segment's last node."""
    grid = PiecewiseLinSpacedGrid(
        start=1.0,
        stop=10.0,
        breakpoints=(GridBreakpoint(value=4.0, owner="left"),),
        points_per_segment=(3, 6),
    )

    points = grid.to_jax()

    assert float(points[2]) == 4.0
    assert float(points[3]) > 4.0
    assert jnp.nextafter(points[2], jnp.inf) == points[3]


def test_multiple_breakpoints_contribute_the_declared_point_counts() -> None:
    """Each nominal segment contributes its full declared output count."""
    grid = PiecewiseLinSpacedGrid(
        start=0.0,
        stop=100.0,
        breakpoints=(
            GridBreakpoint(value=10.0, owner="right"),
            GridBreakpoint(value=40.0, owner="left"),
        ),
        points_per_segment=(2, 3, 4),
    )

    points = grid.to_jax()

    assert int(grid.n_points) == 9
    assert np.count_nonzero(np.asarray(points) == 10.0) == 1
    assert np.count_nonzero(np.asarray(points) == 40.0) == 1


def test_coordinates_recover_every_node_across_owned_breakpoints() -> None:
    """Generalized coordinates remain contiguous across mixed ownership."""
    grid = PiecewiseLinSpacedGrid(
        start=0.0,
        stop=100.0,
        breakpoints=(
            GridBreakpoint(value=10.0, owner="right"),
            GridBreakpoint(value=40.0, owner="left"),
        ),
        points_per_segment=(4, 5, 3),
    )
    points = grid.to_jax()

    np.testing.assert_allclose(
        grid.get_coordinate(points),
        np.arange(points.shape[0]),
    )


def test_empty_breakpoints_form_one_complete_segment() -> None:
    """An empty breakpoint tuple is a valid one-segment declaration."""
    grid = PiecewiseLinSpacedGrid(
        start=-1.0,
        stop=1.0,
        breakpoints=(),
        points_per_segment=(5,),
    )

    np.testing.assert_allclose(grid.to_jax(), np.linspace(-1.0, 1.0, 5))


def test_piecewise_log_grid_uses_log_spacing_with_the_same_ownership() -> None:
    """Log grids change only within-segment spacing."""
    grid = PiecewiseLogSpacedGrid(
        start=0.1,
        stop=1_000.0,
        breakpoints=(GridBreakpoint(value=10.0, owner="right"),),
        points_per_segment=(3, 3),
    )

    points = grid.to_jax()

    assert float(points[3]) == pytest.approx(10.0)
    np.testing.assert_allclose(points[3:], [10.0, 100.0, 1_000.0])


@pytest.mark.parametrize(
    ("grid_cls", "start", "breakpoint_value", "stop"),
    [
        (PiecewiseLinSpacedGrid, 0.0, 5.0, 10.0),
        (PiecewiseLogSpacedGrid, 1.0, 10.0, 100.0),
    ],
)
@pytest.mark.parametrize(
    ("owner", "expected_coordinate"), [("right", 3.0), ("left", 2.0)]
)
def test_breakpoint_coordinate_follows_ownership(
    *,
    grid_cls: type[PiecewiseLinSpacedGrid | PiecewiseLogSpacedGrid],
    start: float,
    breakpoint_value: float,
    stop: float,
    owner: str,
    expected_coordinate: float,
) -> None:
    """Coordinate lookup assigns equality to the breakpoint's owner."""
    grid = grid_cls(
        start=start,
        stop=stop,
        breakpoints=(GridBreakpoint(value=breakpoint_value, owner=owner),),  # ty: ignore[invalid-argument-type]
        points_per_segment=(3, 3),
    )

    coordinate = grid.get_coordinate(jnp.asarray(breakpoint_value))

    assert float(coordinate) == pytest.approx(expected_coordinate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": float("nan"), "stop": 2.0, "breakpoints": ()}, "finite"),
        ({"start": 0.0, "stop": float("inf"), "breakpoints": ()}, "finite"),
        ({"start": 2.0, "stop": 1.0, "breakpoints": ()}, "start < stop"),
        ({"start": 0.0, "stop": 2.0, "breakpoints": (0.0,)}, "strictly between"),
        ({"start": 0.0, "stop": 2.0, "breakpoints": (2.0,)}, "strictly between"),
        ({"start": 0.0, "stop": 3.0, "breakpoints": (2.0, 1.0)}, "increasing"),
        ({"start": 0.0, "stop": 3.0, "breakpoints": (1.0, 1.0)}, "increasing"),
    ],
)
def test_piecewise_grid_rejects_invalid_bounds(
    *, kwargs: dict[str, object], message: str
) -> None:
    """Endpoints and breakpoints define a finite strictly ordered domain."""
    breakpoints = tuple(
        GridBreakpoint(value=value)
        for value in kwargs["breakpoints"]  # ty: ignore[not-iterable]
    )
    with pytest.raises(GridInitializationError, match=message):
        PiecewiseLinSpacedGrid(
            start=kwargs["start"],  # ty: ignore[invalid-argument-type]
            stop=kwargs["stop"],  # ty: ignore[invalid-argument-type]
            breakpoints=breakpoints,
            points_per_segment=(2,) * (len(breakpoints) + 1),
        )


def test_piecewise_grid_rejects_bounds_collapsed_by_canonical_precision(
    x64_disabled,
) -> None:
    """Distinct inputs must remain distinct after canonical float conversion."""
    assert x64_disabled is None
    with pytest.raises(GridInitializationError, match="strictly between"):
        PiecewiseLinSpacedGrid(
            start=1.0,
            stop=2.0,
            breakpoints=(GridBreakpoint(value=1.0 + 2**-25),),
            points_per_segment=(2, 2),
        )


@pytest.mark.parametrize("points_per_segment", [(2,), (2, 2, 2), (1, 2), (2.5, 2)])
def test_piecewise_grid_rejects_invalid_point_counts(
    points_per_segment: tuple[object, ...],
) -> None:
    """There is one integer count of at least two per nominal segment."""
    with pytest.raises(GridInitializationError):
        PiecewiseLinSpacedGrid(
            start=0.0,
            stop=2.0,
            breakpoints=(GridBreakpoint(value=1.0),),
            points_per_segment=points_per_segment,  # ty: ignore[invalid-argument-type]
        )


def test_piecewise_grid_rejects_an_unknown_owner() -> None:
    """Breakpoint ownership is exactly left or right."""
    with pytest.raises(GridInitializationError, match="owner"):
        PiecewiseLinSpacedGrid(
            start=0.0,
            stop=2.0,
            breakpoints=(GridBreakpoint(value=1.0, owner="middle"),),  # ty: ignore[invalid-argument-type]
            points_per_segment=(2, 2),
        )


def test_piecewise_grid_rejects_a_collapsed_effective_segment() -> None:
    """An open boundary must leave a representable interval on each side."""
    start = jnp.asarray(1.0, dtype=canonical_float_dtype())
    adjacent = jnp.nextafter(start, jnp.inf)

    with pytest.raises(GridInitializationError, match="representable"):
        PiecewiseLinSpacedGrid(
            start=start,
            stop=2.0,
            breakpoints=(GridBreakpoint(value=float(adjacent), owner="right"),),
            points_per_segment=(2, 2),
        )


@pytest.mark.parametrize("start", [-1.0, 0.0])
def test_piecewise_log_grid_requires_a_positive_domain(start: float) -> None:
    """Every log-grid boundary is strictly positive."""
    with pytest.raises(GridInitializationError, match="positive"):
        PiecewiseLogSpacedGrid(
            start=start,
            stop=10.0,
            breakpoints=(),
            points_per_segment=(3,),
        )


def test_piecewise_grid_uses_the_canonical_float_dtype() -> None:
    """Endpoint, breakpoint, and output arithmetic share pylcm's float dtype."""
    grid = PiecewiseLinSpacedGrid(
        start=0,
        stop=2,
        breakpoints=(GridBreakpoint(value=1),),
        points_per_segment=(2, 2),
    )

    assert grid.start.dtype == canonical_float_dtype()
    assert grid.stop.dtype == canonical_float_dtype()
    assert grid.to_jax().dtype == canonical_float_dtype()
