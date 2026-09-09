"""A planner tile changes the working window, preserving the full state product."""

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.utils import dispatchers
from lcm.typing import BoolND, FloatND
from tests.conftest import assert_agrees_to_ulp


def _evaluate_grouped_cell(
    *, first: FloatND, second: FloatND, last: FloatND
) -> FloatND:
    """Expose independent nonlinear work on a coordinate prefix and final axis."""
    return jnp.sqrt(first + second) + jnp.exp(last)


def _primitive_input_shapes(*, graph: Any, name: str) -> list[tuple[int, ...]]:
    """Collect operand shapes through nested mapping and reduction bodies."""
    if not hasattr(graph, "eqns"):
        if hasattr(graph, "jaxpr"):
            return _primitive_input_shapes(graph=graph.jaxpr, name=name)
        return []
    shapes = []
    for equation in graph.eqns:
        if equation.primitive.name == name:
            shapes.append(equation.invars[0].aval.shape)
        for parameter in equation.params.values():
            children = (
                parameter if isinstance(parameter, tuple | list) else (parameter,)
            )
            for child in children:
                shapes.extend(_primitive_input_shapes(graph=child, name=name))
    return shapes


@pytest.mark.parametrize("width", [8, 30, 40])
def test_tiled_product_limits_coordinate_only_nonlinear_work(*, width: int) -> None:
    """The final coordinate's nonlinear work fits its grid and the planned window."""
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_grouped_cell,
                variables=("first", "second", "last"),
                width_keyword="cell_width",
            ),
        ),
        cell_width=width,
    )
    traced = jax.make_jaxpr(mapped)(
        first=jnp.asarray([1.0, 2.0]),
        second=jnp.asarray([1.0, 2.0, 3.0]),
        last=jnp.asarray([0.0, 0.125, 0.25, 0.375, 0.5]),
    )
    shapes = _primitive_input_shapes(graph=traced, name="exp")
    assert max((np.prod(shape) for shape in shapes), default=width + 1) <= min(width, 5)


def test_full_product_bounds_prefix_nonlinear_rank() -> None:
    """Coordinate-prefix work adds at most one batch axis to the scalar body."""
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_grouped_cell,
                variables=("first", "second", "last"),
                width_keyword="cell_width",
            ),
        ),
        cell_width=30,
    )
    traced = jax.make_jaxpr(mapped)(
        first=jnp.asarray([1.0, 2.0]),
        second=jnp.asarray([1.0, 2.0, 3.0]),
        last=jnp.asarray([0.0, 0.125, 0.25, 0.375, 0.5]),
    )
    shapes = _primitive_input_shapes(graph=traced, name="sqrt")
    assert max((len(shape) for shape in shapes), default=2) <= 1


@pytest.mark.parametrize("width", [1, 4, 8, 11, 30, 40])
def test_grouped_product_respects_combined_window(*, width: int) -> None:
    """The two independent nonlinear operand windows fit the admitted cell width."""
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_grouped_cell,
                variables=("first", "second", "last"),
                width_keyword="cell_width",
            ),
        ),
        cell_width=width,
    )
    traced = jax.make_jaxpr(mapped)(
        first=jnp.asarray([1.0, 2.0]),
        second=jnp.asarray([1.0, 2.0, 3.0]),
        last=jnp.asarray([0.0, 0.125, 0.25, 0.375, 0.5]),
    )
    prefix = _primitive_input_shapes(graph=traced, name="sqrt")
    final = _primitive_input_shapes(graph=traced, name="exp")
    prefix_window = max((np.prod(shape) for shape in prefix), default=width + 1)
    final_window = max((np.prod(shape) for shape in final), default=width + 1)
    assert prefix_window * final_window <= width


@pytest.mark.parametrize("width", [1, 4, 8, 30, 40])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "variables", [("first", "second", "last"), ("last", "second", "first")]
)
def test_grouped_product_matches_scalar_enumeration(
    *, width: int, compiled: bool, variables: tuple[str, ...]
) -> None:
    """Every nonlinear output retains its declared Cartesian coordinate order."""
    arguments = {
        "first": jnp.asarray([1.0, 2.0]),
        "second": jnp.asarray([1.0, 2.0, 3.0]),
        "last": jnp.asarray([0.0, 0.125, 0.25, 0.375, 0.5]),
    }
    expected = np.asarray(
        [
            [
                [
                    np.sqrt(first + second) + np.exp(last)
                    for last in [0.0, 0.125, 0.25, 0.375, 0.5]
                ]
                for second in [1.0, 2.0, 3.0]
            ]
            for first in [1.0, 2.0]
        ],
        dtype=arguments["first"].dtype,
    )
    if variables[0] == "last":
        expected = expected.transpose((2, 1, 0))
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_grouped_cell,
                variables=variables,
                width_keyword="cell_width",
            ),
        ),
        cell_width=width,
    )
    run = jax.jit(mapped) if compiled else mapped
    assert_agrees_to_ulp(got=run(**arguments), expected=expected, n_ulp=4)


def _evaluate_cell(
    *, first: FloatND, second: FloatND, offset: FloatND
) -> tuple[FloatND, BoolND]:
    """Distinguish every coordinate, a trailing value axis, and a Boolean leaf."""
    return (
        jnp.stack((10.0 * first + second + offset, first - 2.0 * second)),
        first > second,
    )


def _build_mapper(
    *, variables: tuple[str, ...], untiled_variables: tuple[str, ...] = ()
) -> Callable[..., Any]:
    """Obtain the new mapping boundary without hiding a missing implementation."""
    build = cast("Callable[..., Callable[..., Any]]", dispatchers.tiled_productmap)
    return build(
        func=_evaluate_cell,
        variables=variables,
        width_keyword="cell_width",
        untiled_variables=untiled_variables,
    )


@pytest.mark.parametrize("width", [1, 4, 6, 9])
@pytest.mark.parametrize("variables", [("first", "second"), ("second", "first")])
@pytest.mark.parametrize("compiled", [False, True])
def test_state_tiles_preserve_coordinate_order_and_output_tree(
    *, width: int, variables: tuple[str, ...], compiled: bool
) -> None:
    """Dividing, remainder, full and clamped tiles preserve every coordinate once."""
    mapped = functools.partial(_build_mapper(variables=variables), cell_width=width)
    arguments = {
        "first": jnp.asarray([1.0, 3.0]),
        "second": jnp.asarray([2.0, 4.0, 7.0]),
        "offset": jnp.asarray(5.0),
    }
    expected_values = np.asarray(
        [
            [
                [10.0 * first + second + 5.0, first - 2.0 * second]
                for second in [2.0, 4.0, 7.0]
            ]
            for first in [1.0, 3.0]
        ]
    )
    if variables[0] == "second":
        expected_values = expected_values.swapaxes(0, 1)
    run = jax.jit(mapped) if compiled else mapped
    values, _flags = run(**arguments)
    assert_agrees_to_ulp(got=values, expected=expected_values, n_ulp=4)


@pytest.mark.parametrize("width", [1, 4, 6, 9])
@pytest.mark.parametrize("variables", [("first", "second"), ("second", "first")])
@pytest.mark.parametrize("compiled", [False, True])
def test_state_tiles_preserve_exact_flags(
    *, width: int, variables: tuple[str, ...], compiled: bool
) -> None:
    """Boolean leaves keep their coordinate axes independently of value channels."""
    mapped = functools.partial(_build_mapper(variables=variables), cell_width=width)
    run = jax.jit(mapped) if compiled else mapped
    _values, flags = run(
        first=jnp.asarray([1.0, 3.0]),
        second=jnp.asarray([2.0, 4.0, 7.0]),
        offset=jnp.asarray(5.0),
    )
    expected = np.asarray([[False, False, False], [True, False, False]])
    assert_array_equal(flags, expected if variables[0] == "first" else expected.T)


def test_state_tiles_preserve_boolean_dtype() -> None:
    """Reconstruction retains a decision leaf's Boolean type."""
    _values, flags = _build_mapper(variables=("first", "second"))(
        first=jnp.asarray([1.0, 3.0]),
        second=jnp.asarray([2.0, 4.0, 7.0]),
        offset=jnp.asarray(5.0),
        cell_width=4,
    )
    assert flags.dtype == jnp.bool_


def test_an_empty_product_adds_no_output_axis() -> None:
    """A scalar cell function remains scalar over an empty product of variables."""
    values, _flags = _build_mapper(variables=())(
        first=jnp.asarray(2.0), second=jnp.asarray(3.0), offset=jnp.asarray(5.0)
    )
    assert_array_equal(values, np.asarray([28.0, -4.0]))


def test_singleton_coordinates_retain_their_axes() -> None:
    """A one-cell product still publishes one axis per declared coordinate."""
    values, _flags = _build_mapper(variables=("first", "second"))(
        first=jnp.asarray([2.0]), second=jnp.asarray([3.0]), offset=jnp.asarray(5.0)
    )
    assert_array_equal(values, np.asarray([[[28.0, -4.0]]]))


def test_flat_cell_indices_cannot_overflow_int32() -> None:
    """Oversized products fail before allocating or indexing the flat cell vector."""
    mapped = functools.partial(
        _build_mapper(variables=("first", "second")), cell_width=4
    )
    with pytest.raises(ValueError, match="int32 index range"):
        jax.eval_shape(
            mapped,
            first=jax.ShapeDtypeStruct((50_000,), jnp.float32),
            second=jax.ShapeDtypeStruct((50_000,), jnp.float32),
            offset=jax.ShapeDtypeStruct((), jnp.float32),
        )


@pytest.mark.parametrize("width", [1, 4, 6])
def test_state_width_changes_the_actual_compiled_loop(*, width: int) -> None:
    """A partial Cartesian window scans the prefix; a full window vectorizes it."""
    mapped = functools.partial(
        _build_mapper(variables=("first", "second")), cell_width=width
    )
    traced = jax.make_jaxpr(mapped)(
        first=jnp.asarray([1.0, 3.0]),
        second=jnp.asarray([2.0, 4.0, 7.0]),
        offset=jnp.asarray(5.0),
    )
    lengths = [
        equation.params["length"]
        for equation in traced.jaxpr.eqns
        if equation.primitive.name == "scan"
    ]
    assert lengths == ([2] if width < 6 else [])


@pytest.mark.parametrize("untiled", [("first",), ("second",)])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_untiled_outer_axes_restore_canonical_product_order(
    *, untiled: tuple[str, ...], width: int
) -> None:
    """An outer axis can follow a tiled axis in the published coordinate order."""
    mapped = _build_mapper(
        variables=("first", "second"),
        untiled_variables=untiled,
    )
    values, _flags = jax.jit(functools.partial(mapped, cell_width=width))(
        first=jnp.asarray([1.0, 3.0]),
        second=jnp.asarray([2.0, 4.0, 7.0]),
        offset=jnp.asarray(5.0),
    )
    assert_agrees_to_ulp(
        got=values,
        expected=np.asarray(
            [
                [[17.0, -3.0], [19.0, -7.0], [22.0, -13.0]],
                [[37.0, -1.0], [39.0, -5.0], [42.0, -11.0]],
            ]
        ),
        n_ulp=4,
    )


@pytest.mark.parametrize("untiled", [("first",), ("second",)])
def test_untiled_outer_axes_restore_exact_flag_order(
    *, untiled: tuple[str, ...]
) -> None:
    mapped = _build_mapper(
        variables=("first", "second"),
        untiled_variables=untiled,
    )
    _values, flags = jax.jit(functools.partial(mapped, cell_width=2))(
        first=jnp.asarray([1.0, 3.0]),
        second=jnp.asarray([2.0, 4.0, 7.0]),
        offset=jnp.asarray(5.0),
    )
    assert_array_equal(flags, np.asarray([[False, False, False], [True, False, False]]))
