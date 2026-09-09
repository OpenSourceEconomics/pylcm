"""A cell window retains flat batching until two final-coordinate groups fit."""

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
from tests.test_tiled_productmap import (
    _evaluate_grouped_cell,
    _primitive_input_shapes,
)


def test_width_64_retains_flat_batches_for_final_extent_50() -> None:
    """A 300-cell product uses four 64-cell batches and one 44-cell remainder."""
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_grouped_cell,
                variables=("first", "second", "last"),
                width_keyword="cell_width",
            ),
        ),
        cell_width=64,
    )
    traced = jax.make_jaxpr(mapped)(
        first=jnp.asarray([1.0, 2.0]),
        second=jnp.asarray([1.0, 2.0, 3.0]),
        last=jnp.arange(50, dtype=jnp.float32) / 100.0,
    )
    observed = {
        "scan_lengths": tuple(
            equation.params["length"]
            for equation in traced.jaxpr.eqns
            if equation.primitive.name == "scan"
        ),
        "nonlinear_windows": tuple(
            sorted(_primitive_input_shapes(graph=traced, name="exp"))
        ),
    }
    assert observed == {
        "scan_lengths": (4,),
        "nonlinear_windows": ((44,), (64,)),
    }


@pytest.mark.parametrize(
    ("final_extent", "width", "scan_length", "nonlinear_windows"),
    [
        (1, 1, 6, ((1,),)),
        (1, 2, 3, ((1,),)),
        (5, 4, 7, ((2,), (4,))),
        (5, 5, 6, ((5,),)),
        (5, 9, 3, ((3,), (9,))),
        (5, 10, 3, ((5,),)),
        (50, 49, 6, ((6,), (49,))),
        (50, 50, 6, ((50,),)),
        (50, 99, 3, ((3,), (99,))),
        (50, 100, 3, ((50,),)),
    ],
)
def test_two_final_coordinate_groups_are_the_batching_threshold(
    *,
    final_extent: int,
    width: int,
    scan_length: int,
    nonlinear_windows: tuple[tuple[int, ...], ...],
) -> None:
    """Known flat and grouped loop windows cover each side of the threshold."""
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
        last=jnp.arange(final_extent, dtype=jnp.float32) / 100.0,
    )
    assert {
        "scan_lengths": tuple(
            equation.params["length"]
            for equation in traced.jaxpr.eqns
            if equation.primitive.name == "scan"
        ),
        "nonlinear_windows": tuple(
            sorted(_primitive_input_shapes(graph=traced, name="exp"))
        ),
    } == {"scan_lengths": (scan_length,), "nonlinear_windows": nonlinear_windows}


def _evaluate_value_and_flag(
    *, first: FloatND, second: FloatND, last: FloatND
) -> tuple[FloatND, BoolND]:
    """Evaluate a smooth value and a decision boundary on the same scalar cell."""
    return jnp.sqrt(first + second) + jnp.exp(last), first > second + last


@pytest.mark.parametrize(
    ("variables", "untiled", "width"),
    [
        (variables, untiled, width)
        for variables, untiled, widths in (
            (("first", "second", "last"), (), (4, 5, 9, 10)),
            (("last", "first", "second"), (), (2, 3, 5, 6)),
            (("first", "second", "last"), ("first",), (4, 5, 9, 10)),
            (("first", "second", "last"), ("last",), (2, 3, 5, 6)),
        )
        for width in widths
    ],
)
def test_threshold_routes_preserve_scalar_values_and_exact_flags(
    *, variables: tuple[str, ...], untiled: tuple[str, ...], width: int
) -> None:
    """Reordered and outer coordinates keep scalar economics through each route."""
    scalar_coordinates = {
        "first": (1.0, 2.0),
        "second": (1.0, 2.0, 3.0),
        "last": (0.0, 0.125, 0.25, 0.375, 0.5),
    }
    arguments = {
        name: jnp.asarray(coordinate) for name, coordinate in scalar_coordinates.items()
    }
    expected_values = np.asarray(
        [
            [
                [
                    np.sqrt(first + second) + np.exp(last)
                    for last in scalar_coordinates["last"]
                ]
                for second in scalar_coordinates["second"]
            ]
            for first in scalar_coordinates["first"]
        ],
        dtype=arguments["first"].dtype,
    )
    expected_flags = np.asarray(
        [
            [
                [first > second + last for last in scalar_coordinates["last"]]
                for second in scalar_coordinates["second"]
            ]
            for first in scalar_coordinates["first"]
        ]
    )
    canonical = ("first", "second", "last")
    axes = tuple(canonical.index(variable) for variable in variables)
    mapped = functools.partial(
        cast(
            "Callable[..., Any]",
            dispatchers.tiled_productmap(
                func=_evaluate_value_and_flag,
                variables=variables,
                width_keyword="cell_width",
                untiled_variables=untiled,
            ),
        ),
        cell_width=width,
    )
    values, flags = jax.jit(mapped)(**arguments)
    assert_agrees_to_ulp(got=values, expected=expected_values.transpose(axes), n_ulp=4)
    assert_array_equal(flags, expected_flags.transpose(axes))
