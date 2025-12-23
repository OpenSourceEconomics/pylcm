from __future__ import annotations

import jax.numpy as jnp

from lcm.grids import DiscreteGrid
from lcm.input_processing.create_params_template import (
    _create_function_params,
    create_params_template,
)
from lcm.input_processing.regime_processing import get_grids
from tests.regime_mock import RegimeMock


def test_create_params_without_shocks(binary_category_class):
    regime = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "b": DiscreteGrid(binary_category_class),
        },
        n_periods=None,
        utility=lambda a, b, c: None,  # noqa: ARG005
        transitions={
            "next_b": lambda b: b,
        },
    )
    got = create_params_template(
        regime,  # ty: ignore[invalid-argument-type]
        grids={regime.name: get_grids(regime)},  # ty: ignore[invalid-argument-type]
        n_periods=3,
    )
    assert got == {
        "discount_factor": jnp.nan,
        "utility": {"c": jnp.nan},
        "next_b": {},
    }


def test_create_function_params():
    regime = RegimeMock(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        utility=lambda a, b, c: None,  # noqa: ARG005
    )
    got = _create_function_params(regime)  # ty: ignore[invalid-argument-type]
    assert got == {"utility": {"c": jnp.nan}}


def test_n_periods_and_last_period_are_special_variables(binary_category_class):
    """n_periods and last_period should be excluded from params like period is."""
    regime = RegimeMock(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "b": DiscreteGrid(binary_category_class),
        },
        # n_periods and last_period in signature should be treated as special variables
        utility=lambda a, b, n_periods, last_period: None,  # noqa: ARG005
        transitions={
            "next_b": lambda b: b,
        },
    )
    got = create_params_template(
        regime,  # type: ignore[arg-type]
        grids={regime.name: get_grids(regime)},  # type: ignore[arg-type]
        n_periods=3,
    )
    # n_periods and last_period should NOT appear in params template
    assert got == {
        "discount_factor": jnp.nan,
        "utility": {},  # Empty because n_periods and last_period are special vars
        "next_b": {},
    }
