from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
import pytest

from lcm.grids import DiscreteGrid
from lcm.input_processing.create_params_template import (
    _create_function_params,
    _create_stochastic_transition_params,
    create_params_template,
)
from tests.model_mock import ModelMock


def test_create_params_without_shocks(binary_category_class):
    model = ModelMock(
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
    got = create_params_template(model)  # type: ignore[arg-type]
    assert got == {"beta": jnp.nan, "utility": {"c": jnp.nan}, "next_b": {}}


def test_create_function_params():
    model = ModelMock(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        utility=lambda a, b, c: None,  # noqa: ARG005
    )
    got = _create_function_params(model)  # type: ignore[arg-type]
    assert got == {"utility": {"c": jnp.nan}}


def test_create_shock_params():
    def next_a(a, _period):
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": True},
        index=["a"],
    )

    model = ModelMock(
        n_periods=3,
        utility=lambda a: None,  # noqa: ARG005
        transitions={"next_a": next_a},
    )

    got = _create_stochastic_transition_params(
        model=model,  # type: ignore[arg-type]
        variable_info=variable_info,
        grids={"a": jnp.array([1, 2])},
    )
    jnp.array_equal(got["a"], jnp.full((2, 3, 2), jnp.nan), equal_nan=True)


def test_create_shock_params_invalid_variable():
    def next_a(a):
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": False},
        index=["a"],
    )

    model = ModelMock(
        transitions={"next_a": next_a},
    )

    with pytest.raises(ValueError, match="The following variables are stochastic, but"):
        _create_stochastic_transition_params(
            model=model,  # type: ignore[arg-type]
            variable_info=variable_info,
            grids={"a": jnp.array([1, 2])},
        )


def test_create_shock_params_invalid_dependency():
    def next_a(a, b, _period):
        pass

    variable_info = pd.DataFrame(
        {
            "is_stochastic": [True, False],
            "is_state": [True, False],
            "is_discrete": [True, False],
        },
        index=["a", "b"],
    )

    model = ModelMock(
        transitions={"next_a": next_a},
    )

    with pytest.raises(ValueError, match="Stochastic transition functions can only"):
        _create_stochastic_transition_params(
            model=model,  # type: ignore[arg-type]
            variable_info=variable_info,
            grids={"a": jnp.array([1, 2])},
        )
