from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
import pytest

import lcm
from lcm.grids import DiscreteGrid
from lcm.input_processing.create_params_template import (
    _create_function_params,
    _create_stochastic_transition_params,
    create_params_template,
)
from lcm.input_processing.regime_processing import (
    convert_flat_to_nested_transitions,
    get_grids,
)
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
    nested_transitions = convert_flat_to_nested_transitions(
        regime.transitions, current_regime_name=regime.name
    )
    got = create_params_template(
        regime,  # type: ignore[arg-type]
        nested_transitions=nested_transitions,
        grids={regime.name: get_grids(regime)},  # type: ignore[arg-type]
        n_periods=3,
    )
    # With flat transitions, param keys are flat (no regime prefix)
    assert got == {
        "beta": jnp.nan,
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
    nested_transitions = convert_flat_to_nested_transitions(
        regime.transitions, current_regime_name=regime.name
    )
    got = _create_function_params(regime, nested_transitions)  # type: ignore[arg-type]
    assert got == {"utility": {"c": jnp.nan}}


def test_create_shock_params():
    @lcm.mark.stochastic
    def next_a(a, period, a_transition):
        return a_transition[a, period]

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_continuous": False},
        index=["a"],
    )

    # _create_stochastic_transition_params expects nested format internally
    regime = RegimeMock(
        utility=lambda a: None,  # noqa: ARG005
        transitions={"mock": {"next_a": next_a}},  # type: ignore[dict-item]
    )

    got = _create_stochastic_transition_params(
        regime=regime,  # type: ignore[arg-type]
        variable_info=variable_info,
        n_periods=3,
        grids={"mock": {"a": jnp.array([1, 2])}},
    )
    jnp.array_equal(got["mock"]["next_a"], jnp.full((2, 3, 2), jnp.nan), equal_nan=True)


def test_create_shock_params_invalid_variable():
    @lcm.mark.stochastic
    def next_a(a, a_transition):
        return a_transition[a]

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_continuous": True},
        index=["a"],
    )

    # _create_stochastic_transition_params expects nested format internally
    regime = RegimeMock(
        transitions={"mock": {"next_a": next_a}},  # type: ignore[dict-item]
    )

    with pytest.raises(
        ValueError, match="Stochastic transition functions cannot depend on continuous"
    ):
        _create_stochastic_transition_params(
            regime=regime,  # type: ignore[arg-type]
            variable_info=variable_info,
            n_periods=3,
            grids={"mock": {"a": jnp.array([1, 2])}},
        )


def test_create_shock_params_invalid_dependency():
    @lcm.mark.stochastic
    def next_a(a, b, period, a_transition):
        return a_transition[a, b, period]

    variable_info = pd.DataFrame(
        {
            "is_stochastic": [True, False],
            "is_state": [True, False],
            "is_continuous": [False, True],
        },
        index=["a", "b"],
    )

    # _create_stochastic_transition_params expects nested format internally
    regime = RegimeMock(
        transitions={"mock": {"next_a": next_a}},  # type: ignore[dict-item]
    )

    with pytest.raises(
        ValueError, match="Stochastic transition functions cannot depend on continuous"
    ):
        _create_stochastic_transition_params(
            regime=regime,  # type: ignore[arg-type]
            variable_info=variable_info,
            n_periods=3,
            grids={"mock": {"a": jnp.array([1, 2])}},
        )
