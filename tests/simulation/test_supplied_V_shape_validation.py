"""A supplied value function must match the state space of the regime it belongs to.

`simulate()` accepts a solution the caller computed earlier or loaded from disk.
Nothing about that array is self-describing: a regime's value function is a table
indexed by its states, and a regime declaring no states has a single number. An
array of the wrong rank is not a type error — broadcasting it against a scalar is
a perfectly legal operation — so it flows into the Bellman equation and either
surfaces far away as a confusing shape error or, when the ranks happen to be
compatible, silently averages over a dimension the regime does not have.

The check therefore has to live where the declared state space is known.
"""

import re
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ScalarInt

_LAST_AGE = 22


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def _utility(consumption):
    return jnp.log(consumption)


def _next_regime(wealth, age):
    leaves = (wealth >= 3.0) | (age >= _LAST_AGE - 1)
    return jnp.where(leaves, RegimeId.gone, RegimeId.alive)


def _build_model():
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        functions={"utility": _utility},
    )
    gone = Regime(transition=None, functions={"utility": lambda: jnp.array(5.0)})
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )


_PARAMS = {
    "alive": {
        "utility": {},
        "koopmans_aggregator": {"discount_factor": 0.95},
        "next_wealth": {},
        "next_regime": {},
    },
    "gone": {"utility": {}},
}

_INITIAL_CONDITIONS = {
    "age": jnp.array([20.0]),
    "wealth": jnp.array([5.0]),
    "regime_id": jnp.array([RegimeId.alive]),
}


def _simulate_with(period_to_regime_to_V_arr):
    return _build_model().simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        log_level="debug",
    )


def test_a_stateless_regime_given_a_nonscalar_value_function_is_rejected():
    """A regime with no states must be handed a single number, not an array."""
    model = _build_model()
    solution = model.solve(params=_PARAMS, log_level="debug")
    corrupted = MappingProxyType(
        {
            period: MappingProxyType(
                {
                    name: (jnp.zeros(2) if name == "gone" else V)
                    for name, V in per_regime.items()
                }
            )
            for period, per_regime in solution.items()
        }
    )

    with pytest.raises(
        InvalidValueFunctionError, match=re.compile("gone", re.IGNORECASE)
    ):
        _simulate_with(corrupted)


def test_a_stateful_regime_given_an_extra_axis_is_rejected():
    """A regime with one state must be handed a one-dimensional value function."""
    model = _build_model()
    solution = model.solve(params=_PARAMS, log_level="debug")
    corrupted = MappingProxyType(
        {
            period: MappingProxyType(
                {
                    name: (jnp.zeros((4, 1)) if name == "alive" else V)
                    for name, V in per_regime.items()
                }
            )
            for period, per_regime in solution.items()
        }
    )

    with pytest.raises(
        InvalidValueFunctionError, match=re.compile("alive", re.IGNORECASE)
    ):
        _simulate_with(corrupted)


def test_a_correctly_shaped_solution_is_accepted():
    """The solver's own output passes the check unchanged."""
    model = _build_model()
    solution = model.solve(params=_PARAMS, log_level="debug")
    result = _simulate_with(solution)
    first_period_wealth = np.atleast_1d(result.to_dataframe().loc[0, "wealth"])
    np.testing.assert_array_almost_equal(first_period_wealth, [5.0], decimal=6)
