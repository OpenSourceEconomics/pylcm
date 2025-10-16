from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
from pybaum import tree_equal

from lcm.input_processing import process_regime
from lcm.interfaces import InternalRegime, ShockType, Target
from lcm.next_state import _create_stochastic_next_func, get_next_state_function
from tests.test_models.utils import get_regime

if TYPE_CHECKING:
    from lcm.typing import ContinuousState, FloatND, ParamsDict


def test_get_next_state_function_with_solve_target():
    internal_regime = process_regime(
        get_regime("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )
    got_func = get_next_state_function(
        internal_functions=internal_regime.internal_functions,
        grids=internal_regime.grids,
        next_states=("wealth",),
        target=Target.SOLVE,
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }

    action = {"retirement": 1, "consumption": 10}
    state = {"wealth": 20}

    got = got_func(**action, **state, _period=1, params=params)
    assert got == {"next_wealth": 1.05 * (20 - 10)}


def test_get_next_state_function_with_simulate_target():
    def f_a(state: ContinuousState, params: ParamsDict) -> ContinuousState:  # noqa: ARG001
        return state[0]

    def f_b(state: ContinuousState, params: ParamsDict) -> ContinuousState:  # noqa: ARG001
        return None  # type: ignore[return-value]

    def f_weight_b(state: ContinuousState, params: ParamsDict) -> FloatND:  # noqa: ARG001
        return jnp.array([0.0, 1.0])

    grids = {"b": jnp.arange(2)}

    internal_regime = InternalRegime(
        utility=lambda: 0,  # type: ignore[arg-type]
        constraints={},
        transitions={"next_a": f_a, "next_b": f_b},  # type: ignore[dict-item]
        functions={"f_weight_b": f_weight_b},  # type: ignore[dict-item]
        grids=grids,
        gridspecs={},
        variable_info=pd.DataFrame(),
        params={},
        random_utility_shocks=ShockType.NONE,
        n_periods=1,
    )

    got_func = get_next_state_function(
        internal_functions=internal_regime.internal_functions,
        grids=internal_regime.grids,
        next_states=("a", "b"),
        target=Target.SIMULATE,
    )

    key = jnp.arange(2, dtype="uint32")
    got = got_func(state=jnp.arange(2), key_b=key, params={})

    expected = {"a": jnp.array([0]), "b": jnp.array([1])}
    assert tree_equal(expected, got)


def test_create_stochastic_next_func():
    labels = jnp.arange(2)
    got_func = _create_stochastic_next_func(name="a", labels=labels)

    key = jnp.arange(2, dtype="uint32")  # PRNG dtype
    weights = jnp.array([0.0, 1])

    got = got_func(key_a=key, weight_a=weights)

    assert jnp.array_equal(got, jnp.array(1))
