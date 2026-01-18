from typing import cast

import jax.numpy as jnp
from pybaum import tree_equal

from lcm.ages import AgeGrid
from lcm.input_processing import process_regimes
from lcm.interfaces import InternalFunctions, PhaseVariantContainer, Target
from lcm.next_state import _create_stochastic_next_func, get_next_state_function
from lcm.typing import ContinuousState, FloatND, InternalUserFunction, ParamsDict
from tests.test_models.deterministic.regression import RegimeId, dead, working


def test_get_next_state_function_with_solve_target():
    ages = AgeGrid(start=0, stop=4, step="Y")
    internal_regimes = process_regimes(
        regimes=[working, dead],
        ages=ages,
        regime_id_cls=RegimeId,
        enable_jit=True,
    )

    internal_working = internal_regimes[working.name]

    got_func = get_next_state_function(
        transitions=internal_working.transitions[working.name],
        functions=internal_working.functions,
        grids={
            working.name: internal_working.grids,
            dead.name: internal_regimes[dead.name].grids,
        },
        target=Target.SOLVE,
    )

    params = {
        "discount_factor": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }

    action = {"labor_supply": 1, "consumption": 10}
    state = {"wealth": 20}

    got = got_func(**action, **state, period=1, age=1.0, params=params)
    assert got == {"next_wealth": 1.05 * (20 - 10)}


def test_get_next_state_function_with_simulate_target():
    def f_a(state: ContinuousState, params: ParamsDict) -> ContinuousState:  # noqa: ARG001
        return state[0]

    def f_b(state: ContinuousState, params: ParamsDict) -> ContinuousState:  # noqa: ARG001
        return None  # ty: ignore[invalid-return-type]

    def f_weight_b(state: ContinuousState, params: ParamsDict) -> FloatND:  # noqa: ARG001
        return jnp.array([0.0, 1.0])

    grids = {"mock": {"b": jnp.arange(2)}}
    mock_transition_solve = lambda *args, params, **kwargs: {"mock": 1.0}  # noqa: E731, ARG005
    mock_transition_simulate = lambda *args, params, **kwargs: {  # noqa: E731, ARG005
        "mock": jnp.array([1.0])
    }
    internal_functions = InternalFunctions(
        utility=lambda: 0,  # ty: ignore[invalid-argument-type]
        constraints={},
        transitions={"next_a": f_a, "next_b": f_b},  # ty: ignore[invalid-argument-type]
        functions={"f_weight_b": f_weight_b},  # ty: ignore[invalid-argument-type]
        regime_transition_probs=PhaseVariantContainer(
            solve=mock_transition_solve, simulate=mock_transition_simulate
        ),
    )
    got_func = get_next_state_function(
        transitions=cast(
            "dict[str, InternalUserFunction]", internal_functions.transitions
        ),
        functions=internal_functions.functions,
        grids=grids,
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
