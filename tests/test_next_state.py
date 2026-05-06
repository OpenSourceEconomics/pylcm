from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
import pandas as pd

from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid
from lcm.regime_building import process_regimes
from lcm.regime_building.next_state import (
    _create_discrete_stochastic_next_func,
    get_next_state_function_for_simulation,
    get_next_state_function_for_solution,
)
from lcm.typing import ContinuousState
from tests.test_models.deterministic.regression import dead, working_life


def test_get_next_state_function_with_solve_target():
    ages = AgeGrid(start=0, stop=4, step="Y")
    regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: idx for idx, name in enumerate(regimes.keys())}
    )
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
    )

    internal_working = internal_regimes["working_life"]

    got_func = get_next_state_function_for_solution(
        transitions=internal_working.solve_functions.transitions["working_life"],
        functions=internal_working.solve_functions.functions,
    )

    flat_regime_params = {
        "discount_factor": 1.0,
        "utility__disutility_of_work": 1.0,
        "next_wealth__interest_rate": 0.05,
    }
    action = {"labor_supply": 1, "consumption": 10}
    state = {"wealth": 20}

    got = got_func(
        **action,
        **state,
        period=1,
        age=1.0,
        **flat_regime_params,
    )
    assert got == {"next_wealth": 1.05 * (20 - 10)}


def test_get_next_state_function_with_simulate_target():
    """Outputs are namespaced by target regime: `<target>__<next_state>`.

    The combined function dispatches inputs to the per-target DAG and
    qualifies each output with its target name, matching the format
    `_update_states_for_subjects` consumes.
    """

    def f_a(state: ContinuousState) -> ContinuousState:
        return state * 2.0

    def f_b(state: ContinuousState) -> ContinuousState:
        return state + 1.0

    @dataclass
    class MockCategory:
        cat_0: int = 0
        cat_1: int = 1

    all_grids = MappingProxyType(
        {"mock": MappingProxyType({"b": DiscreteGrid(MockCategory)})}
    )
    variable_info = pd.DataFrame({"is_shock": [False]}, index=["b"])
    transitions = MappingProxyType(
        {"mock": MappingProxyType({"next_a": f_a, "next_b": f_b})}
    )
    functions = MappingProxyType({"utility": lambda: 0})

    got_func = get_next_state_function_for_simulation(
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        functions=functions,  # ty: ignore[invalid-argument-type]
        all_grids=all_grids,
        variable_info=variable_info,
    )

    got = got_func(state=jnp.array([1.0, 2.0]))

    assert set(got.keys()) == {"mock__next_a", "mock__next_b"}
    assert jnp.array_equal(got["mock__next_a"], jnp.array([2.0, 4.0]))
    assert jnp.array_equal(got["mock__next_b"], jnp.array([2.0, 3.0]))


def test_create_stochastic_next_func():
    labels = jnp.arange(2)
    got_func = _create_discrete_stochastic_next_func(name="a", labels=labels)

    key = jnp.arange(2, dtype="uint32")  # PRNG dtype
    weights = jnp.array([0.0, 1])

    got = got_func(key_a=key, weight_a=weights)

    assert jnp.array_equal(got, jnp.array(1))
