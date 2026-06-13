from types import MappingProxyType

import jax.numpy as jnp

from _lcm.engine import VariableInfo, Variables
from _lcm.grids import DiscreteGrid, categorical
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.next_state import (
    _create_discrete_stochastic_next_func,
    get_next_state_function_for_simulation,
    get_next_state_function_for_solution,
)
from _lcm.regime_building.processing import process_regimes
from lcm.ages import AgeGrid
from lcm.typing import ContinuousState, ScalarInt
from tests.test_models.deterministic.regression import dead, working_life


def test_get_next_state_function_with_solve_target():
    ages = AgeGrid(start=0, stop=4, step="Y")
    user_regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(idx) for idx, name in enumerate(user_regimes.keys())}
    )
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=user_regimes, derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
    )

    internal_working = regimes["working_life"]

    got_func = get_next_state_function_for_solution(
        transitions=internal_working.solution.transitions["working_life"],
        functions=internal_working.solution.functions,
    )

    flat_regime_params = {
        "discount_factor": jnp.asarray(1.0),
        "utility__disutility_of_work": jnp.asarray(1.0),
        "next_wealth__interest_rate": jnp.asarray(0.05),
    }
    action = {"labor_supply": jnp.asarray(1), "consumption": jnp.asarray(10.0)}
    state = {"wealth": jnp.asarray(20.0)}

    got = got_func(
        **action,
        **state,
        period=jnp.int32(1),
        age=jnp.asarray(1.0),
        **flat_regime_params,
    )
    assert got == {"next_wealth": 1.05 * (20 - 10)}


def test_get_next_state_function_with_simulate_target():
    """Outputs are nested by target regime: `{target: {next_state: array}}`.

    The combined function dispatches inputs to the per-target DAG and
    returns a mapping from target regime name to that target's
    `{next_<state>: array}` outputs, matching what
    `_update_states_for_subjects` consumes.
    """

    def f_a(state: ContinuousState) -> ContinuousState:
        return state * 2.0

    def f_b(state: ContinuousState) -> ContinuousState:
        return state + 1.0

    @categorical(ordered=False)
    class MockCategory:
        cat_0: ScalarInt
        cat_1: ScalarInt

    all_grids = MappingProxyType(
        {"mock": MappingProxyType({"b": DiscreteGrid(MockCategory)})}
    )
    variables = Variables(
        info=MappingProxyType(
            {"b": VariableInfo(kind="state", topology="discrete", is_process=False)}
        )
    )
    transitions = MappingProxyType(
        {"mock": MappingProxyType({"next_a": f_a, "next_b": f_b})}
    )
    functions = MappingProxyType({"utility": lambda: 0})

    got_func = get_next_state_function_for_simulation(
        transitions=transitions,  # ty: ignore[invalid-argument-type]
        functions=functions,  # ty: ignore[invalid-argument-type]
        all_grids=all_grids,
        variables=variables,
    )

    got = got_func(state=jnp.array([1.0, 2.0]))

    assert set(got.keys()) == {"mock"}
    assert set(got["mock"].keys()) == {"next_a", "next_b"}
    assert jnp.array_equal(got["mock"]["next_a"], jnp.array([2.0, 4.0]))
    assert jnp.array_equal(got["mock"]["next_b"], jnp.array([2.0, 3.0]))


def test_create_stochastic_next_func():
    labels = jnp.arange(2, dtype=jnp.int32)
    got_func = _create_discrete_stochastic_next_func(
        target="t", next_state_name="next_a", labels=labels
    )

    key = jnp.arange(2, dtype="uint32")  # PRNG dtype
    weights = jnp.array([0.0, 1])

    got = got_func(key_t__next_a=key, weight_t__next_a=weights)

    assert jnp.array_equal(got, jnp.array(1))
