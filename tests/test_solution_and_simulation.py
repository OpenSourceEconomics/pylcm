from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from pybaum import tree_equal, tree_map

from lcm.input_processing import process_model
from lcm.max_Q_over_c import (
    get_argmax_and_max_Q_over_c,
    get_max_Q_over_c,
)
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import create_state_space_info
from tests.test_models import get_model
from tests.test_models.deterministic import RetirementStatus
from tests.test_models.deterministic import utility as iskhakov_et_al_2017_utility
from tests.test_models.discrete_deterministic import ConsumptionChoice

if TYPE_CHECKING:
    from typing import Any

    from lcm.typing import BoolND, DiscreteAction, DiscreteState
    from lcm.user_model import Model

# ======================================================================================
# Test cases
# ======================================================================================


STRIPPED_DOWN_AND_DISCRETE_MODELS = [
    "iskhakov_et_al_2017_stripped_down",
    "iskhakov_et_al_2017_discrete",
]


# ======================================================================================
# Solve
# ======================================================================================


def test_solve_stripped_down():
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    model.solve(params)


def test_solve_fully_discrete():
    model = get_model("iskhakov_et_al_2017_discrete", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    model.solve(params)


# ======================================================================================
# Simulate
# ======================================================================================


def test_solve_and_simulate_stripped_down():
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    params = tree_map(lambda _: 0.2, model.params_template)

    model.solve_and_simulate(
        params,
        initial_states={
            "wealth": jnp.array([1.0, 10.0, 50.0]),
        },
        additional_targets=["age"] if "age" in model.functions else None,
    )


def test_solve_and_simulate_fully_discrete():
    model = get_model("iskhakov_et_al_2017_discrete", n_periods=3)

    params = tree_map(lambda _: 0.2, model.params_template)

    model.solve_and_simulate(
        params,
        initial_states={
            "wealth": jnp.array([1.0, 10.0, 50.0]),
        },
        additional_targets=["age"] if "age" in model.functions else None,
    )


@pytest.mark.parametrize(
    "model",
    [get_model(name, n_periods=3) for name in STRIPPED_DOWN_AND_DISCRETE_MODELS],
    ids=STRIPPED_DOWN_AND_DISCRETE_MODELS,
)
def test_solve_then_simulate_is_equivalent_to_solve_and_simulate(model: Model) -> None:
    """Test that solve_and_simulate creates same output as solve then simulate."""
    # solve then simulate
    # ==================================================================================

    # solve
    params = tree_map(lambda _: 0.2, model.params_template)
    V_arr_dict = model.solve(params)

    # simulate using solution
    solve_then_simulate = model.simulate(
        params,
        V_arr_dict=V_arr_dict,
        initial_states={
            "wealth": jnp.array([1.0, 10.0, 50.0]),
        },
    )

    # solve and simulate
    # ==================================================================================
    solve_and_simulate = model.solve_and_simulate(
        params,
        initial_states={
            "wealth": jnp.array([1.0, 10.0, 50.0]),
        },
    )

    assert tree_equal(solve_then_simulate, solve_and_simulate)


@pytest.mark.parametrize(
    "model",
    [get_model("iskhakov_et_al_2017", n_periods=3)],
    ids=["iskhakov_et_al_2017"],
)
def test_simulate_iskhakov_et_al_2017(model: Model) -> None:
    # solve model
    params = tree_map(lambda _: 0.9, model.params_template)
    V_arr_dict = model.solve(params)

    # simulate using solution
    model.simulate(
        params,
        V_arr_dict=V_arr_dict,
        initial_states={
            "wealth": jnp.array([10.0, 10.0, 20.0]),
            "lagged_retirement": jnp.array(
                [
                    RetirementStatus.working,
                    RetirementStatus.retired,
                    RetirementStatus.retired,
                ]
            ),
        },
    )


# ======================================================================================
# Create compute conditional continuation value
# ======================================================================================


def test_get_max_Q_over_c():
    model = process_model(
        get_model("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    state_space_info = create_state_space_info(
        internal_model=model,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        internal_model=model,
        next_state_space_info=state_space_info,
        period=model.n_periods - 1,
    )

    max_Q_over_c = get_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=("consumption",),
        states_and_discrete_actions_names=(),
    )

    val = max_Q_over_c(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=jnp.array(RetirementStatus.retired),
        wealth=jnp.array(30),
        params=params,
        next_V_arr=jnp.empty(0),
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=jnp.array(30.0),
        working=jnp.array(RetirementStatus.working),
        disutility_of_work=1.0,
    )


def test_get_max_Q_over_c_with_discrete_model():
    model = process_model(
        get_model("iskhakov_et_al_2017_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    state_space_info = create_state_space_info(
        internal_model=model,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        internal_model=model,
        next_state_space_info=state_space_info,
        period=model.n_periods - 1,
    )

    max_Q_over_c = get_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=(),
        states_and_discrete_actions_names=(),
    )

    val = max_Q_over_c(
        consumption=jnp.array([ConsumptionChoice.low, ConsumptionChoice.high]),
        retirement=jnp.array(RetirementStatus.retired),
        wealth=jnp.array(2),
        params=params,
        next_V_arr=jnp.empty(0),
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=jnp.array(2),
        working=jnp.array(RetirementStatus.working),
        disutility_of_work=1.0,
    )


# ======================================================================================
# Test argmax_and_max_Q_over_c
# ======================================================================================


def test_argmax_and_max_Q_over_c():
    model = process_model(
        get_model("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    state_space_info = create_state_space_info(
        internal_model=model,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        internal_model=model,
        next_state_space_info=state_space_info,
        period=model.n_periods - 1,
    )

    argmax_and_max_Q_over_c = get_argmax_and_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=("consumption",),
    )

    policy, val = argmax_and_max_Q_over_c(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=jnp.array(RetirementStatus.retired),
        wealth=jnp.array(30),
        params=params,
        next_V_arr=jnp.empty(0),
    )
    assert policy == 2
    assert val == iskhakov_et_al_2017_utility(
        consumption=jnp.array(30.0),
        working=jnp.array(RetirementStatus.working),
        disutility_of_work=1.0,
    )


def test_argmax_and_max_Q_over_c_with_discrete_model():
    model = process_model(
        get_model("iskhakov_et_al_2017_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    state_space_info = create_state_space_info(
        internal_model=model,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        internal_model=model,
        next_state_space_info=state_space_info,
        period=model.n_periods - 1,
    )

    argmax_and_max_Q_over_c = get_argmax_and_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=(),
    )

    _argmax, _max = argmax_and_max_Q_over_c(
        consumption=jnp.array([ConsumptionChoice.low, ConsumptionChoice.high]),
        retirement=jnp.array(RetirementStatus.retired),
        wealth=jnp.array(2),
        params=params,
        next_V_arr=jnp.empty(0),
    )
    assert _argmax == 1
    assert _max == iskhakov_et_al_2017_utility(
        consumption=jnp.array(2),
        working=jnp.array(RetirementStatus.working),
        disutility_of_work=1.0,
    )


# ======================================================================================
# Test constraints with _period argument
# ======================================================================================


def test_solve_with_period_argument_in_constraint():
    model = get_model("iskhakov_et_al_2017", n_periods=3)

    def absorbing_retirement_constraint(
        retirement: DiscreteAction,
        lagged_retirement: DiscreteState,
        _period: int,
    ) -> BoolND:
        return jnp.logical_or(
            retirement == RetirementStatus.retired,
            lagged_retirement == RetirementStatus.working,
        )

    model.constraints["absorbing_retirement_constraint"] = (
        absorbing_retirement_constraint
    )

    params = tree_map(lambda _: 0.2, model.params_template)
    model.solve(params)


# ======================================================================================
# Test that order of states / actions does not matter
# ======================================================================================


def _reverse_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Reverse the order of keys in a dictionary."""
    return {k: d[k] for k in reversed(list(d))}


def test_order_of_states_and_actions_does_not_matter():
    model = get_model("iskhakov_et_al_2017", n_periods=3)

    # Create a new model with the order of states and actions swapped
    model_swapped = model.replace(
        states=_reverse_dict(model.states),
        actions=_reverse_dict(model.actions),
    )

    params = tree_map(lambda _: 0.2, model.params_template)
    V_arr_dict = model.solve(params)

    V_arr_dict_swapped = model_swapped.solve(params)

    assert tree_equal(V_arr_dict, V_arr_dict_swapped)
