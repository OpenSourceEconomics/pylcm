import inspect

from dags.tree import flatten_to_qnames
from jax import Array

from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.random import generate_simulation_keys
from lcm.state_action_space import create_state_action_space
from lcm.typing import Int1D, ParamsDict, Period, RegimeName


def get_regime_name_to_id_mapping(internal_regimes):
    return {
        regime.name: i
        for regime, i in zip(
            internal_regimes.values(), range(len(internal_regimes)), strict=False
        )
    }


def create_regime_state_action_space(
    regime_name: RegimeName,
    states: dict[str, Array],
    internal_regime: InternalRegime,
    subjects_in_regime: Int1D,
    *,
    is_last_period: bool,
):
    if is_last_period:
        query = "is_state and enters_concurrent_valuation"
    else:
        query = "is_state and (enters_concurrent_valuation | enters_transition)"

    # Create state action space with current subjects
    states_for_state_action_space = {
        state_name: states[f"{regime_name}__{state_name}"][subjects_in_regime]
        for state_name in internal_regime.variable_info.query(query).index
    }
    return create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states=states_for_state_action_space,
        is_last_period=is_last_period,
    )


def calculate_next_states(
    internal_regime: InternalRegime,
    subjects_in_regime: Int1D,
    optimal_actions: dict[str, Array],
    period: int,
    params: dict[RegimeName, dict],
    regime_name: RegimeName,
    states: dict[str, Array],
    state_action_space: StateActionSpace,
    key: Array,
):
    next_vars = _calculate_next_vars(
        internal_regime=internal_regime,
        subjects_in_regime=subjects_in_regime,
        optimal_actions=optimal_actions,
        period=period,
        params=params,
        regime_name=regime_name,
        state_action_space=state_action_space,
        key=key,
    )

    next_state_vmapped = internal_regime.next_state_simulation_function
    signature = inspect.signature(next_state_vmapped)
    parameters = set(signature.parameters)

    next_state_input = {parameter: next_vars[parameter] for parameter in parameters}

    states_with_next_prefix = next_state_vmapped(**next_state_input)

    # 'next_' prefix is added by the next_state function, but needs to be
    # removed for the next iteration of the loop, where these will be the
    # current states.
    return {
        k.replace("next_", ""): states[k.replace("next_", "")]
        .at[subjects_in_regime]
        .set(v)
        for k, v in states_with_next_prefix.items()
    }


def calculate_next_regime_membership():
    signature = inspect.signature(internal_regime.regime_transition_probs["simulate"])
    parameters = set(signature.parameters)
    next_regime_input = {parameter: next_vars[parameter] for parameter in parameters}
    _regime_transition_probs = internal_regime.regime_transition_probs["simulate"](
        **next_regime_input
    )

    key, regime_transition_key = generate_simulation_keys(
        key=key,
        names=["regime_transition"],
        n_initial_states=subjects_in_regime.shape[0],
    )
    new_regimes = draw_key_from_dict(
        d=_regime_transition_probs,
        keys=regime_transition_key["key_regime_transition"],
        regime_name_to_id=regime_name_to_id,
    )
    new_subject_regime_ids = new_subject_regime_ids.at[subjects_in_regime].set(
        new_regimes
    )
    return new_subject_regime_ids


def _calculate_next_vars(
    internal_regime: InternalRegime,
    subjects_in_regime: Int1D,
    optimal_actions: dict[str, Array],
    period: int,
    params: dict[RegimeName, dict],
    regime_name: RegimeName,
    state_action_space: StateActionSpace,
    key: Array,
) -> dict[str, Array | Period | ParamsDict]:
    stochastic_next_function_names = [
        next_fn_name
        for next_fn_name, next_fn in flatten_to_qnames(
            internal_regime.transitions
        ).items()
        if is_stochastic_transition(next_fn)
    ]

    key, stochastic_variables_keys = generate_simulation_keys(
        key=key,
        names=stochastic_next_function_names,
        n_initial_states=subjects_in_regime.shape[0],
    )

    return (
        state_action_space.states  # type: ignore[assignment]
        | optimal_actions
        | stochastic_variables_keys
        | {"period": period, "params": params[regime_name]}
    )
