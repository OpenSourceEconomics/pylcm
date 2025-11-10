import jax
from dags.tree import flatten_to_qnames
from jax import Array, vmap
from jax import numpy as jnp

from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.random import generate_simulation_keys
from lcm.state_action_space import create_state_action_space
from lcm.typing import Int1D, RegimeName


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
    states: dict[str, Array],
    state_action_space: StateActionSpace,
    key: Array,
) -> dict[str, Array]:
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

    next_state_vmapped = internal_regime.next_state_simulation_function

    states_with_next_prefix = next_state_vmapped(
        **state_action_space.states,  # type: ignore[assignment]
        **optimal_actions,
        **stochastic_variables_keys,
        period=period,
        params=params,
    )

    # 'next_' prefix is added by the next_state function, but needs to be
    # removed for the next iteration of the loop, where these will be the
    # current states.
    return {
        k.replace("next_", ""): states[k.replace("next_", "")]
        .at[subjects_in_regime]
        .set(v)
        for k, v in states_with_next_prefix.items()
    }


def calculate_next_regime_membership(
    internal_regime: InternalRegime,
    subjects_in_regime: Int1D,
    state_action_space: StateActionSpace,
    optimal_actions: dict[str, Array],
    period: int,
    params: dict[RegimeName, dict],
    regime_name_to_id: dict[RegimeName, int],
    new_subject_regime_ids: Int1D,
    key: Array,
) -> Int1D:
    _regime_transition_probs = (
        internal_regime.internal_functions.regime_transition_probs["simulate"](
            **state_action_space.states,
            **optimal_actions,
            period=period,
            params=params,
        )
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


def draw_key_from_dict(
    d: dict[str, Array], regime_name_to_id: dict[str, int], keys: Array
) -> list[str]:
    """Draw a random key from a dictionary of arrays.

    Args:
        d: Dictionary of arrays, all of the same length. The values in the arrays
            represent a probability distribution over the keys. That is, for the
            dictionary {'regime1': jnp.array([0.2, 0.5]), 'regime2': jnp.array([0.8, 0.5])},
            0.2 + 0.8 = 1.0 and 0.5 + 0.5 = 1.0.
        keys: JAX random keys.

    Returns:
        A random key from the dictionary for each entry in the arrays.

    """
    regime_ids = jnp.array([regime_name_to_id[key] for key in d])

    def draw_single_key(
        key: Array,
        p: Array,
    ) -> str:
        return jax.random.choice(
            key,
            regime_ids,
            p=p,
        )

    draw_key = vmap(draw_single_key, in_axes=(0, 0))
    draw = draw_key(keys, jnp.array(list(d.values())).T)
    return draw
