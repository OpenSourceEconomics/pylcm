from collections.abc import Mapping
from types import MappingProxyType

import jax
from jax import Array, vmap
from jax import numpy as jnp

from lcm.exceptions import (
    InvalidInitialStatesError,
    InvalidRegimeTransitionProbabilitiesError,
)
from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.random import generate_simulation_keys
from lcm.state_action_space import create_state_action_space
from lcm.typing import Bool1D, Float1D, Int1D, ParamsDict, RegimeName
from lcm.utils import flatten_regime_namespace, normalize_regime_transition_probs


def create_regime_state_action_space(
    internal_regime: InternalRegime,
    states: MappingProxyType[str, Array],
) -> StateActionSpace:
    """Create the state-action space containing only the relevant subjects in a regime.

    Args:
        internal_regime: The internal regime instance.
        states: The current states of all subjects.
        subject_ids_in_regime: Indices of subjects in the current regime.

    Returns:
        The state-action space for the subjects in the regime.

    """
    query = "is_state and (enters_concurrent_valuation | enters_transition)"

    relevant_states_names = internal_regime.variable_info.query(query).index

    states_for_state_action_space = {
        sn: states[f"{internal_regime.name}__{sn}"] for sn in relevant_states_names
    }

    return create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states=states_for_state_action_space,
    )


def calculate_next_states(
    internal_regime: InternalRegime,
    optimal_actions: MappingProxyType[str, Array],
    period: int,
    age: float,
    params: dict[RegimeName, ParamsDict],
    states: MappingProxyType[str, Array],
    state_action_space: StateActionSpace,
    key: Array,
    subjects_in_regime: Bool1D,
) -> MappingProxyType[str, Array]:
    """Calculate next period states for subjects in a regime.

    Args:
        internal_regime: The internal regime instance.
        subjects_in_regime: Boolean array indicating if subject is in regime.
        optimal_actions: Optimal actions computed for these subjects.
        period: Current period.
        age: Age corresponding to current period.
        params: Model parameters for the regime.
        states: Current states for all subjects (all regimes).
        state_action_space: State-action space for subjects in this regime.
        key: JAX random key.

    Returns:
        Updated states dictionary with next period states for subjects in this regime.
        The returned dict contains states for all subjects, with updates only for
        those in the current regime.

    """
    # Identify stochastic transitions and generate random keys
    # ---------------------------------------------------------------------------------
    stochastic_next_function_names = [
        next_fn_name
        for next_fn_name, next_fn in flatten_regime_namespace(
            internal_regime.transitions
        ).items()
        if is_stochastic_transition(next_fn)
    ]

    key, stochastic_variables_keys = generate_simulation_keys(
        key=key,
        names=stochastic_next_function_names,
        n_initial_states=subjects_in_regime.shape[0],
    )

    # Compute next states using regime's transition functions
    # ---------------------------------------------------------------------------------
    next_state_vmapped = internal_regime.next_state_simulation_function

    states_with_next_prefix = next_state_vmapped(
        **state_action_space.states,
        **optimal_actions,
        **stochastic_variables_keys,
        period=period,
        age=age,
        params=params,
    )

    # Update global states array with computed next states for subjects in regime
    # ---------------------------------------------------------------------------------
    # The transition function adds a 'next_' prefix to all state names. We remove
    # this prefix and update only the entries corresponding to subjects in this regime.
    return _update_states_for_subjects(
        all_states=states,
        computed_next_states=states_with_next_prefix,
        subject_indices=subjects_in_regime,
    )


def calculate_next_regime_membership(
    internal_regime: InternalRegime,
    state_action_space: StateActionSpace,
    optimal_actions: MappingProxyType[str, Array],
    period: int,
    age: float,
    params: dict[RegimeName, ParamsDict],
    regime_id: Mapping[RegimeName, int],
    new_subject_regime_ids: Int1D,
    active_regimes_next_period: list[RegimeName],
    key: Array,
    subjects_in_regime: Bool1D,
) -> Int1D:
    """Calculate next period regime membership for subjects in a regime.

    Computes the probability distribution over regimes for the next period based on
    current states and actions, then draws random regime assignments for each subject.

    Args:
        internal_regime: The internal regime instance.
        state_action_space: State-action space for subjects in this regime.
        optimal_actions: Optimal actions computed for these subjects.
        period: Current period.
        age: Age corresponding to current period.
        params: Model parameters for the regime.
        regime_id: Mapping from regime names to integer IDs.
        new_subject_regime_ids: Array to update with next regime assignments.
        active_regimes_next_period: List of active regimes in the next period.
        key: JAX random key.
        subjects_in_regime: Boolean array indicating if subject is in regime.


    Returns:
        Updated array of regime IDs with next period assignments for subjects in this
        regime. The returned array contains regime IDs for all subjects, with updates
        only for those in the current regime.

    """
    # Compute regime transition probabilities
    # ---------------------------------------------------------------------------------
    regime_transition_probs = (
        internal_regime.internal_functions.regime_transition_probs.simulate(  # ty: ignore[possibly-missing-attribute]
            **state_action_space.states,
            **optimal_actions,
            period=period,
            age=age,
            params=params,
        )
    )
    normalized_regime_transition_probs = normalize_regime_transition_probs(
        regime_transition_probs, active_regimes_next_period
    )

    _validate_normalized_regime_transition_probs(
        normalized_regime_transition_probs,
        regime_name=internal_regime.name,
        period=period,
    )

    # Generate random keys and draw next regimes
    # ---------------------------------------------------------------------------------
    key, regime_transition_key = generate_simulation_keys(
        key=key,
        names=["regime_transition"],
        n_initial_states=subjects_in_regime.shape[0],
    )

    next_regime_ids = draw_key_from_dict(
        d=normalized_regime_transition_probs,
        keys=regime_transition_key["key_regime_transition"],
        regime_id=regime_id,
    )

    # Update global regime membership array
    # ---------------------------------------------------------------------------------
    return jnp.where(subjects_in_regime, next_regime_ids, new_subject_regime_ids)


def draw_key_from_dict(
    d: Mapping[str, Array], regime_id: Mapping[str, int], keys: Array
) -> Int1D:
    """Draw a random key from a dictionary of arrays.

    Args:
        d: Dictionary of arrays, all of the same length. The values in the arrays
            represent a probability distribution over the keys. That is, for the
            dictionary {'regime1': jnp.array([0.2, 0.5]),
            'regime2': jnp.array([0.8, 0.5])}, 0.2 + 0.8 = 1.0 and 0.5 + 0.5 = 1.0.
        regime_id: Mapping of regime names to regime ids.
        keys: JAX random keys.

    Returns:
        A random key from the dictionary for each entry in the arrays.

    """
    regime_names = list(d)
    regime_transition_probs = jnp.array(list(d.values())).T
    regime_ids = jnp.array([regime_id[name] for name in regime_names])

    def random_id(
        key: Array,
        p: Array,
    ) -> Int1D:
        return jax.random.choice(
            key,
            regime_ids,
            p=p,
        )

    random_ids = vmap(random_id, in_axes=(0, 0))

    return random_ids(keys, regime_transition_probs)


def _update_states_for_subjects(
    all_states: MappingProxyType[str, Array],
    computed_next_states: dict[str, Array],
    subject_indices: Bool1D,
) -> MappingProxyType[str, Array]:
    """Update the global states dictionary with next states for specific subjects.

    The transition functions add a 'next_' prefix to state variable names. This function
    removes that prefix and updates only the entries corresponding to the specified
    subjects, leaving other subjects' states unchanged.

    Args:
        all_states: Current states for all subjects across all regimes.
        computed_next_states: Newly computed states (with 'next_' prefix) for specific
            subjects.
        subject_indices: Indices of subjects whose states should be updated.

    Returns:
        Updated states dictionary with next states for the specified subjects.

    """
    updated_states = dict(all_states)
    for state_name, next_state_values in computed_next_states.items():
        updated_states[state_name.replace("next_", "")] = jnp.where(
            subject_indices,
            next_state_values,
            all_states[state_name.replace("next_", "")],
        )

    return MappingProxyType(updated_states)


def validate_flat_initial_states(
    flat_initial_states: dict[str, Array],
    internal_regimes: dict[RegimeName, InternalRegime],
) -> None:
    """Validate flat initial_states dict.

    Checks that:
    1. All required state names (across all regimes) are provided
    2. No extra/unknown state names are provided
    3. All arrays have the same length (same number of subjects)

    Args:
        flat_initial_states: Dict mapping state names to arrays.
        internal_regimes: Dict of internal regime instances.

    Raises:
        InvalidInitialStatesError: If validation fails with descriptive message.

    """
    # Collect all required state names across all regimes
    required_states: set[str] = set()
    for internal_regime in internal_regimes.values():
        regime_states = set(internal_regime.variable_info.query("is_state").index)
        required_states.update(regime_states)

    provided_states = set(flat_initial_states.keys())

    # Check for missing states
    missing = required_states - provided_states
    if missing:
        raise InvalidInitialStatesError(
            f"Missing initial states: {sorted(missing)}. "
            f"Required states are: {sorted(required_states)}"
        )

    # Check for extra states
    extra = provided_states - required_states
    if extra:
        raise InvalidInitialStatesError(
            f"Unknown initial states: {sorted(extra)}. "
            f"Valid states are: {sorted(required_states)}"
        )

    # Check array lengths are consistent
    if flat_initial_states:
        lengths = {name: len(arr) for name, arr in flat_initial_states.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise InvalidInitialStatesError(
                f"All initial state arrays must have the same length. "
                f"Got lengths: {lengths}"
            )


def convert_flat_to_nested_initial_states(
    flat_initial_states: dict[str, Array],
    internal_regimes: dict[RegimeName, InternalRegime],
) -> dict[RegimeName, dict[str, Array]]:
    """Convert flat initial_states dict to nested format.

    Takes user-provided flat format and converts to the nested format
    expected by internal simulation code.

    Args:
        flat_initial_states: Dict mapping state names to arrays.
            Example: {"wealth": arr, "health": arr}
        internal_regimes: Dict of internal regime instances.

    Returns:
        Nested dict mapping regime names to state dicts.
            Example: {"work": {"wealth": arr, "health": arr}, ...}

    """
    nested: dict[RegimeName, dict[str, Array]] = {}

    for regime_name, internal_regime in internal_regimes.items():
        regime_state_names = set(internal_regime.variable_info.query("is_state").index)
        nested[regime_name] = {
            state_name: flat_initial_states[state_name]
            for state_name in regime_state_names
        }

    return nested


def _validate_normalized_regime_transition_probs(
    normalized_probs: Mapping[str, Float1D],
    regime_name: str,
    period: int,
) -> None:
    probs = jnp.array(list(normalized_probs.values()))
    sum_probs = jnp.sum(probs, axis=0)
    if not jnp.allclose(sum_probs, 1.0):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' in period {period} "
            "do not sum to 1 after normalization. This indicates an error in the "
            "'next_regime' function of the regime."
        )
    if jnp.any(~jnp.isfinite(probs)):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-finite values in regime transition probabilities from "
            f"'{regime_name}' in period {period} after normalization. This usually "
            "means no active regime can be reached. Check that the 'next_regime' "
            f"function of the '{regime_name}' regime assigns positive probability to "
            "regimes that are active in the next period."
        )
