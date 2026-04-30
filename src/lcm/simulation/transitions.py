"""Simulation-phase state and regime transitions.

These functions apply pre-compiled transition functions to subject arrays during
forward simulation. They are distinct from `lcm.regime_building`, which compiles
transition functions from user-defined regimes; this module executes them.

"""

from types import MappingProxyType

import jax
from dags.tree import tree_path_from_qname
from jax import Array, vmap
from jax import numpy as jnp

from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.simulation.random import generate_simulation_keys
from lcm.state_action_space import _validate_all_states_present
from lcm.typing import (
    ActionName,
    Bool1D,
    FlatRegimeParams,
    Int1D,
    RegimeName,
    RegimeNamesToIds,
)
from lcm.utils.namespace import flatten_regime_namespace


def create_regime_state_action_space(
    *,
    internal_regime: InternalRegime,
    states: MappingProxyType[str, Array],
    regime_params: FlatRegimeParams,
) -> StateActionSpace:
    """Create the state-action space containing only the relevant subjects in a regime.

    Continuous action grids declared with `pass_points_at_runtime=True` are
    completed from `regime_params` (via
    `InternalRegime.state_action_space`) — otherwise they would carry the
    NaN placeholder used during compilation, which propagates into
    `optimal_actions` and ultimately `next_states`.

    Args:
        internal_regime: The internal regime instance.
        states: The current states of all subjects.
        regime_params: Flat regime parameters supplied at runtime, used to
            substitute runtime-supplied action gridpoints.

    Returns:
        The state-action space for the subjects in the regime.

    """
    base = internal_regime.state_action_space(regime_params=regime_params)

    relevant_state_names = internal_regime.variable_info.query("is_state").index
    states_for_state_action_space = {
        sn: states[f"{internal_regime.name}__{sn}"] for sn in relevant_state_names
    }
    _validate_all_states_present(
        provided_states=states_for_state_action_space,
        required_state_names=set(relevant_state_names),
    )

    return base.replace(states=MappingProxyType(states_for_state_action_space))


def calculate_next_states(
    *,
    internal_regime: InternalRegime,
    optimal_actions: MappingProxyType[ActionName, Array],
    period: int,
    age: float,
    regime_params: FlatRegimeParams,
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
        regime_params: Flat regime parameters.
        states: Current states for all subjects (all regimes).
        state_action_space: State-action space for subjects in this regime.
        key: JAX random key.

    Returns:
        Updated states dictionary with next period states for subjects in this regime.
        Immutable mapping of updated states for all subjects, with updates only for
        those in the current regime.

    """
    # Identify stochastic transitions and generate random keys
    # ---------------------------------------------------------------------------------
    stochastic_transition_names = (
        internal_regime.simulate_functions.stochastic_transition_names
    )
    stochastic_next_function_names = [
        next_func_name
        for next_func_name in flatten_regime_namespace(
            internal_regime.simulate_functions.transitions
        )
        if tree_path_from_qname(next_func_name)[-1] in stochastic_transition_names
    ]
    # There is a bug that sometimes changes the order of the names,
    # sorting fixes this
    stochastic_next_function_names.sort()

    key, stochastic_variables_keys = generate_simulation_keys(
        key=key,
        names=stochastic_next_function_names,
        n_initial_states=subjects_in_regime.shape[0],
    )

    # Compute next states using regime's transition functions
    # ---------------------------------------------------------------------------------
    next_state_vmapped = internal_regime.simulate_functions.next_state

    states_with_next_prefix = next_state_vmapped(
        **state_action_space.states,
        **optimal_actions,
        **stochastic_variables_keys,
        period=period,
        age=age,
        **regime_params,
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
    *,
    internal_regime: InternalRegime,
    state_action_space: StateActionSpace,
    optimal_actions: MappingProxyType[ActionName, Array],
    period: int,
    age: float,
    regime_params: FlatRegimeParams,
    regime_names_to_ids: MappingProxyType[RegimeName, int],
    new_subject_regime_ids: Int1D,
    active_regimes_next_period: tuple[RegimeName, ...],
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
        regime_params: Flat regime parameters.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        new_subject_regime_ids: Array to update with next regime assignments.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key.
        subjects_in_regime: Boolean array indicating if subject is in regime.


    Returns:
        Updated array of regime IDs with next period assignments for subjects in this
        regime. The returned array contains regime IDs for all subjects, with updates
        only for those in the current regime.

    """
    # Compute regime transition probabilities
    # ---------------------------------------------------------------------------------
    regime_transition_probs: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
        internal_regime.simulate_functions.compute_regime_transition_probs(  # ty: ignore[call-non-callable]
            **state_action_space.states,
            **optimal_actions,
            period=period,
            age=age,
            **regime_params,
        )
    )
    active_regime_probs = MappingProxyType(
        {r: regime_transition_probs[r] for r in active_regimes_next_period}
    )

    # Generate random keys and draw next regimes
    # ---------------------------------------------------------------------------------
    key, regime_transition_key = generate_simulation_keys(
        key=key,
        names=["regime_transition"],
        n_initial_states=subjects_in_regime.shape[0],
    )

    next_regime_ids = draw_key_from_dict(
        d=active_regime_probs,
        regime_names_to_ids=regime_names_to_ids,
        keys=regime_transition_key["key_regime_transition"],
    )

    # Update global regime membership array
    # ---------------------------------------------------------------------------------
    return jnp.where(subjects_in_regime, next_regime_ids, new_subject_regime_ids)


def draw_key_from_dict(
    *,
    d: MappingProxyType[RegimeName, Array],
    regime_names_to_ids: RegimeNamesToIds,
    keys: Array,
) -> Int1D:
    """Draw a random key from a dictionary of arrays.

    Args:
        d: Immutable mapping of arrays, all of the same length. The values in the arrays
            represent a probability distribution over the keys. That is, for the
            dictionary {'regime1': jnp.array([0.2, 0.5]),
            'regime2': jnp.array([0.8, 0.5])}, 0.2 + 0.8 = 1.0 and 0.5 + 0.5 = 1.0.
        regime_names_to_ids: Mapping of regime names to regime ids.
        keys: JAX random keys.

    Returns:
        A random key from the dictionary for each entry in the arrays.

    """
    regime_names = list(d)
    regime_transition_probs = jnp.array(list(d.values())).T
    regime_ids = jnp.array(
        [regime_names_to_ids[regime_name] for regime_name in regime_names]
    )

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
    *,
    all_states: MappingProxyType[str, Array],
    computed_next_states: MappingProxyType[str, Array],
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
    for next_state_name, next_state_values in computed_next_states.items():
        # Namespaced outputs: "regime__next_wealth" → "regime__wealth"
        state_name = next_state_name.replace("__next_", "__", 1)
        updated_states[state_name] = jnp.where(
            subject_indices,
            next_state_values,
            all_states[state_name],
        )

    return MappingProxyType(updated_states)
