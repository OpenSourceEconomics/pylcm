from collections.abc import Mapping
from types import MappingProxyType

import jax
from dags.tree import tree_path_from_qname
from jax import Array, vmap
from jax import numpy as jnp

from lcm._utils.namespace import flatten_regime_namespace
from lcm._utils.state_action_space import create_state_action_space
from lcm.grids import DiscreteGrid
from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.simulation.random import generate_simulation_keys
from lcm.typing import (
    Bool1D,
    FlatRegimeParams,
    Int1D,
    RegimeName,
    RegimeNamesToIds,
)

# Sentinel for categorical states not in initial conditions.  Using int32 min
# instead of -1 so that JAX indexing produces obviously wrong values rather than
# silently returning the last element.
MISSING_CAT_CODE = jnp.iinfo(jnp.int32).min


def get_regime_state_names(
    internal_regime: InternalRegime,
) -> set[str]:
    """Get state names from an internal regime's variable info.

    Args:
        internal_regime: The internal regime instance.

    Returns:
        Set of state variable names.

    """
    return set(internal_regime.variable_info.query("is_state").index)


def create_regime_state_action_space(
    *,
    internal_regime: InternalRegime,
    states: MappingProxyType[str, Array],
) -> StateActionSpace:
    """Create the state-action space containing only the relevant subjects in a regime.

    Args:
        internal_regime: The internal regime instance.
        states: The current states of all subjects.

    Returns:
        The state-action space for the subjects in the regime.

    """
    relevant_state_names = internal_regime.variable_info.query("is_state").index

    states_for_state_action_space = {
        sn: states[f"{internal_regime.name}__{sn}"] for sn in relevant_state_names
    }

    return create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states=states_for_state_action_space,
    )


def calculate_next_states(
    *,
    internal_regime: InternalRegime,
    optimal_actions: MappingProxyType[str, Array],
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
    optimal_actions: MappingProxyType[str, Array],
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
    d: MappingProxyType[str, Array],
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
    regime_ids = jnp.array([regime_names_to_ids[name] for name in regime_names])

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


def build_initial_states(
    *,
    initial_states: Mapping[str, Array],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
) -> MappingProxyType[str, Array]:
    """Build flat regime-namespaced state dict from user-provided initial states.

    For each regime, copies provided states and fills missing ones with
    `jnp.nan` (continuous) or `MISSING_CAT_CODE` (discrete).

    Args:
        initial_states: Mapping of state names to arrays.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.

    Returns:
        Immutable mapping of regime-namespaced state names to arrays.
        Example: `{"work__wealth": arr, "work__health": arr, ...}`

    """
    flat: dict[str, Array] = {}
    n_subjects = len(next(iter(initial_states.values())))

    for regime_name, internal_regime in internal_regimes.items():
        for state_name in get_regime_state_names(internal_regime):
            key = f"{regime_name}__{state_name}"
            if state_name in initial_states:
                flat[key] = initial_states[state_name]
            elif isinstance(internal_regime.grids[state_name], DiscreteGrid):
                flat[key] = jnp.full(n_subjects, MISSING_CAT_CODE, dtype=jnp.int32)
            else:
                flat[key] = jnp.full(n_subjects, jnp.nan)

    return MappingProxyType(flat)
