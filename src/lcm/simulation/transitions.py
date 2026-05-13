"""Simulation-phase state and regime transitions.

These functions apply pre-compiled transition functions to subject arrays during
forward simulation. They are distinct from `lcm.regime_building`, which compiles
transition functions from user-defined regimes; this module executes them.

"""

from types import MappingProxyType

import jax
from dags.tree import qname_from_tree_path
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
    ScalarFloat,
    ScalarInt,
    StatesPerRegime,
)


def create_regime_state_action_space(
    *,
    internal_regime: InternalRegime,
    current_states_per_regime: StatesPerRegime,
    regime_params: FlatRegimeParams,
) -> StateActionSpace:
    """Create the state-action space containing only the relevant subjects in a regime.

    Continuous action grids declared with `pass_points_at_runtime=True` are
    completed from `regime_params` (via
    `InternalRegime.state_action_space`).

    Args:
        internal_regime: The internal regime instance.
        current_states_per_regime: Carrier of state arrays for every regime and
            state, indexed by regime name then state name.
        regime_params: Flat regime parameters supplied at runtime, used to
            substitute runtime-supplied action gridpoints.

    Returns:
        The state-action space for the subjects in the regime.

    """
    base = internal_regime.state_action_space(regime_params=regime_params)

    relevant_state_names = internal_regime.variable_info.query("is_state").index
    regime_states = current_states_per_regime[internal_regime.name]
    states_for_state_action_space = {
        sn: regime_states[sn] for sn in relevant_state_names
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
    age: ScalarInt | ScalarFloat,
    regime_params: FlatRegimeParams,
    current_states_per_regime: StatesPerRegime,
    state_action_space: StateActionSpace,
    key: Array,
    subjects_in_regime: Bool1D,
) -> StatesPerRegime:
    """Calculate next period states for subjects in a regime.

    Args:
        internal_regime: The internal regime instance.
        subjects_in_regime: Boolean array indicating if subject is in regime.
        optimal_actions: Optimal actions computed for these subjects.
        period: Current period.
        age: Age corresponding to current period.
        regime_params: Flat regime parameters.
        current_states_per_regime: Carrier of current-period state arrays for
            every regime and state, indexed by regime name then state name.
        state_action_space: State-action space for subjects in this regime.
        key: JAX random key.

    Returns:
        Updated carrier with next-period state values for subjects in this regime;
        entries for other subjects are left untouched.

    """
    # Identify stochastic transitions and generate random keys
    # ---------------------------------------------------------------------------------
    stochastic_transition_names = (
        internal_regime.simulate_functions.stochastic_transition_names
    )
    # Sorted to fix a downstream-ordering bug when the nested iteration
    # yields names in a non-deterministic order.
    stochastic_next_function_names = sorted(
        qname_from_tree_path((target_regime, transition_name))
        for target_regime, target_transitions in (
            internal_regime.simulate_functions.transitions.items()
        )
        for transition_name in target_transitions
        if transition_name in stochastic_transition_names
    )

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
        period=jnp.int32(period),
        age=age,
        **regime_params,
    )

    # Transition functions are DAG-named `next_<state>` to distinguish them from
    # the current-period state values they consume. Strip the prefix here so
    # `_advance_states_for_subjects` sees keys that line up with the carrier's
    # `StateName` keys directly.
    next_states_per_regime = MappingProxyType(
        {
            target: MappingProxyType(
                {
                    name.removeprefix("next_"): value
                    for name, value in target_next_states.items()
                }
            )
            for target, target_next_states in states_with_next_prefix.items()
        }
    )

    return _advance_states_for_subjects(
        current_states_per_regime=current_states_per_regime,
        next_states_per_regime=next_states_per_regime,
        subject_indices=subjects_in_regime,
    )


def calculate_next_regime_membership(
    *,
    internal_regime: InternalRegime,
    state_action_space: StateActionSpace,
    optimal_actions: MappingProxyType[ActionName, Array],
    period: int,
    age: ScalarInt | ScalarFloat,
    regime_params: FlatRegimeParams,
    regime_names_to_ids: RegimeNamesToIds,
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
            period=jnp.int32(period),
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
    regime_ids = jnp.asarray(
        [regime_names_to_ids[regime_name] for regime_name in regime_names],
        dtype=jnp.int32,
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


def _advance_states_for_subjects(
    *,
    current_states_per_regime: StatesPerRegime,
    next_states_per_regime: StatesPerRegime,
    subject_indices: Bool1D,
) -> StatesPerRegime:
    """Merge next-period state values into the carrier for selected subjects.

    The carrier and the per-regime update share the `StatesPerRegime` shape: an
    outer mapping by regime name, an inner mapping by state name. Where
    `subject_indices` is true, the corresponding slot is replaced with the
    matching entry from `next_states_per_regime`; otherwise the carrier value
    is left untouched. Regimes that don't appear in `next_states_per_regime`
    are passed through unchanged.

    Args:
        current_states_per_regime: Carrier of state arrays prior to this advance,
            indexed by regime name then state name.
        next_states_per_regime: Per-regime next-period state values to merge in,
            indexed by regime name then state name (no `next_` prefix — the
            caller in `calculate_next_states` strips it before invoking).
        subject_indices: Boolean mask selecting which subjects' values are
            overwritten by the next-period entries.

    Returns:
        Updated carrier with next-period values written in for selected subjects.

    """
    updated: dict[RegimeName, dict[str, Array]] = {
        regime_name: dict(regime_states)
        for regime_name, regime_states in current_states_per_regime.items()
    }
    for target, target_next_states in next_states_per_regime.items():
        for state_name, next_state_values in target_next_states.items():
            current_arr = current_states_per_regime[target][state_name]
            target_dtype = current_arr.dtype
            # Preserve storage dtype only when the transition output is the
            # same numeric kind. Across kinds (e.g. int storage + float
            # transition output) leave JAX's promotion in place; the
            # cross-kind boundary cast belongs to Package B.
            new_values = (
                next_state_values.astype(target_dtype)
                if next_state_values.dtype.kind == target_dtype.kind
                else next_state_values
            )
            updated[target][state_name] = jnp.where(
                subject_indices,
                new_values,
                current_arr,
            )

    return MappingProxyType(
        {
            regime_name: MappingProxyType(regime_states)
            for regime_name, regime_states in updated.items()
        }
    )
