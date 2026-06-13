"""Simulation-phase state and regime transitions.

These functions apply pre-compiled transition functions to subject arrays during
forward simulation. They are distinct from `_lcm.regime_building`, which compiles
transition functions from user-defined regimes; this module executes them.

"""

from types import MappingProxyType

import jax
from dags.tree import qname_from_tree_path
from jax import numpy as jnp
from jax import vmap

from _lcm.engine import Regime, StateActionSpace
from _lcm.simulation.random import generate_simulation_keys
from _lcm.state_action_space import _validate_all_states_present
from _lcm.typing import (
    ActionName,
    FlatRegimeParams,
    PRNGKeyND,
    RegimeName,
    RegimeNamesToIds,
    RegimeStates,
    StateName,
    StatesPerRegime,
)
from lcm.typing import Bool1D, Float1D, FloatND, Int1D, IntND, ScalarFloat, ScalarInt


def create_regime_state_action_space(
    *,
    regime: Regime,
    regime_states: RegimeStates,
    regime_params: FlatRegimeParams,
) -> StateActionSpace:
    """Create the state-action space containing only the relevant subjects in a regime.

    Continuous action grids declared with `pass_points_at_runtime=True` are
    completed from `regime_params` (via
    `regime.solution.state_action_space`).

    Args:
        regime: The internal regime instance.
        regime_states: State arrays for this regime, keyed by state name.
        regime_params: Flat regime parameters supplied at runtime, used to
            substitute runtime-supplied action gridpoints.

    Returns:
        The state-action space for the subjects in the regime.

    """
    base = regime.solution.state_action_space(regime_params=regime_params)

    states_for_state_action_space = {
        sn: regime_states[sn] for sn in regime.solution.state_names
    }
    _validate_all_states_present(
        provided_states=states_for_state_action_space,
        required_state_names=set(regime.solution.state_names),
    )

    return base.replace(states=MappingProxyType(states_for_state_action_space))


def calculate_next_states(
    *,
    regime: Regime,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    period: int,
    age: ScalarInt | ScalarFloat,
    regime_params: FlatRegimeParams,
    states_per_regime: StatesPerRegime,
    state_action_space: StateActionSpace,
    key: PRNGKeyND,
    subjects_in_regime: Bool1D,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None = None,
) -> StatesPerRegime:
    """Calculate next period states for subjects in a regime.

    Args:
        regime: The internal regime instance.
        subjects_in_regime: Boolean array indicating if subject is in regime.
        optimal_actions: Optimal actions computed for these subjects.
        period: Current period.
        age: Age corresponding to current period.
        regime_params: Flat regime parameters.
        states_per_regime: Carrier of current-period state arrays for
            every regime and state, indexed by regime name then state name.
        state_action_space: State-action space for subjects in this regime.
        key: JAX random key.
        n_subjects: Total number of subjects the dispatch sees (the full
            population, possibly padded for sharding). Keys are generated for
            the full population so each subject's draw is independent of how
            subjects are chunked.
        subject_slice: Global-index slice of the subjects in this chunk.
        original_n_subjects: Subject count before per-device padding; threaded to
            keep real subjects' draws device-count-invariant.

    Returns:
        Updated carrier with next-period state values for subjects in this regime;
        entries for other subjects are left untouched.

    """
    # Identify stochastic transitions and generate random keys
    # ---------------------------------------------------------------------------------
    stochastic_transition_names = regime.simulation.stochastic_transition_names
    # Sorted to fix a downstream-ordering bug when the nested iteration
    # yields names in a non-deterministic order.
    stochastic_next_function_names = sorted(
        qname_from_tree_path((target_regime, transition_name))
        for target_regime, target_transitions in (regime.simulation.transitions.items())
        for transition_name in target_transitions
        if transition_name in stochastic_transition_names
    )

    key, stochastic_variables_keys = generate_simulation_keys(
        key=key,
        names=stochastic_next_function_names,
        n_initial_states=n_subjects,
        subject_slice=subject_slice,
        original_n_subjects=original_n_subjects,
    )

    # Compute next states using regime's transition functions
    # ---------------------------------------------------------------------------------
    next_state_vmapped = regime.simulation.next_state

    # Carried states are true values that the decision's state-action space
    # deliberately excludes. Feed them to the realized transition so it reads
    # each carried state as a leaf — the actual carried value — rather than
    # the solve-phase imputation.
    simulate_only_states = {
        name: states_per_regime[regime.name][name]
        for name in regime.simulation.carried_grids
    }

    states_with_next_prefix = next_state_vmapped(
        **state_action_space.states,
        **simulate_only_states,
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
        states_per_regime=states_per_regime,
        next_states_per_regime=next_states_per_regime,
        subject_indices=subjects_in_regime,
    )


def calculate_next_regime_membership(
    *,
    regime: Regime,
    state_action_space: StateActionSpace,
    optimal_actions: MappingProxyType[ActionName, FloatND | IntND],
    period: int,
    age: ScalarInt | ScalarFloat,
    regime_params: FlatRegimeParams,
    regime_names_to_ids: RegimeNamesToIds,
    states_per_regime: StatesPerRegime,
    new_subject_regime_ids: Int1D,
    active_regimes_next_period: tuple[RegimeName, ...],
    key: PRNGKeyND,
    subjects_in_regime: Bool1D,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None = None,
) -> Int1D:
    """Calculate next period regime membership for subjects in a regime.

    Computes the probability distribution over regimes for the next period based on
    current states and actions, then draws random regime assignments for each subject.

    Args:
        regime: The internal regime instance.
        state_action_space: State-action space for subjects in this regime.
        optimal_actions: Optimal actions computed for these subjects.
        period: Current period.
        age: Age corresponding to current period.
        regime_params: Flat regime parameters.
        regime_names_to_ids: Mapping from regime names to integer IDs.
        states_per_regime: Carrier of current-period state arrays for every
            regime and state; supplies the carried values the realized draw
            reads.
        new_subject_regime_ids: Array to update with next regime assignments.
        active_regimes_next_period: Tuple of active regime names in the next period.
        key: JAX random key.
        subjects_in_regime: Boolean array indicating if subject is in regime.
        n_subjects: Total number of subjects the dispatch sees (the full
            population, possibly padded for sharding). Keys are generated for
            the full population so each subject's draw is independent of how
            subjects are chunked.
        subject_slice: Global-index slice of the subjects in this chunk.
        original_n_subjects: Subject count before per-device padding; threaded to
            keep real subjects' draws device-count-invariant.


    Returns:
        Updated array of regime IDs with next period assignments for subjects in this
        regime. The returned array contains regime IDs for all subjects, with updates
        only for those in the current regime.

    """
    # Compute regime transition probabilities
    # ---------------------------------------------------------------------------------
    # The realized draw is built against the published pair-free pool, so it
    # reads each carried state as the subject's true carried value — feed
    # those values like `calculate_next_states` does.
    simulate_only_states = {
        name: states_per_regime[regime.name][name]
        for name in regime.simulation.carried_grids
    }
    regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
        regime.simulation.compute_regime_transition_probs(  # ty: ignore[call-non-callable]
            **state_action_space.states,
            **simulate_only_states,
            **optimal_actions,
            period=jnp.int32(period),
            age=age,
            **regime_params,
        )
    )
    # A per-target regime transition's probs dict covers only its declared
    # targets — anything else is structurally unreachable (zero probability).
    active_regime_probs = MappingProxyType(
        {
            r: regime_transition_probs[r]
            for r in active_regimes_next_period
            if r in regime_transition_probs
        }
    )

    # Generate random keys and draw next regimes
    # ---------------------------------------------------------------------------------
    key, regime_transition_key = generate_simulation_keys(
        key=key,
        names=["regime_transition"],
        n_initial_states=n_subjects,
        subject_slice=subject_slice,
        original_n_subjects=original_n_subjects,
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
    d: MappingProxyType[RegimeName, Float1D],
    regime_names_to_ids: RegimeNamesToIds,
    keys: PRNGKeyND,
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
    # A regime whose transition reads no per-subject state or action (e.g. it
    # depends only on `age`) yields one unbatched distribution shared by
    # every subject. Broadcast it across the subjects' keys so the
    # per-subject draw below sees a probability vector per key.
    if regime_transition_probs.ndim == 1:
        regime_transition_probs = jnp.broadcast_to(
            regime_transition_probs,
            (keys.shape[0], regime_transition_probs.shape[0]),
        )
    regime_ids = jnp.asarray(
        [regime_names_to_ids[regime_name] for regime_name in regime_names],
        dtype=jnp.int32,
    )

    def random_id(
        key: PRNGKeyND,
        p: Float1D,
    ) -> ScalarInt:
        return jax.random.choice(
            key,
            regime_ids,
            p=p,
        )

    random_ids = vmap(random_id, in_axes=(0, 0))

    return random_ids(keys, regime_transition_probs)


def _advance_states_for_subjects(
    *,
    states_per_regime: StatesPerRegime,
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
        states_per_regime: Carrier of state arrays prior to this advance,
            indexed by regime name then state name.
        next_states_per_regime: Per-regime next-period state values to merge in,
            indexed by regime name then state name (no `next_` prefix — the
            caller in `calculate_next_states` strips it before invoking).
        subject_indices: Boolean mask selecting which subjects' values are
            overwritten by the next-period entries.

    Returns:
        Updated carrier with next-period values written in for selected subjects.

    """
    updated: dict[RegimeName, dict[StateName, Float1D | Int1D]] = {
        regime_name: dict(regime_states)
        for regime_name, regime_states in states_per_regime.items()
    }
    for target, target_next_states in next_states_per_regime.items():
        for state_name, next_state_values in target_next_states.items():
            current_arr = states_per_regime[target][state_name]
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
