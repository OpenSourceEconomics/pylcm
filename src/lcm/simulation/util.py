from collections.abc import Mapping
from types import MappingProxyType

import jax
from dags.tree import QNAME_DELIMITER
from jax import Array, vmap
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidRegimeTransitionProbabilitiesError,
    format_messages,
)
from lcm.functools import get_union_of_args
from lcm.grids import DiscreteGrid, DiscreteMarkovGrid
from lcm.interfaces import InternalRegime, StateActionSpace
from lcm.Q_and_F import _get_feasibility
from lcm.random import generate_simulation_keys
from lcm.state_action_space import create_state_action_space
from lcm.typing import (
    Bool1D,
    FlatRegimeParams,
    Int1D,
    InternalParams,
    RegimeName,
    RegimeNamesToIds,
)
from lcm.utils import flatten_regime_namespace, normalize_regime_transition_probs


def create_regime_state_action_space(
    *,
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

    relevant_state_names = internal_regime.variable_info.query(query).index

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
        internal_regime.internal_functions.stochastic_transition_names
    )
    stochastic_next_function_names = [
        next_func_name
        for next_func_name in flatten_regime_namespace(internal_regime.transitions)
        if next_func_name.split(QNAME_DELIMITER)[-1] in stochastic_transition_names
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
    next_state_vmapped = internal_regime.next_state_simulation_function

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
        internal_regime.internal_functions.regime_transition_probs.simulate(  # ty: ignore[unresolved-attribute]
            **state_action_space.states,
            **optimal_actions,
            period=period,
            age=age,
            **regime_params,
        )
    )
    normalized_regime_transition_probs = normalize_regime_transition_probs(
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
    )

    _validate_normalized_regime_transition_probs(
        normalized_regime_transition_probs=normalized_regime_transition_probs,
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
        # State names may be prefixed with regime (e.g., "working__next_wealth")
        # We need to replace "next_" with "" to get "working__wealth"
        state_name = next_state_name.replace("next_", "")
        updated_states[state_name] = jnp.where(
            subject_indices,
            next_state_values,
            all_states[state_name],
        )

    return MappingProxyType(updated_states)


def validate_initial_conditions(
    *,
    initial_states: Mapping[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
) -> None:
    """Validate initial conditions (regimes, states, and feasibility).

    Checks that:
    1. initial_regimes is non-empty and contains only valid regime names
    2. All required state names (across all regimes) are provided, with no extras
    3. All state arrays have the same length
    4. Discrete state values are valid codes
    5. Each subject has at least one feasible action combination

    Args:
        initial_states: Mapping of state names to arrays.
        initial_regimes: List of regime names the subjects start in.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.

    Raises:
        InvalidInitialConditionsError: If any validation check fails.

    """
    # Validate regime names and state names/shapes first; early-exit on errors so that
    # downstream checks (discrete codes, feasibility) can assume correct names.
    structural_errors = _collect_structural_errors(
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        internal_regimes=internal_regimes,
        ages=ages,
    )
    if structural_errors:
        raise InvalidInitialConditionsError(format_messages(structural_errors))

    # Validate discrete state values
    _validate_discrete_state_values(
        initial_states=initial_states, internal_regimes=internal_regimes
    )

    # Validate feasibility
    feasibility_errors = _collect_feasibility_errors(
        initial_states=initial_states,
        initial_regimes=initial_regimes,
        internal_regimes=internal_regimes,
        internal_params=internal_params,
        ages=ages,
    )
    if feasibility_errors:
        raise InvalidInitialConditionsError(format_messages(feasibility_errors))


def _format_missing_states_message(missing: set[str], required: set[str]) -> str:
    """Format an error message for missing initial states.

    Provides a specific hint when 'age' is missing, since users often omit it.

    Args:
        missing: Set of missing state names.
        required: Set of all required state names.

    Returns:
        A formatted error message string.

    """
    parts: list[str] = []
    if "age" in missing:
        parts.append(
            "'age' must be provided in initial_states so the validation "
            "knows each subject's starting age. Example: "
            "initial_states={'age': jnp.array([25.0, 25.0]), ...}"
        )
    missing_model_states = sorted(missing - {"age"})
    if missing_model_states:
        parts.append(f"Missing model states: {missing_model_states}.")
    parts.append(f"Required initial states are: {sorted(required)}")
    return " ".join(parts)


def _collect_structural_errors(
    *,
    initial_states: Mapping[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    ages: AgeGrid,
) -> list[str]:
    """Collect errors about regime names, state names, age values, and array shapes.

    Args:
        initial_states: Mapping of state names to arrays.
        initial_regimes: List of regime names the subjects start in.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        ages: AgeGrid for the model.

    Returns:
        List of error message strings (empty if everything is valid).

    """
    errors: list[str] = []

    # Validate initial regimes
    if not initial_regimes:
        errors.append("initial_regimes must not be empty.")

    valid_regime_names = set(internal_regimes.keys())
    invalid_names = sorted({r for r in initial_regimes if r not in valid_regime_names})
    if invalid_names:
        errors.append(
            f"Invalid regime names {invalid_names} in initial_regimes. "
            f"Valid regime names are: {sorted(valid_regime_names)}"
        )

    # Validate initial states — "age" is always required alongside regime states
    required_states: set[str] = {"age"}
    for internal_regime in internal_regimes.values():
        regime_states = set(internal_regime.variable_info.query("is_state").index)
        required_states.update(regime_states)

    provided_states = set(initial_states.keys())

    missing = required_states - provided_states
    if missing:
        errors.append(_format_missing_states_message(missing, required_states))

    extra = provided_states - required_states
    if extra:
        errors.append(
            f"Unknown initial states: {sorted(extra)}. "
            f"Valid states are: {sorted(required_states)}"
        )

    if initial_states:
        lengths = {name: len(arr) for name, arr in initial_states.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            errors.append(
                f"All initial state arrays must have the same length. "
                f"Got lengths: {lengths}"
            )

    # Early exit before value-level checks if names/shapes are wrong
    if errors:
        return errors

    # Validate that all age values are representable on the age grid.  Compare
    # against float64 conversions of AgeGrid.precise_values to avoid float32
    # precision issues with sub-annual steps.
    valid_ages = {float(v) for v in ages.precise_values}
    age_values = initial_states["age"]
    invalid_ages = sorted({float(a) for a in age_values if float(a) not in valid_ages})
    if invalid_ages:
        errors.append(
            f"Invalid age values {invalid_ages} in initial_states. "
            f"Valid ages are: {sorted(valid_ages)}"
        )

    return errors


def _collect_feasibility_errors(
    *,
    initial_states: Mapping[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
) -> list[str]:
    """Collect errors about action feasibility for each subject.

    Args:
        initial_states: Mapping of state names to arrays.
        initial_regimes: List of regime names the subjects start in.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.

    Returns:
        List of error message strings (empty if everything is feasible).

    """
    age_to_period = {float(v): i for i, v in enumerate(ages.precise_values)}

    errors: list[str] = []
    for regime_name, internal_regime in internal_regimes.items():
        subject_indices = [i for i, r in enumerate(initial_regimes) if r == regime_name]
        if not subject_indices:
            continue

        regime_params = {
            **internal_regime.resolved_fixed_params,
            **dict(internal_params.get(regime_name, MappingProxyType({}))),
        }

        msg = _check_regime_feasibility(
            internal_regime=internal_regime,
            regime_name=regime_name,
            initial_states=initial_states,
            subject_indices=subject_indices,
            regime_params=regime_params,
            age_to_period=age_to_period,
        )
        if msg is not None:
            errors.append(msg)

    return errors


def _validate_discrete_state_values(
    *,
    initial_states: Mapping[str, Array],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
) -> None:
    """Validate that discrete state values are valid codes.

    Args:
        initial_states: Mapping of state names to arrays.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.

    Raises:
        InvalidInitialConditionsError: If any discrete state contains invalid codes.

    """
    discrete_valid_codes: dict[str, set[int]] = {}
    for internal_regime in internal_regimes.values():
        for state_name in internal_regime.variable_info.query(
            "is_state and is_discrete"
        ).index:
            gridspec = internal_regime.gridspecs[state_name]
            if isinstance(gridspec, DiscreteGrid | DiscreteMarkovGrid):
                discrete_valid_codes[state_name] = set(gridspec.codes)

    for state_name, valid_codes in discrete_valid_codes.items():
        values = initial_states[state_name]
        invalid_mask = jnp.isin(values, jnp.array(sorted(valid_codes)), invert=True)
        if jnp.any(invalid_mask):
            invalid_vals = sorted({int(v) for v in values[invalid_mask]})
            raise InvalidInitialConditionsError(
                f"Invalid values {invalid_vals} for discrete state "
                f"'{state_name}'. Valid codes are: {sorted(valid_codes)}"
            )


def _check_regime_feasibility(
    *,
    internal_regime: InternalRegime,
    regime_name: str,
    initial_states: Mapping[str, Array],
    subject_indices: list[int],
    regime_params: Mapping[str, object],
    age_to_period: dict[float, int],
) -> str | None:
    """Check whether all subjects in a regime have at least one feasible action.

    Args:
        internal_regime: The internal regime instance.
        regime_name: Name of the regime.
        initial_states: Mapping of state names to arrays (includes "age").
        subject_indices: Indices of subjects starting in this regime.
        regime_params: Merged fixed and runtime parameters for this regime.
        age_to_period: Mapping from float age values to period indices.

    Returns:
        An error message string if any subjects are infeasible, or None.

    """
    feasibility_func = _get_feasibility(internal_regime.internal_functions)
    accepted = get_union_of_args([feasibility_func])

    action_names = list(internal_regime.variable_info.query("is_action").index)
    if not action_names:
        return None

    flat_actions = _build_flat_action_grid(
        action_names=action_names, grids=internal_regime.grids
    )

    filtered_params = {k: v for k, v in regime_params.items() if k in accepted}
    state_names = list(internal_regime.variable_info.query("is_state").index)
    needs_age = "age" in accepted
    needs_period = "period" in accepted

    infeasible_indices: list[int] = []
    for idx in subject_indices:
        kwargs: dict[str, Array | float | int] = {}
        for sn in state_names:
            if sn in accepted:
                kwargs[sn] = initial_states[sn][idx]
        kwargs.update({k: v for k, v in flat_actions.items() if k in accepted})
        kwargs.update(filtered_params)  # ty: ignore[no-matching-overload]

        subject_age = float(initial_states["age"][idx])
        if needs_age:
            kwargs["age"] = subject_age
        if needs_period:
            kwargs["period"] = age_to_period[subject_age]

        result = feasibility_func(**kwargs)
        if not jnp.any(result):
            infeasible_indices.append(idx)

    if not infeasible_indices:
        return None

    state_values = {
        name: [float(initial_states[name][i]) for i in infeasible_indices]
        for name in state_names
        if name in initial_states
    }
    return (
        f"All actions are infeasible for subject(s) at indices "
        f"{infeasible_indices} in regime '{regime_name}'. "
        f"State values: {state_values}. No action combination satisfies "
        f"the model's constraints for these initial states."
    )


def _build_flat_action_grid(
    *,
    action_names: list[str],
    grids: MappingProxyType[str, Array],
) -> dict[str, Array]:
    """Build a flat array of all action combinations from action grids.

    Args:
        action_names: List of action variable names.
        grids: Immutable mapping of variable names to grid arrays.

    Returns:
        Mapping of action names to flat arrays covering all combinations.

    """
    action_grids = [grids[name] for name in action_names]
    if len(action_grids) > 1:
        mesh = jnp.meshgrid(*action_grids, indexing="ij")
        return {name: m.ravel() for name, m in zip(action_names, mesh, strict=True)}
    return {action_names[0]: action_grids[0]}


def convert_initial_states_to_nested(
    *,
    initial_states: Mapping[str, Array],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
) -> dict[RegimeName, dict[str, Array]]:
    """Convert flat initial_states dict to nested format.

    Takes user-provided flat format and converts to the nested format
    expected by internal simulation code.

    Args:
        initial_states: Mapping of state names to arrays.
            Example: {"wealth": arr, "health": arr}
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.

    Returns:
        Nested dict mapping regime names to state dicts.
            Example: {"work": {"wealth": arr, "health": arr}, ...}

    """
    nested: dict[RegimeName, dict[str, Array]] = {}

    for regime_name, internal_regime in internal_regimes.items():
        regime_state_names = set(internal_regime.variable_info.query("is_state").index)
        nested[regime_name] = {
            state_name: initial_states[state_name] for state_name in regime_state_names
        }

    return nested


def _validate_normalized_regime_transition_probs(
    *,
    normalized_regime_transition_probs: MappingProxyType[str, Array],
    regime_name: str,
    period: int,
) -> None:
    probs = jnp.array(list(normalized_regime_transition_probs.values()))
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
