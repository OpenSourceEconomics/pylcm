from collections.abc import Mapping
from types import MappingProxyType

from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidInitialConditionsError,
    format_messages,
)
from lcm.functools import get_union_of_args
from lcm.grids import DiscreteGrid, DiscreteMarkovGrid
from lcm.interfaces import InternalRegime
from lcm.Q_and_F import _get_feasibility
from lcm.simulation.utils import get_regime_state_names
from lcm.typing import (
    InternalParams,
    RegimeName,
)


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


def _collect_state_name_errors(
    *,
    initial_states: Mapping[str, Array],
    initial_regimes: list[RegimeName],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    valid_regime_names: set[str],
) -> list[str]:
    """Collect errors about missing or unknown state names.

    Only states from regimes that appear in `initial_regimes` are required. States
    from other regimes are accepted but not mandatory. States that don't belong to
    any regime are flagged as unknown.

    Args:
        initial_states: Mapping of state names to arrays.
        initial_regimes: List of regime names the subjects start in.
        internal_regimes: Immutable mapping of regime names to internal regime
            instances.
        valid_regime_names: Set of valid regime names.

    Returns:
        List of error message strings (empty if everything is valid).

    """
    errors: list[str] = []

    # All known states (union across all regimes) — used for the "extra" check
    all_known_states: set[str] = {"age"}
    for internal_regime in internal_regimes.values():
        all_known_states.update(get_regime_state_names(internal_regime))

    # Required states — only from regimes subjects actually start in
    required_states: set[str] = {"age"}
    used_regime_names = set(initial_regimes) & valid_regime_names
    for regime_name in used_regime_names:
        required_states.update(get_regime_state_names(internal_regimes[regime_name]))

    provided_states = set(initial_states.keys())

    missing = required_states - provided_states
    if missing:
        errors.append(_format_missing_states_message(missing, required_states))

    extra = provided_states - all_known_states
    if extra:
        errors.append(
            f"Unknown initial states: {sorted(extra)}. "
            f"Valid states are: {sorted(all_known_states)}"
        )

    return errors


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

    errors.extend(
        _collect_state_name_errors(
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            internal_regimes=internal_regimes,
            valid_regime_names=valid_regime_names,
        )
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
    else:
        # Validate that each subject's initial regime is active at their starting age.
        # Only safe to run when all ages are valid (so age_to_period lookup succeeds).
        age_to_period = {float(v): i for i, v in enumerate(ages.precise_values)}
        periods = jnp.array([age_to_period[float(a)] for a in age_values])

        active_mask = jnp.ones(len(initial_regimes), dtype=bool)
        for regime_name, internal_regime in internal_regimes.items():
            in_regime = jnp.array([r == regime_name for r in initial_regimes])
            period_active = jnp.isin(periods, jnp.array(internal_regime.active_periods))
            active_mask = active_mask & (~in_regime | period_active)

        if not jnp.all(active_mask):
            invalid_combos = {
                (initial_regimes[i], float(age_values[i]))
                for i in jnp.where(~active_mask)[0]
            }
            details = "\n".join(
                f"  regime '{name}' is not active at age {age}"
                for name, age in sorted(invalid_combos)
            )
            errors.append(
                f"Subjects are assigned to regimes that are not active "
                f"at their starting age:\n{details}"
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
        if state_name not in initial_states:
            continue
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
