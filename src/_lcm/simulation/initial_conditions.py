"""Build and validate initial conditions for simulation.

Consolidates initial condition construction (`build_initial_states`) and validation
(`validate_initial_conditions`) into a single module.

"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import NoReturn, cast

import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp

from _lcm.dtypes import (
    canonical_float_dtype,
    safe_to_float_dtype,
    safe_to_int_dtype,
)
from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.grids import DiscreteGrid
from _lcm.regime_building.Q_and_F import _get_feasibility
from _lcm.typing import (
    ActionName,
    FlatParams,
    FlatRegimeParams,
    InitialConditions,
    RegimeIdsToNames,
    RegimeName,
    RegimeNamesToIds,
    StateName,
    StatesPerRegime,
)
from _lcm.utils.containers import invert_regime_ids
from _lcm.utils.error_messages import format_messages
from _lcm.utils.functools import get_union_of_args
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidInitialConditionsError, PyLCMError
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, UserInitialConditions

# Sentinel for categorical states not in initial conditions.  Using int32 min
# instead of -1 so that JAX indexing produces obviously wrong values rather than
# silently returning the last element.
MISSING_CAT_CODE = jnp.iinfo(jnp.int32).min

# Names that behave like states in initial conditions but are not declared on
# any `Regime.states`. `age` is required for every subject regardless of regime.
PSEUDO_STATE_NAMES: frozenset[str] = frozenset({"age"})


def canonicalize_initial_conditions(
    *,
    initial_conditions: UserInitialConditions,
    regimes: MappingProxyType[RegimeName, Regime],
) -> InitialConditions:
    """Cast every initial-conditions array to its canonical pylcm dtype.

    This is pylcm's simulation input boundary: `"regime_id"` and discrete
    states cast to `int32`; `"age"` and continuous states cast to the canonical
    float dtype. Keys that match no model state are cast by their array kind
    (integer arrays to `int32`, otherwise to canonical float) and left for
    `validate_initial_conditions` to report. Downstream validation and the
    simulate stack receive canonical-dtype arrays and do not re-cast.

    Args:
        initial_conditions: Mapping of state names (plus `"regime_id"`) to
            user-supplied arrays of any integer or floating dtype.
        regimes: Immutable mapping of regime names to internal regime
            instances, used to classify each state as discrete or continuous.

    Returns:
        Mapping of the same keys to JAX arrays at their canonical dtype.

    """
    discrete_state_names = {
        state_name
        for regime in regimes.values()
        for state_name, grid in regime.simulation.grids.items()
        if isinstance(grid, DiscreteGrid)
    }
    known_state_names = {
        state_name
        for regime in regimes.values()
        for state_name in regime.simulation.grids
    }
    canonical: dict[str, FloatND | IntND] = {}
    for name, value in initial_conditions.items():
        if name == "regime_id" or name in discrete_state_names:
            canonical[name] = safe_to_int_dtype(value, name=name)
        elif name == "age" or name in known_state_names:
            canonical[name] = safe_to_float_dtype(value, name=name)
        elif np.asarray(value).dtype.kind in "iu":
            canonical[name] = safe_to_int_dtype(value, name=name)
        else:
            canonical[name] = safe_to_float_dtype(value, name=name)
    return MappingProxyType(canonical)


def build_initial_states(
    *,
    initial_states: Mapping[StateName, Float1D | Int1D],
    regimes: MappingProxyType[RegimeName, Regime],
) -> StatesPerRegime:
    """Build the regime-keyed state carrier from user-provided initial states.

    For each regime, copies provided states and fills missing ones with
    `jnp.nan` (continuous) or `MISSING_CAT_CODE` (discrete). If a state has been
    declared as distributed, the initial states will also be distributed over the
    available devices.

    Args:
        initial_states: Mapping of state names to arrays.
        regimes: Immutable mapping of regime names to internal regime
            instances.

    Returns:
        Nested immutable mapping `{regime_name: {state_name: array}}`.

    """
    n_subjects = len(next(iter(initial_states.values())))
    states_per_regime: dict[
        RegimeName, MappingProxyType[StateName, Float1D | Int1D]
    ] = {}

    sharding = subject_array_sharding(regimes=regimes, n_subjects=n_subjects)
    for regime_name, regime in regimes.items():
        regime_states: dict[StateName, Float1D | Int1D] = {}
        for state_name in regime.simulation.state_names:
            grid = regime.simulation.grids[state_name]
            if isinstance(grid, DiscreteGrid):
                # Cast user-supplied discrete states to the grid's index
                # dtype so every period's argmax sees a single signature
                # for that state.
                target_dtype = grid.to_jax().dtype
                if state_name in initial_states:
                    regime_states[state_name] = initial_states[state_name].astype(
                        target_dtype
                    )
                else:
                    regime_states[state_name] = jnp.full(
                        n_subjects, MISSING_CAT_CODE, dtype=target_dtype
                    )
            elif state_name in initial_states:
                # Cast user-supplied continuous states to the canonical float
                # dtype so the simulate state pool has one signature across
                # periods regardless of the user-supplied dtype.
                regime_states[state_name] = safe_to_float_dtype(
                    initial_states[state_name], name=f"initial_states.{state_name}"
                )
            else:
                regime_states[state_name] = jnp.full(
                    n_subjects, jnp.nan, dtype=canonical_float_dtype()
                )
            if sharding is not None:
                regime_states[state_name] = jax.device_put(
                    regime_states[state_name], device=sharding
                )
        states_per_regime[regime_name] = MappingProxyType(regime_states)

    return MappingProxyType(states_per_regime)


def pad_initial_conditions_to_multiple(
    *,
    initial_conditions: InitialConditions,
    multiple: int,
) -> tuple[InitialConditions, int]:
    """Pad `initial_conditions`' leading axis up to the next multiple of `multiple`.

    Two simulate paths need the subject axis aligned to a fixed block size:
    distributed grids shard it across the visible devices (the shard must divide
    the axis evenly), and single-device chunking runs fixed-size passes (every
    chunk must match the AOT-compiled shape). Both reduce to padding the leading
    axis up to the next multiple of one number — the device count or the chunk
    size — adding at most `multiple - 1` rows. Callers pick any `n_subjects` and
    pylcm aligns it internally, dropping the pad rows on the way out.

    Pad rows duplicate the last real subject — they pass validation automatically
    (the last real row already did) and produce identical simulate-side outputs
    in their assigned regime, which are then trimmed in `simulate` before
    constructing `SimulationResult`.

    Args:
        initial_conditions: Canonicalized initial conditions (state arrays keyed
            by name, plus `regime_id`).
        multiple: Block size the leading axis is padded up to a multiple of. A
            `multiple` of `1` (single-device, single pass) — or any value that
            already divides the count — is a no-op.

    Returns:
        Tuple of `(padded_initial_conditions, original_n_subjects)`. Returns the
        original mapping unchanged and the existing length when `multiple` already
        divides the count.

    """
    original_n_subjects = len(next(iter(initial_conditions.values())))
    if multiple <= 1 or original_n_subjects % multiple == 0:
        return initial_conditions, original_n_subjects
    pad = multiple - (original_n_subjects % multiple)
    padded: dict[str, jax.Array] = {}
    for name, arr in initial_conditions.items():
        # Duplicate the last subject's row `pad` times along the leading axis.
        pad_block = jnp.repeat(arr[-1:], pad, axis=0)
        padded[name] = jnp.concatenate([arr, pad_block], axis=0)
    return cast("InitialConditions", MappingProxyType(padded)), original_n_subjects


def trim_pad_from_raw_results(
    *,
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    original_n_subjects: int,
) -> MappingProxyType[RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]]:
    """Slice every per-subject array in `raw_results` back to `original_n_subjects`.

    The simulate dispatch runs against a padded leading axis (see
    `pad_initial_conditions_to_multiple`); this drops the trailing pad rows so
    `SimulationResult` and downstream consumers see only the user's real subjects.
    No-op for any period whose leading-axis length already equals
    `original_n_subjects`.

    Args:
        raw_results: Immutable nested mapping `regime_name -> period ->
            PeriodRegimeSimulationData` produced by the simulate loop.
        original_n_subjects: Subject count from the user's `initial_conditions`
            before any pylcm-internal padding.

    Returns:
        New immutable mapping with the trailing pad rows removed from every
        per-subject array (`V_arr`, `actions`, `states`, `in_regime`).

    """
    trimmed: dict[RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]] = {}
    for regime_name, periods in raw_results.items():
        new_periods: dict[int, PeriodRegimeSimulationData] = {}
        for period, data in periods.items():
            if data.V_arr.shape[0] == original_n_subjects:
                new_periods[period] = data
                continue
            new_periods[period] = dataclasses.replace(
                data,
                V_arr=data.V_arr[:original_n_subjects],
                actions=MappingProxyType(
                    {k: v[:original_n_subjects] for k, v in data.actions.items()}
                ),
                states=MappingProxyType(
                    {k: v[:original_n_subjects] for k, v in data.states.items()}
                ),
                in_regime=data.in_regime[:original_n_subjects],
            )
        trimmed[regime_name] = MappingProxyType(new_periods)
    return MappingProxyType(trimmed)


def subject_array_sharding(
    *, regimes: MappingProxyType[RegimeName, Regime], n_subjects: int
) -> jax.NamedSharding | None:
    """Return the model-wide device sharding for per-subject simulation arrays.

    Subjects propagate across regime transitions inside the simulate loop, so
    every regime's per-subject arrays must carry the same device sharding —
    otherwise an AOT-compiled program lowered with one regime's sharding rejects
    the inputs it receives from another. When any grid in any regime is
    distributed, the `n_subjects` subjects are scattered across all available
    devices along a single mesh axis. When no grid is distributed the arrays
    stay on the default device.

    Args:
        regimes: Immutable mapping of regime names to internal regime instances.
        n_subjects: Number of simulated subjects (per simulate dispatch — the
            chunk size when subject-batching).

    Returns:
        The `NamedSharding` over the device mesh, or `None` when no grid in any
        regime is distributed.

    """
    distributes_any = any(
        grid.distributed
        for regime in regimes.values()
        for grid in regime.solution.grids.values()
    )
    if not distributes_any:
        return None
    devices = jax.devices()
    if n_subjects % len(devices) != 0:
        # Defensive: with distributed grids the dispatch runs one pass over the
        # device-padded population, so this divides evenly. `Model.simulate`
        # rejects subject_batch_size > 0 under multi-device distribution before
        # reaching here, so a non-multiple count signals direct/internal misuse.
        raise PyLCMError(
            "When using distributed grids, the number of subjects per simulate "
            "dispatch must be a multiple of the available devices. "
            f"Subjects: {n_subjects} Available Devices: {len(devices)}"
        )
    mesh = jax.make_mesh(
        (len(devices),), ("X"), (jax.sharding.AxisType.Auto,), devices=devices
    )
    return jax.NamedSharding(mesh=mesh, spec=jax.P("X"))


def validate_initial_conditions(
    *,
    initial_conditions: InitialConditions,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    ages: AgeGrid,
) -> None:
    """Validate initial conditions (regimes, states, and feasibility).

    Checks that:
    1. `"regime_id"` is present, non-empty, and contains only valid regime IDs
    2. All required state names (across all regimes) are provided, with no extras
    3. All arrays have the same length
    4. Discrete state values are valid codes
    5. Each subject has at least one feasible action combination

    Args:
        initial_conditions: Mapping of state names (plus `"regime_id"`) to arrays.
        regimes: Immutable mapping of regime names to internal regime
            instances.
        regime_names_to_ids: Immutable mapping of regime names to integer IDs.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.

    Raises:
        InvalidInitialConditionsError: If any validation check fails.

    """
    # Build reverse lookup from regime IDs to names. `regime_names_to_ids`
    # values are `ScalarInt` (jax 0-d arrays), which can't serve as dict
    # keys directly; `invert_regime_ids` coerces them to Python `int`.
    regime_ids_to_names = invert_regime_ids(regime_names_to_ids)

    # Extract regime array
    regime_arr = initial_conditions.get("regime_id")
    if regime_arr is None:
        raise InvalidInitialConditionsError(
            format_messages(["'regime_id' must be provided in initial_conditions."])
        )

    # Vectorized regime ID validity check
    valid_ids_arr = jnp.array(sorted(regime_ids_to_names.keys()))
    invalid_mask = ~jnp.isin(regime_arr, valid_ids_arr)
    if jnp.any(invalid_mask):
        invalid_ids = sorted({int(i) for i in jnp.unique(regime_arr[invalid_mask])})
        raise InvalidInitialConditionsError(
            format_messages(
                [
                    f"Invalid regime IDs {invalid_ids}. "
                    f"Valid IDs: {sorted(regime_ids_to_names.keys())}"
                ]
            )
        )

    initial_states = {k: v for k, v in initial_conditions.items() if k != "regime_id"}

    # Validate regime names and state names/shapes first; early-exit on errors so that
    # downstream checks (discrete codes, feasibility) can assume correct names.
    structural_errors = _collect_structural_errors(
        initial_states=initial_states,
        regime_id_arr=regime_arr,
        regime_ids_to_names=regime_ids_to_names,
        regime_names_to_ids=regime_names_to_ids,
        regimes=regimes,
        ages=ages,
    )
    if structural_errors:
        raise InvalidInitialConditionsError(format_messages(structural_errors))

    # Validate discrete state values
    _validate_discrete_state_values(
        initial_states=initial_states,
        regimes=regimes,
        regime_id_arr=regime_arr,
        regime_names_to_ids=regime_names_to_ids,
    )

    # Validate feasibility
    feasibility_errors = _collect_feasibility_errors(
        initial_states=initial_states,
        regime_id_arr=regime_arr,
        regime_names_to_ids=regime_names_to_ids,
        regimes=regimes,
        flat_params=flat_params,
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
    missing_model_states = sorted(missing - PSEUDO_STATE_NAMES)
    if missing_model_states:
        parts.append(f"Missing model states: {missing_model_states}.")
    parts.append(f"Required initial states are: {sorted(required)}")
    return " ".join(parts)


def _collect_state_name_errors(
    *,
    initial_states: Mapping[StateName, FloatND | IntND],
    regime_id_arr: Int1D,
    regime_ids_to_names: RegimeIdsToNames,
    regimes: MappingProxyType[RegimeName, Regime],
    valid_regime_names: set[RegimeName],
) -> list[str]:
    """Collect errors about missing or unknown state names.

    Only states from regimes that appear in `regime_id_arr` are required. States
    from other regimes are accepted but not mandatory. States that don't belong to
    any regime are flagged as unknown.

    Args:
        initial_states: Mapping of state names to arrays.
        regime_id_arr: Array of integer regime IDs.
        regime_ids_to_names: Immutable mapping of regime integer IDs to regime names.
        regimes: Immutable mapping of regime names to internal regime
            instances.
        valid_regime_names: Set of valid regime names.

    Returns:
        List of error message strings (empty if everything is valid).

    """
    errors: list[str] = []

    # All known states (union across all regimes) — used for the "extra" check
    all_known_states: set[str] = set(PSEUDO_STATE_NAMES)
    for regime in regimes.values():
        all_known_states.update(regime.simulation.state_names)

    # Required states — only from regimes subjects actually start in
    required_states: set[str] = set(PSEUDO_STATE_NAMES)
    used_ids = jnp.unique(regime_id_arr)
    used_regime_names = {
        regime_ids_to_names[int(i)] for i in used_ids if int(i) in regime_ids_to_names
    } & valid_regime_names
    for regime_name in used_regime_names:
        required_states.update(regimes[regime_name].simulation.state_names)

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
    initial_states: Mapping[StateName, FloatND | IntND],
    regime_id_arr: Int1D,
    regime_ids_to_names: RegimeIdsToNames,
    regime_names_to_ids: RegimeNamesToIds,
    regimes: MappingProxyType[RegimeName, Regime],
    ages: AgeGrid,
) -> list[str]:
    """Collect errors about regime names, state names, age values, and array shapes.

    Args:
        initial_states: Mapping of state names to arrays.
        regime_id_arr: Array of integer regime IDs.
        regime_ids_to_names: Immutable mapping of regime integer IDs to regime names.
        regime_names_to_ids: Immutable mapping of regime names to integer IDs.
        regimes: Immutable mapping of regime names to internal regime
            instances.
        ages: AgeGrid for the model.

    Returns:
        List of error message strings (empty if everything is valid).

    """
    errors: list[str] = []

    if regime_id_arr.size == 0:
        errors.append("initial_regimes must not be empty.")

    valid_regime_names = set(regimes.keys())

    errors.extend(
        _collect_state_name_errors(
            initial_states=initial_states,
            regime_id_arr=regime_id_arr,
            regime_ids_to_names=regime_ids_to_names,
            regimes=regimes,
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
    # against float64 conversions of AgeGrid.exact_values to avoid float32
    # precision issues with sub-annual steps.
    valid_ages = {float(v) for v in ages.exact_values}
    age_values = initial_states["age"]
    valid_ages_arr = jnp.array(sorted(valid_ages))
    age_invalid_mask = ~jnp.isin(age_values, valid_ages_arr)
    invalid_ages = (
        sorted(set(np.asarray(age_values[age_invalid_mask]).tolist()))
        if jnp.any(age_invalid_mask)
        else []
    )
    if invalid_ages:
        errors.append(
            f"Invalid age values {invalid_ages} in initial_states. "
            f"Valid ages are: {sorted(valid_ages)}"
        )
    else:
        # Validate that each subject's initial regime is active at their starting age.
        # Only safe to run when all ages are valid (so age_to_period lookup succeeds).
        periods = jnp.array(
            [ages.age_to_period(a.item()) for a in age_values], dtype=jnp.int32
        )

        active_mask = jnp.ones(regime_id_arr.size, dtype=bool)
        for regime_name, regime in regimes.items():
            in_regime = regime_id_arr == regime_names_to_ids[regime_name]
            period_active = jnp.isin(
                periods,
                jnp.array(regime.active_periods, dtype=jnp.int32),
            )
            active_mask = active_mask & (~in_regime | period_active)

        if not jnp.all(active_mask):
            invalid_indices = jnp.where(~active_mask)[0].astype(jnp.int32)
            invalid_combos = {
                (regime_ids_to_names[int(regime_id_arr[i])], float(age_values[i]))
                for i in invalid_indices
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
    initial_states: Mapping[StateName, FloatND | IntND],
    regime_id_arr: Int1D,
    regime_names_to_ids: RegimeNamesToIds,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
) -> list[str]:
    """Collect errors about action feasibility for each subject.

    Args:
        initial_states: Mapping of state names to arrays.
        regime_id_arr: Array of integer regime IDs.
        regime_names_to_ids: Immutable mapping of regime names to integer IDs.
        regimes: Immutable mapping of regime names to internal regime
            instances.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: AgeGrid for the model.

    Returns:
        List of error message strings (empty if everything is feasible).

    """
    errors: list[str] = []
    for regime_name, regime in regimes.items():
        regime_id = regime_names_to_ids[regime_name]
        idx_arr = jnp.where(regime_id_arr == regime_id)[0].astype(jnp.int32)
        subject_indices = idx_arr.tolist() if idx_arr.size > 0 else []
        if not subject_indices:
            continue

        regime_params = {
            **regime.resolved_fixed_params,
            **dict(flat_params.get(regime_name, MappingProxyType({}))),
        }

        msg = _check_regime_feasibility(
            regime=regime,
            regime_name=regime_name,
            initial_states=initial_states,
            subject_indices=subject_indices,
            regime_params=regime_params,
            ages=ages,
        )
        if msg is not None:
            errors.append(msg)

    return errors


def _validate_discrete_state_values(
    *,
    initial_states: Mapping[StateName, FloatND | IntND],
    regimes: MappingProxyType[RegimeName, Regime],
    regime_id_arr: Int1D,
    regime_names_to_ids: RegimeNamesToIds,
) -> None:
    """Validate that discrete state values are valid codes.

    Only check subjects in regimes that actually have the state.

    Args:
        initial_states: Mapping of state names to arrays.
        regimes: Immutable mapping of regime names to internal regime
            instances.
        regime_id_arr: Array of regime IDs for each subject.
        regime_names_to_ids: Mapping from regime names to integer IDs.

    Raises:
        InvalidInitialConditionsError: If any discrete state contains invalid codes.

    """
    # Build per-state: valid codes + regime IDs that have this state.
    # `regime_id` is `ScalarInt` (a 0-d jax array); coerce to Python `int`
    # before set insertion.
    discrete_info: dict[str, tuple[set[int], set[int]]] = {}
    for regime_name, regime in regimes.items():
        regime_id = int(regime_names_to_ids[regime_name])
        for state_name in regime.simulation.variables.discrete_state_names:
            grid = regime.simulation.grids[state_name]
            if isinstance(grid, DiscreteGrid):
                codes, regime_ids = discrete_info.get(state_name, (set(), set()))
                discrete_info[state_name] = (
                    codes | set(grid.codes),
                    regime_ids | {regime_id},
                )

    for state_name, (valid_codes, regime_ids) in discrete_info.items():
        if state_name not in initial_states:
            continue
        values = initial_states[state_name]
        # Only validate subjects in regimes that have this state
        in_relevant_regime = jnp.isin(regime_id_arr, jnp.array(sorted(regime_ids)))
        relevant_values = values[in_relevant_regime]
        if relevant_values.size == 0:
            continue
        invalid_mask = jnp.isin(
            relevant_values, jnp.array(sorted(valid_codes)), invert=True
        )
        if jnp.any(invalid_mask):
            invalid_vals = sorted({int(v) for v in relevant_values[invalid_mask]})
            raise InvalidInitialConditionsError(
                f"Invalid values {invalid_vals} for discrete state "
                f"'{state_name}'. Valid codes are: {sorted(valid_codes)}"
            )


# Target peak memory budget for the vmapped feasibility computation.
# Assumes float32 intermediates (JAX default); doubles under x64 mode, but this is
# a heuristic — being off by 2x just means larger batches, not correctness issues.
_TARGET_BATCH_BYTES = 256 * 1024 * 1024
_BYTES_PER_ACTION_ELEMENT = 4


def _batched_feasibility_check(
    *,
    feasibility_func: Callable[..., BoolND],
    subject_states: Mapping[str, FloatND | IntND],
    action_kwargs: Mapping[str, FloatND | IntND],
    filtered_params: Mapping[str, object],
    flat_actions: Mapping[ActionName, FloatND | IntND],
) -> BoolND:
    """Check feasibility for all subjects, batching to avoid OOM.

    Vmaps over action combos individually (like solve/simulate do) so each
    feasibility call receives scalar actions that broadcast naturally with
    MappingLeaf parameters.

    Args:
        feasibility_func: Feasibility function for this regime.
        subject_states: Per-subject state arrays (shape `(n_subjects,)` each).
        action_kwargs: Action arrays from the flat action grid, keyed by name.
        filtered_params: Parameter values accepted by the feasibility function.
        flat_actions: Mapping of action names to flat grid arrays (used to compute
            batch size).

    Returns:
        Boolean array of shape `(n_subjects,)` — True where at least one action is
        feasible.

    """
    if action_kwargs:

        def _is_combo_feasible(
            action_kw: dict[str, FloatND | IntND],
            subject_kw: dict[str, FloatND | IntND],
        ) -> BoolND:
            return feasibility_func(**action_kw, **subject_kw, **filtered_params)

        def _is_any_action_feasible(
            per_subject_kwargs: dict[str, FloatND | IntND],
        ) -> BoolND:
            per_combo = jax.vmap(_is_combo_feasible, in_axes=(0, None))(
                action_kwargs,
                per_subject_kwargs,
            )
            return jnp.any(per_combo)

    else:

        def _is_any_action_feasible(
            per_subject_kwargs: dict[str, FloatND | IntND],
        ) -> BoolND:
            return jnp.any(feasibility_func(**per_subject_kwargs, **filtered_params))

    vmapped_check = jax.vmap(_is_any_action_feasible)

    n_subjects = len(next(iter(subject_states.values())))
    n_action_combos = max(len(v) for v in flat_actions.values())
    batch_size = max(
        1,
        _TARGET_BATCH_BYTES // max(n_action_combos * _BYTES_PER_ACTION_ELEMENT, 1),
    )

    if n_subjects <= batch_size:
        return vmapped_check(subject_states)

    results: list[BoolND] = []
    for start in range(0, n_subjects, batch_size):
        end = min(start + batch_size, n_subjects)
        batch = {k: v[start:end] for k, v in subject_states.items()}
        results.append(vmapped_check(batch))
    return jnp.concatenate(results)


def _check_regime_feasibility(  # noqa: C901
    *,
    regime: Regime,
    regime_name: RegimeName,
    initial_states: Mapping[StateName, FloatND | IntND],
    subject_indices: list[int],
    regime_params: Mapping[str, object],
    ages: AgeGrid,
) -> str | None:
    """Check whether all subjects in a regime have at least one feasible action.

    Args:
        regime: The internal regime instance.
        regime_name: Name of the regime.
        initial_states: Mapping of state names to arrays (includes "age").
        subject_indices: Indices of subjects starting in this regime.
        regime_params: Merged fixed and runtime parameters for this regime.
        ages: AgeGrid for the model.

    Returns:
        An error message string if any subjects are infeasible, or None.

    """
    feasibility_func = _get_feasibility(
        functions=regime.simulation.functions,
        constraints=regime.simulation.constraints,
    )
    accepted = get_union_of_args([feasibility_func])

    action_names = list(regime.solution.variables.action_names)
    if not action_names:
        return None

    # Build the state-action space with runtime-supplied grid points
    # substituted. The base grid's `to_jax()` raises for runtime-supplied
    # `IrregSpacedGrid`s declared with `pass_points_at_runtime=True`, so the
    # validator must read points from `state_action_space(regime_params=...)`.
    state_action_space = regime.solution.state_action_space(
        regime_params=cast("FlatRegimeParams", MappingProxyType(dict(regime_params))),
    )
    action_grids: dict[ActionName, FloatND | IntND] = {
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
    }
    flat_actions = _build_flat_action_grid(
        action_names=action_names,
        grids=MappingProxyType(action_grids),
    )

    filtered_params = {k: v for k, v in regime_params.items() if k in accepted}
    # Simulate state set: a carried pair state is a leaf of the feasibility
    # function, so the constraint is checked against the seeded true value,
    # not the solve-phase imputation.
    state_names = list(regime.simulation.state_names)
    needs_age = "age" in accepted
    needs_period = "period" in accepted

    # Build per-subject state arrays
    idx_arr = jnp.array(subject_indices, dtype=jnp.int32)
    subject_states: dict[StateName, FloatND | IntND] = {}
    for sn in state_names:
        if sn in accepted:
            subject_states[sn] = initial_states[sn][idx_arr]

    if needs_age:
        subject_states["age"] = initial_states["age"][idx_arr]
    if needs_period:
        subject_states["period"] = jnp.array(
            [ages.age_to_period(a.item()) for a in initial_states["age"][idx_arr]],
            dtype=jnp.int32,
        )

    # Split actions and params — actions are vmapped over, params are not
    action_kwargs: dict[str, FloatND | IntND] = {
        k: v for k, v in flat_actions.items() if k in accepted
    }

    if subject_states:
        try:
            any_feasible = _batched_feasibility_check(
                feasibility_func=feasibility_func,
                subject_states=subject_states,
                action_kwargs=action_kwargs,
                filtered_params=filtered_params,
                flat_actions=flat_actions,
            )
        except TypeError as exc:
            _raise_feasibility_type_error(
                exc=exc,
                regime_name=regime_name,
                regime=regime,
                subject_states=subject_states,
            )
        infeasible_mask = np.asarray(~any_feasible)
        infeasible_indices = np.asarray(idx_arr)[infeasible_mask].tolist()
    else:
        # No per-subject varying states: feasibility is identical for all subjects.
        if action_kwargs:

            def _check_combo(action_kw: dict[str, FloatND | IntND]) -> BoolND:
                return feasibility_func(**action_kw, **filtered_params)  # ty: ignore[invalid-argument-type]

            result = jax.vmap(_check_combo)(action_kwargs)
        else:
            result = feasibility_func(**filtered_params)  # ty: ignore[invalid-argument-type]
        infeasible_indices = [] if jnp.any(result) else subject_indices

    if not infeasible_indices:
        return None

    per_constraint_admits_any = _per_constraint_feasibility(
        regime=regime,
        subject_states=subject_states,
        regime_params=regime_params,
        flat_actions=flat_actions,
        idx_arr=idx_arr,
        infeasible_indices=infeasible_indices,
    )

    return _format_infeasibility_message(
        infeasible_indices=infeasible_indices,
        regime=regime,
        regime_name=regime_name,
        initial_states=initial_states,
        state_names=state_names,
        per_constraint_admits_any=per_constraint_admits_any,
    )


def _admits_any_action(
    *,
    feasibility_func: Callable[..., BoolND],
    action_kwargs: Mapping[str, FloatND | IntND],
    params: Mapping[str, object],
) -> bool:
    """Return True iff the feasibility function admits ≥ 1 action under params."""
    if action_kwargs:

        def _check_combo(action_kw: dict[str, FloatND | IntND]) -> BoolND:
            return feasibility_func(**action_kw, **params)

        per_combo = jax.vmap(_check_combo)(action_kwargs)
        return bool(jnp.any(per_combo))
    return bool(feasibility_func(**params))


def _per_constraint_feasibility(
    *,
    regime: Regime,
    subject_states: Mapping[str, FloatND | IntND],
    regime_params: Mapping[str, object],
    flat_actions: Mapping[ActionName, FloatND | IntND],
    idx_arr: Int1D,
    infeasible_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Per-constraint feasibility for the infeasible subjects.

    For each constraint, returns a boolean array (one entry per infeasible
    subject) indicating whether that constraint *individually* admits at
    least one action. Combined with the regime's feasibility verdict, this
    distinguishes "constraint X rejects every action by itself" from
    "constraints jointly reject everything despite each admitting some".

    Each constraint's feasibility function has its own argument set (a
    subset of the combined feasibility's union); filter `subject_states`,
    `action_kwargs`, and `filtered_params` per constraint so dags doesn't
    raise on stray kwargs.
    """
    constraints = regime.simulation.constraints
    functions = regime.simulation.functions
    if not constraints or not subject_states:
        return {}

    infeasible_positions = np.flatnonzero(
        np.isin(np.asarray(idx_arr), np.asarray(infeasible_indices))
    )
    infeasible_states = {
        name: arr[infeasible_positions] for name, arr in subject_states.items()
    }

    out: dict[str, np.ndarray] = {}
    for name, constraint_func in constraints.items():
        single_feasibility = _get_feasibility(
            functions=functions,
            constraints=MappingProxyType({name: constraint_func}),
        )
        accepted = get_union_of_args([single_feasibility])
        single_states = {k: v for k, v in infeasible_states.items() if k in accepted}
        single_actions = {k: v for k, v in flat_actions.items() if k in accepted}
        single_params = {k: v for k, v in regime_params.items() if k in accepted}
        n = len(infeasible_indices)
        if not single_states:
            # Action-only / parameter-only constraint — identical for all subjects.
            admits_any = _admits_any_action(
                feasibility_func=single_feasibility,
                action_kwargs=single_actions,
                params=single_params,
            )
            out[name] = np.full(n, admits_any, dtype=bool)
            continue
        any_feasible = _batched_feasibility_check(
            feasibility_func=single_feasibility,
            subject_states=single_states,
            action_kwargs=single_actions,
            filtered_params=single_params,
            flat_actions=flat_actions,
        )
        out[name] = np.asarray(any_feasible)
    return out


def _raise_feasibility_type_error(
    *,
    exc: TypeError,
    regime_name: RegimeName,
    regime: Regime,
    subject_states: dict[StateName, FloatND | IntND],
) -> NoReturn:
    """Re-raise a TypeError from feasibility checking with diagnostic context.

    Args:
        exc: The original TypeError from the feasibility check.
        regime_name: Name of the regime being checked.
        regime: The internal regime containing variable info.
        subject_states: Mapping of state names to arrays for subjects in
            this regime.

    Raises:
        InvalidInitialConditionsError: Always — wraps `exc` with a dtype hint
            when any discrete state has a non-integer dtype.

    """
    discrete_names = set(regime.simulation.discrete_grids)

    bad_dtypes: list[str] = []
    for name, arr in subject_states.items():
        if name in discrete_names and not jnp.issubdtype(arr.dtype, jnp.integer):
            bad_dtypes.append(f"  {name!r}: dtype={arr.dtype} (expected integer)")

    hint = ""
    if bad_dtypes:
        hint = (
            "\n\nDiscrete states with wrong dtype:\n"
            + "\n".join(bad_dtypes)
            + "\n\nDiscrete states are used as array indices and must have integer "
            "dtype. Check that initial conditions encode categorical states as int "
            "codes, not floats."
        )

    msg = f"TypeError in feasibility check for regime {regime_name!r}: {exc}{hint}"
    raise InvalidInitialConditionsError(msg) from exc


def _format_infeasibility_message(
    *,
    infeasible_indices: Sequence[int],
    regime: Regime,
    regime_name: RegimeName,
    initial_states: Mapping[StateName, FloatND | IntND],
    state_names: Sequence[str],
    per_constraint_admits_any: Mapping[str, np.ndarray],
) -> str:
    """Format an error message for infeasible subjects.

    Args:
        infeasible_indices: Indices of subjects with no feasible action.
        regime: The internal regime instance.
        regime_name: Name of the regime.
        initial_states: Mapping of state names to arrays.
        state_names: List of state variable names.
        per_constraint_admits_any: Mapping from constraint name to a boolean
            array (one entry per infeasible subject) — True where that
            constraint *individually* admits at least one action. False
            entries identify constraints that reject every action on their
            own; rows with all-True entries are infeasible only because the
            constraints jointly reject the action set.

    Returns:
        Formatted error message string.

    """
    # Build DataFrame of infeasible subjects' states
    state_df = pd.DataFrame(
        {
            name: [float(initial_states[name][i]) for i in infeasible_indices]
            for name in state_names
            if name in initial_states
        },
        index=list(infeasible_indices),
    )
    state_df.index.name = "subject"

    # Convert discrete codes to labels
    for name, grid in regime.simulation.discrete_grids.items():
        if name in state_df.columns:
            state_df[name] = [grid.categories[int(v)] for v in state_df[name]]

    # Append one boolean column per constraint: True = admits ≥ 1 action,
    # False = rejects every action by itself for that subject.
    for name, mask in per_constraint_admits_any.items():
        state_df[name] = list(mask)

    # Truncate for large groups
    n = len(infeasible_indices)
    max_show = 10
    if n > max_show:
        table_str = state_df.head(max_show).to_string()
        table_str += f"\n  ... and {n - max_show} more"
    else:
        table_str = state_df.to_string()

    return (
        f"All actions are infeasible for {n} subject(s) "
        f"in regime '{regime_name}'.\n\n"
        f"Per-constraint admissibility (True = constraint admits ≥ 1 "
        f"action by itself; False = constraint rejects every action):\n"
        f"{table_str}\n\n"
        f"No action combination satisfies all constraints jointly for "
        f"these initial states."
    )


def _build_flat_action_grid(
    *,
    action_names: list[ActionName],
    grids: MappingProxyType[str, FloatND | IntND],
) -> dict[str, FloatND | IntND]:
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
