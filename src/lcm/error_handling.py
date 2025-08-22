from datetime import UTC, datetime
from uuid import uuid4

import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm._config import get_log_dir
from lcm.exceptions import InvalidValueFunctionError
from lcm.interfaces import StateActionSpace, StateSpaceInfo


def validate_value_function_array_integrity(
    V_arr: Array,
    state_action_space: StateActionSpace,
    period: int,
    state_space_info: StateSpaceInfo,
) -> None:
    """Validate the value function array for NaN and infinity values.

    This function checks the value function array for any NaN or infinity values. If any
    such values are found, we log the corresponding state-combinations to a CSV file in
    the '.pylcm' directory and raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        state_action_space: The state-action space to check against.
        state_space_info: The state space info to check against.
        period: The period for which the value function is being validated.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN or infinity
            values. The invalid states are logged to CSV files in the '.pylcm' directory
            for further analysis.

    """
    # Early return if the value function array contains only valid values
    if not jnp.any(_is_invalid(V_arr)):
        return

    invalid_states = _get_dataframe_of_invalid_states(
        V_arr=V_arr,
        ordered_state_names=list(state_action_space.states),
        states=state_action_space.states,
        period=period,
    )
    extrapolation = False
    if state_space_info.continuous_states is not None:
        for key in state_space_info.continuous_states:
            if (
                invalid_states[key]
                < state_space_info.continuous_states[key].to_jax()[0]
            ).any() or (
                invalid_states[key]
                > state_space_info.continuous_states[key].to_jax()[-1]
            ).any():
                extrapolation = True
    log_dir = get_log_dir()
    file_path = log_dir / f"invalid_states_{_generate_unique_suffix()}.csv"

    invalid_states.to_csv(file_path, index=False, header=True)
    if extrapolation:
        raise InvalidValueFunctionError(
            f"The value function array in period {period} contains NaN or infinite "
            f"values. This might occur because the value function is extrapolated "
            f"without considering the constraints. Check if your simulated agents "
            f"take actions that transition them to states outside of the specified "
            f"grid that have no valid actions. \n\n Invalid state combinations "
            f"were logged to:\n {file_path}"
        )
    raise InvalidValueFunctionError(
        f"The value function array in period {period} contains NaN or infinite values."
        f"\n\n Invalid state combinations were logged to:\n {file_path}"
    )


def _get_dataframe_of_invalid_states(
    V_arr: Array,
    ordered_state_names: list[str],
    states: dict[str, Array],
    period: int,
) -> pd.DataFrame:
    """Get a DataFrame of invalid states based on a check function.

    We define invalid states as those that have NaN or infinity values in the
    value function array.

    Returns:
        A DataFrame containing the invalid state combinations, with state names as
        columns and the corresponding invalid values as rows. Additionally, the
        following columns are included:
        - '__value__' (values from the that correspond to the invalid states)
        - '__period__' (the period for which the value function is being validated)

    """
    invalid_idx = jnp.argwhere(_is_invalid(V_arr))
    invalid_states: dict[str, Array | int] = {}
    for state_idx, state_name in enumerate(ordered_state_names):
        invalid_states[state_name] = states[state_name][invalid_idx[:, state_idx]]
    invalid_states["__value__"] = V_arr[tuple(invalid_idx.T)]
    invalid_states["__period__"] = period
    return pd.DataFrame(invalid_states)


def _is_invalid(a: Array) -> Array:
    """Check if the array contains any invalid (NaN or infinity) values."""
    return jnp.isnan(a) | jnp.isinf(a)


def _generate_unique_suffix() -> str:
    """Return a unique, filename-safe suffix based on the current time and a UUID.

    The timestamp makes filenames readable, while the UUID ensures uniqueness (e.g.,
    when multiple files are created in the same second).

    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid4().hex[:8]}"
