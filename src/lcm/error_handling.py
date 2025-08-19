from collections.abc import Callable

import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm._config import get_log_dir
from lcm.exceptions import InvalidValueFunctionError
from lcm.interfaces import StateActionSpace


def validate_value_function_array_integrity(
    V_arr: Array,
    state_action_space: StateActionSpace,
) -> None:
    """Validate the value function array for NaN and infinity values.

    This function checks the value function array for any NaN or infinity values. If any
    such values are found, we log the corresponding state-combinations to

    Args:
        V_arr: The value function array to validate.
        state_action_space: The state-action space to check against.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN or infinity
            values. The invalid states are logged to CSV files in the '.pylcm' directory
            for further analysis.

    """
    # Early return if the value function array contains only valid values
    if not jnp.any(jnp.isnan(V_arr) | jnp.isinf(V_arr)):
        return

    nan_states_df = _get_dataframe_of_invalid_states(
        V_arr=V_arr,
        check_fn=jnp.isnan,
        ordered_state_names=list(state_action_space.states),
        states=state_action_space.states,
    )

    inf_states_df = _get_dataframe_of_invalid_states(
        V_arr=V_arr,
        check_fn=jnp.isinf,
        ordered_state_names=list(state_action_space.states),
        states=state_action_space.states,
    )

    log_dir = get_log_dir()

    for df, filename in zip(
        [nan_states_df, inf_states_df],
        ["invalid_nan_states.csv", "invalid_inf_states.csv"],
        strict=False,
    ):
        if not df.empty:
            df.to_csv(log_dir / filename, index=False, header=True)

    raise InvalidValueFunctionError(
        "The value function array contains NaN or infinite values.\n\n"
        "The invalid states have been logged to their respective CSV files in the "
        "'.pylcm' directory."
    )


def _get_dataframe_of_invalid_states(
    V_arr: Array,
    check_fn: Callable[[Array], Array],
    ordered_state_names: list[str],
    states: dict[str, Array],
) -> pd.DataFrame:
    """Get a DataFrame of invalid states based on a check function."""
    invalid_indices = jnp.argwhere(check_fn(V_arr))
    invalid_states = {}
    for state_idx, state_name in enumerate(ordered_state_names):
        invalid_states[state_name] = states[state_name][invalid_indices[:, state_idx]]
    return pd.DataFrame(invalid_states)
