import jax.numpy as jnp
from jax import Array

from lcm.exceptions import InvalidValueFunctionError


def validate_value_function_array_integrity(
    V_arr: Array,
    period: int,
) -> None:
    """Validate the value function array for NaN values.

    This function checks the value function array for any NaN values. If any such values
    are found, we raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        period: The period for which the value function is being validated.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if jnp.any(jnp.isnan(V_arr)):
        raise InvalidValueFunctionError(
            f"The value function array in period {period} contains NaN values."
        )
