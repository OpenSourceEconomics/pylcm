import jax.numpy as jnp
from jax import Array

from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ScalarFloat


def validate_value_function_array(*, V_arr: Array, age: ScalarFloat) -> None:
    """Validate the value function array for NaN values.

    This function checks the value function array for any NaN values. If any such values
    are found, we raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if jnp.any(jnp.isnan(V_arr)):
        raise InvalidValueFunctionError(
            f"The value function array at age {age} contains NaN values. This "
            "may be due to various reasons:\n"
            "- The user-defined functions returned invalid values.\n"
            "- It is impossible to reach an active regime, resulting in NaN regime\n"
            "  transition probabilities in the normalized transition probabilities."
        )
