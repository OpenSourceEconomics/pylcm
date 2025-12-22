from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from lcm.regime import Regime
    from lcm.typing import GridsDict, ParamsDict


def create_params_template(
    regime: Regime,
    grids: GridsDict,  # noqa: ARG001
    n_periods: int,  # noqa: ARG001
    default_params: dict[str, float] = {"discount_factor": jnp.nan},  # noqa: B006
) -> ParamsDict:
    """Create parameter template from a regime specification.

    Args:
        regime: The regime as provided by the user.
        grids: Dictionary containing the state grids for each regime.
        n_periods: Number of periods of the model.
        default_params: A dictionary of default parameters. Default is None. If None,
            the default {"discount_factor": np.nan} is used. For other lifetime reward
            objectives, additional parameters may be required, for example
            {"discount_factor": np.nan, "delta": np.nan} for beta-delta discounting.

    Returns:
        The regime parameter template.

    """
    function_params = _create_function_params(regime)

    return default_params | function_params


def _create_function_params(
    regime: Regime,
) -> dict[str, dict[str, float]]:
    """Get function parameters from a regime specification.

    Explanation: We consider the arguments of all regime functions, from which we
    exclude all variables that are states, actions or the period argument. Everything
    else is considered a parameter of the respective regime function that is provided by
    the user.

    Args:
        regime: The regime as provided by the user.

    Returns:
        A dictionary for each regime function, containing a parameters required in the
        regime functions, initialized with jnp.nan.

    """
    # Collect all regime variables, that includes actions, states, the period, and
    # auxiliary variables (regime function names).
    variables = {
        *regime.functions,
        *regime.actions,
        *regime.states,
        "period",
    }

    function_params = {}
    # For each user function, capture the arguments of the function that are not in the
    # set of regime variables, and initialize them.
    for name, func in regime.get_all_functions().items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        function_params[name] = dict.fromkeys(params, jnp.nan)

    return function_params
