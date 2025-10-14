"""Create a parameter template for a regime specification."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from lcm.input_processing.util import get_grids, get_variable_info

if TYPE_CHECKING:
    import pandas as pd

    from lcm.regime import Regime
    from lcm.typing import ParamsDict


def create_params_template(
    regime: Regime,
    default_params: dict[str, float] = {"beta": jnp.nan},  # noqa: B006
) -> ParamsDict:
    """Create parameter template from a regime specification.

    Args:
        regime: The regime as provided by the user.
        default_params: A dictionary of default parameters. Default is None. If None,
            the default {"beta": np.nan} is used. For other lifetime reward objectives,
            additional parameters may be required, for example {"beta": np.nan, "delta":
            np.nan} for beta-delta discounting.

    Returns:
        A nested dictionary of regime parameters.

    """
    variable_info = get_variable_info(regime)
    grids = get_grids(regime)

    if variable_info["is_stochastic"].any():
        stochastic_transitions = _create_stochastic_transition_params(
            regime=regime,
            variable_info=variable_info,
            grids=grids,
        )
        stochastic_transition_params = {"shocks": stochastic_transitions}
    else:
        stochastic_transition_params = {}

    function_params = _create_function_params(regime)

    return default_params | function_params | stochastic_transition_params


def _create_function_params(regime: Regime) -> dict[str, dict[str, float]]:
    """Get function parameters from a regime specification.

    Explanation: We consider the arguments of all regime functions, from which
    we exclude all variables that are states, actions or the period argument.
    Everything else is considered a parameter of the respective regime function
    that is provided by the user.

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
        "_period",
    }

    if hasattr(regime, "shocks"):
        variables = variables | set(regime.shocks)

    function_params = {}
    # For each regime function, capture the arguments of the function that are
    # not in the set of regime variables, and initialize them.
    for name, func in regime.functions.items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        function_params[name] = dict.fromkeys(params, jnp.nan)

    # Process regime_transition_probs if it exists
    if (
        hasattr(regime, "regime_transition_probs")
        and regime.regime_transition_probs is not None
    ):
        arguments = set(inspect.signature(regime.regime_transition_probs).parameters)
        params = sorted(arguments.difference(variables))
        function_params["regime_transition_probs"] = dict.fromkeys(params, jnp.nan)

    return function_params


def _create_stochastic_transition_params(
    regime: Regime,
    variable_info: pd.DataFrame,
    grids: dict[str, Array],
) -> dict[str, Array]:
    """Create parameters for stochastic transitions.

    Args:
        regime: The regime as provided by the user.
        variable_info: A dataframe with information about the variables.
        grids: A dictionary of grids consistent with regime.

    Returns:
        A dictionary of parameters required for stochastic transitions, initialized with
        jnp.nan matrices of the correct dimensions.

    """
    stochastic_variables = variable_info.query("is_stochastic").index.tolist()

    # Assert that all stochastic variables are discrete state variables
    # ==================================================================================
    discrete_state_vars = set(variable_info.query("is_state & is_discrete").index)

    invalid = set(stochastic_variables) - discrete_state_vars
    if invalid:
        raise ValueError(
            f"The following variables are stochastic, but are not discrete state "
            f"variables: {invalid}. This is currently not supported.",
        )

    # Create template matrices for stochastic transitions
    # ==================================================================================

    # Stochastic transition functions can only depend on discrete vars or '_period'.
    valid_vars = set(variable_info.query("is_discrete").index) | {"_period"}

    stochastic_transition_params = {}
    invalid_dependencies = {}

    for var in stochastic_variables:
        # Retrieve corresponding next function and its arguments
        next_var = regime.functions[f"next_{var}"]
        dependencies = list(inspect.signature(next_var).parameters)

        # If there are invalid dependencies, store them in a dictionary and continue
        # with the next variable to collect as many invalid arguments as possible.
        invalid = set(dependencies) - valid_vars
        if invalid:
            invalid_dependencies[var] = invalid
        else:
            # Get the dimensions of variables that influence the stochastic variable
            dimensions_of_deps = [
                len(grids[arg])
                if arg != "_period"
                else 1  # TODO: This should be list(regime.active) or n_periods if regime.active is None.
                for arg in dependencies
            ]
            # Add the dimension of the stochastic variable itself at the end
            dimensions = (*dimensions_of_deps, len(grids[var]))

            stochastic_transition_params[var] = jnp.full(dimensions, jnp.nan)

    # Raise an error if there are invalid arguments
    # ==================================================================================
    if invalid_dependencies:
        raise ValueError(
            f"Stochastic transition functions can only depend on discrete variables or "
            "'_period'. The following variables have invalid arguments: "
            f"{invalid_dependencies}.",
        )

    return stochastic_transition_params
