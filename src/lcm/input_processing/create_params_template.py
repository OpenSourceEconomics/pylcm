from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from lcm.input_processing.util import (
    get_variable_info,
    is_stochastic_transition,
)
from lcm.utils import unflatten_regime_namespace

if TYPE_CHECKING:
    import pandas as pd

    from lcm.regime import Regime
    from lcm.typing import GridsDict, ParamsDict


def create_params_template(
    regime: Regime,
    grids: GridsDict,
    n_periods: int,
    default_params: dict[str, float] = {"beta": jnp.nan},  # noqa: B006
) -> ParamsDict:
    """Create parameter template from a regime specification.

    Args:
        regime: The regime as provided by the user.
        grids: Dictionary containing the state grids for each regime.
        n_periods: Number of periods of the model.
        default_params: A dictionary of default parameters. Default is None. If None,
            the default {"beta": np.nan} is used. For other lifetime reward objectives,
            additional parameters may be required, for example {"beta": np.nan, "delta":
            np.nan} for beta-delta discounting.

    Returns:
        A nested dictionary of regime parameters.

    """
    variable_info = get_variable_info(regime)

    stochastic_transitions = _create_stochastic_transition_params(
        regime=regime, variable_info=variable_info, grids=grids, n_periods=n_periods
    )
    stochastic_transition_params = {"shocks": stochastic_transitions}

    function_params = _create_function_params(regime)

    return default_params | function_params | stochastic_transition_params


def _create_function_params(regime: Regime) -> dict[str, dict[str, float]]:
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

    if hasattr(regime, "shocks"):
        variables = variables | set(regime.shocks)

    function_params = {}
    # For each user function, capture the arguments of the function that are not in the
    # set of regime variables, and initialize them.
    for name, func in regime.get_all_functions().items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        function_params[name] = dict.fromkeys(params, jnp.nan)

    return function_params


def _create_stochastic_transition_params(
    regime: Regime,
    variable_info: pd.DataFrame,
    grids: GridsDict,
    n_periods: int,
) -> dict[str, Array]:
    """Create parameters for stochastic transitions.

    Args:
        regime: The regime as provided by the user.
        variable_info: A dataframe with information about the variables.
        grids: A dictionary of grids consistent with regime.
        n_periods: Number of periods in the model.

    Returns:
        A dictionary of parameters required for stochastic transitions, initialized with
        jnp.nan matrices of the correct dimensions.

    """
    # Create template matrices for stochastic transitions
    # ==================================================================================

    # Stochastic transition functions can only depend on discrete vars or 'period'.
    valid_vars = set(variable_info.query("is_discrete").index) | {"period"}

    stochastic_transition_params = {}
    invalid_dependencies = {}
    for regime_name, regime_transitions in regime.transitions.items():
        if regime_name == "next_regime":
            transitions_to_process = {"next_regime": regime_transitions}
        else:
            transitions_to_process = regime_transitions  # type: ignore[assignment]

        for func_name, transition in transitions_to_process.items():
            if is_stochastic_transition(transition):  # type: ignore[arg-type]
                # Retrieve corresponding next function and its arguments
                dependencies = list(inspect.signature(transition).parameters)  # type: ignore[arg-type]

                # If there are invalid dependencies, store them in a dictionary and
                # continue with the next variable to collect as many invalid
                # arguments as possible.
                invalid = set(dependencies) - valid_vars
                if invalid:
                    invalid_dependencies[func_name] = invalid
                else:
                    # Get the dims of variables that influence the stochastic variable
                    dimensions_of_deps = [
                        len(grids[regime_name][arg]) if arg != "period" else n_periods
                        for arg in dependencies
                    ]
                    # Add the dimension of the stochastic variable itself at the end
                    dimensions = (
                        *dimensions_of_deps,
                        len(grids[regime_name][func_name.removeprefix("next_")]),
                    )

                    stochastic_transition_params[f"{regime_name}__{func_name}"] = (
                        jnp.full(dimensions, jnp.nan)
                    )

    # Raise an error if there are invalid arguments
    # ==================================================================================
    if invalid_dependencies:
        raise ValueError(
            f"Stochastic transition functions can only depend on discrete variables or "
            "'period'. The following variables have invalid arguments: "
            f"{invalid_dependencies}.",
        )

    return unflatten_regime_namespace(stochastic_transition_params)
