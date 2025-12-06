from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from lcm.input_processing.util import (
    is_stochastic_transition,
)
from lcm.utils import flatten_regime_namespace, unflatten_regime_namespace

if TYPE_CHECKING:
    import pandas as pd

    from lcm.regime import Regime
    from lcm.typing import GridsDict, ParamsDict, UserFunction


def create_params_template(
    regime: Regime,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    grids: GridsDict,  # noqa: ARG001
    n_periods: int,  # noqa: ARG001
    default_params: dict[str, float] = {"beta": jnp.nan},  # noqa: B006
) -> ParamsDict:
    """Create parameter template from a regime specification.

    Args:
        regime: The regime as provided by the user.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": fn, ...}, "next_regime": fn}
        grids: Dictionary containing the state grids for each regime.
        n_periods: Number of periods of the model.
        default_params: A dictionary of default parameters. Default is None. If None,
            the default {"beta": np.nan} is used. For other lifetime reward objectives,
            additional parameters may be required, for example {"beta": np.nan, "delta":
            np.nan} for beta-delta discounting.

    Returns:
        A nested dictionary of regime parameters.

    """
    function_params = _create_function_params(regime, nested_transitions)

    return default_params | function_params


def _create_function_params(
    regime: Regime,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],  # noqa: ARG001
) -> dict[str, dict[str, float]]:
    """Get function parameters from a regime specification.

    Explanation: We consider the arguments of all regime functions, from which we
    exclude all variables that are states, actions or the period argument. Everything
    else is considered a parameter of the respective regime function that is provided by
    the user.

    Args:
        regime: The regime as provided by the user.
        nested_transitions: Nested transitions dict for internal processing.

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

    # Build all_functions using flat transitions (user-facing names without prefix)
    # The user provides flat transitions like {"next_wealth": fn, "next_regime": fn}
    # and params should use the same flat names
    all_functions = flatten_regime_namespace(
        regime.functions
        | {"utility": regime.utility}
        | regime.constraints
        | regime.transitions  # Use flat transitions for flat param keys
    )

    function_params = {}
    # For each user function, capture the arguments of the function that are not in the
    # set of regime variables, and initialize them.
    for name, func in all_functions.items():
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

    invalid_vars = set(variable_info.query("is_continuous").index)

    stochastic_transition_params = {}
    invalid_dependencies = {}
    for regime_name, regime_transitions in regime.transitions.items():
        if regime_name == "next_regime":
            transitions_to_process = {"next_regime": regime_transitions}
        else:
            transitions_to_process = regime_transitions  # type: ignore[assignment]

        for func_name, transition in transitions_to_process.items():
            if is_stochastic_transition(transition):
                # Retrieve corresponding next function and its arguments
                dependencies = list(inspect.signature(transition).parameters)

                # Filter out parameters (arguments that are not model variables).
                # Model variables are in variable_info or are 'period'.
                model_variable_deps = [
                    dep
                    for dep in dependencies
                    if dep == "period" or dep in variable_info.index
                ]

                # If there are invalid dependencies, store them in a dictionary and
                # continue with the next variable to collect as many invalid
                # arguments as possible.
                invalid = set(model_variable_deps).intersection(invalid_vars)
                if invalid:
                    invalid_dependencies[func_name] = invalid
                else:
                    # Filter to dependencies that contribute to dimensions (in grids
                    # or 'period'). Other model variables may exist but not be in
                    # grids for this regime.
                    deps_for_dims = [
                        dep
                        for dep in model_variable_deps
                        if dep == "period" or dep in grids[regime_name]
                    ]
                    # Get the dims of variables that influence the stochastic variable
                    dimensions_of_deps = [
                        len(grids[regime_name][arg]) if arg != "period" else n_periods
                        for arg in deps_for_dims
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
            f"Stochastic transition functions cannot depend on continuous variables. "
            f"The following transitions have invalid arguments: {invalid_dependencies}."
        )

    return unflatten_regime_namespace(stochastic_transition_params)
