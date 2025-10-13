from __future__ import annotations

import dataclasses
import functools
import inspect
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, Any, cast

import jax
from dags import get_annotations
from dags.signature import with_signature

from lcm.functools import convert_kwargs_to_args
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.util import (
    get_function_info,
    get_grids,
    get_gridspecs,
    get_variable_info,
)
from lcm.interfaces import ShockType
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.regime import Regime
from lcm.state_action_space import create_state_action_space, create_state_space_info

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from jax import Array

    from lcm.grids import Grid
    from lcm.interfaces import StateActionSpace, StateSpaceInfo
    from lcm.model import Model
    from lcm.typing import (
        ArgmaxQOverAFunction,
        DiscreteAction,
        DiscreteState,
        Float1D,
        FloatND,
        Int1D,
        InternalUserFunction,
        MaxQOverAFunction,
        ParamsDict,
        UserFunction,
    )


@dataclasses.dataclass(frozen=False)
class InternalRegime:
    """An internal representation of a regime."""

    name: str
    description: str | None
    active: list[int]
    grids: dict[str, Float1D | Int1D]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    functions: dict[str, InternalUserFunction]
    function_info: pd.DataFrame
    params: ParamsDict
    # Not properly processed yet
    random_utility_shocks: ShockType
    regime_transition_probs: InternalUserFunction | None

    # Computed model components (set in __post_init__)
    params_template: ParamsDict = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    state_space_infos: dict[int, StateSpaceInfo] = field(init=False)
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = field(init=False)
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = field(
        init=False
    )


def process_regimes(
    model: Model, regimes: Regime | Sequence[Regime]
) -> list[InternalRegime]:
    """Process the user regimes.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the regime specification is valid.

    Args:
        regimes: The regimes as provided by the user. Can be a single Regime or a
            sequence of Regimes.

    Returns:
        The list of processed regimes.

    """
    if isinstance(regimes, Regime):
        regimes = [regimes]

    internal_regimes = []
    ssi = {}
    sas = {}
    for regime in regimes:
        internal_regime = _process_regime(regime)
        _initialize_state_space(internal_regime, model.n_periods)
        ssi[regime.name] = internal_regime.state_space_infos
        sas[regime.name] = internal_regime.state_action_spaces
    for regime in regimes:
        _initialize_regime_components(
            internal_regime, model.n_periods, model.enable_jit, ssi, sas
        )
        internal_regimes.append(internal_regime)
    return internal_regimes


def _process_regime(regime: Regime) -> InternalRegime:
    params = create_params_template(regime)

    # Process regime_transition_probs if it exists
    regime_transition_probs_processed = None
    if regime.regime_transition_probs is not None:
        func_name = "regime_transition_probs"
        if params.get(func_name):
            regime_transition_probs_processed = _replace_func_parameters_by_params(
                func=regime.regime_transition_probs,
                params=params,
                name=func_name,
            )
        else:
            regime_transition_probs_processed = _add_dummy_params_argument(
                regime.regime_transition_probs
            )

    return InternalRegime(
        name=regime.name,
        description=regime.description,
        active=list(regime.active),
        grids=get_grids(regime),
        gridspecs=get_gridspecs(regime),
        variable_info=get_variable_info(regime),
        functions=_get_internal_functions(regime, params=params),
        function_info=get_function_info(regime),
        params=params,
        # currently no additive utility shocks are supported
        random_utility_shocks=ShockType.NONE,
        regime_transition_probs=regime_transition_probs_processed,
    )


def _initialize_state_space(
    internal_regime: InternalRegime, n_periods
) -> dict[str, Any]:
    # Process model to internal representation

    # Initialize containers
    state_action_spaces: dict[int, StateActionSpace] = {}
    state_space_infos: dict[int, StateSpaceInfo] = {}

    # Create functions for each period (reversed order following Backward induction)
    for period in reversed(internal_regime.active):
        is_last_period = period == n_periods - 1

        # Create state action space
        state_action_space = create_state_action_space(
            internal_model=internal_regime,
            is_last_period=is_last_period,
        )

        # Create state space info
        state_space_info = create_state_space_info(
            internal_model=internal_regime,
            is_last_period=is_last_period,
        )
        state_action_spaces[period] = state_action_space
        state_space_infos[period] = state_space_info

    internal_regime.state_action_spaces = state_action_spaces
    internal_regime.state_space_infos = state_space_infos


def _initialize_regime_components(
    internal_regime: InternalRegime, model: Model, ssi, sas
):
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = {}
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = {}

    # Create last period's next state space info
    last_periods_next_state_space_info = StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )
    for period in reversed(internal_regime.active):
        state_action_space = internal_regime.state_action_spaces[period]
        state_space_info = internal_regime.state_space_infos[period]
        is_last_period = period == model.n_periods - 1
        if is_last_period:
            next_state_space_info = last_periods_next_state_space_info
        else:
            next_state_space_info = {
                name: regime_ssi[period + 1] for name, regime_ssi in ssi.items()
            }

        # Create Q and F functions
        Q_and_F = get_Q_and_F(
            internal_model=internal_regime,
            next_state_space_info=next_state_space_info,
            period=period,
        )

        # Create optimization functions
        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.continuous_actions)
            + tuple(state_action_space.discrete_actions),
            states_names=tuple(state_action_space.states),
        )

        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.discrete_actions)
            + tuple(state_action_space.continuous_actions),
        )

        # Store results
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if model.enable_jit else max_Q_over_a
        )
        argmax_and_max_Q_over_a_functions[period] = (
            jax.jit(argmax_and_max_Q_over_a)
            if model.enable_jit
            else argmax_and_max_Q_over_a
        )

    internal_regime.max_Q_over_a_functions = max_Q_over_a_functions
    internal_regime.argmax_and_max_Q_over_a_functions = (
        argmax_and_max_Q_over_a_functions
    )


def _get_internal_functions(
    model: Model,
    params: ParamsDict,
) -> dict[str, InternalUserFunction]:
    """Process the user provided model functions.

    Args:
        model: The model as provided by the user.
        params: The parameters of the model.

    Returns:
        Dictionary containing all functions of the model. The keys are the names of the
        functions. The values are the processed functions. The main difference between
        processed and unprocessed functions is that processed functions take `params` as
        argument.

    """
    variable_info = get_variable_info(model)
    grids = get_grids(model)

    raw_functions = deepcopy(model.functions)

    for var in model.states:
        if variable_info.loc[var, "is_stochastic"]:
            raw_functions[f"next_{var}"] = _get_stochastic_next_function(
                raw_func=raw_functions[f"next_{var}"],
                grid=grids[var],
            )

            raw_functions[f"weight_next_{var}"] = _get_stochastic_weight_function(
                raw_func=raw_functions[f"next_{var}"],
                name=var,
                variable_info=variable_info,
            )

    # ==================================================================================
    # Add 'params' argument to functions
    # ==================================================================================
    # We wrap the user functions such that they can be called with the 'params' argument
    # instead of the individual parameters. This is done for all functions except for
    # the dynamically generated weighting functions for stochastic next functions, since
    # they are constructed to accept the 'params' argument by default.

    functions: dict[str, InternalUserFunction] = {}

    for func_name, func in raw_functions.items():
        is_weight_next_function = func_name.startswith("weight_next_")

        if is_weight_next_function:
            processed_func = cast("InternalUserFunction", func)

        # params[name] contains the dictionary of parameters for the function, which
        # is empty if the function does not depend on any model parameters.
        elif params[func_name]:
            processed_func = _replace_func_parameters_by_params(
                func=func,
                params=params,
                name=func_name,
            )

        else:
            processed_func = _add_dummy_params_argument(func)

        functions[func_name] = processed_func

    return functions


def _replace_func_parameters_by_params(
    func: UserFunction, params: ParamsDict, name: str
) -> InternalUserFunction:
    annotations = {
        k: v for k, v in get_annotations(func).items() if k not in params[name]
    }
    annotations_with_params = annotations | {"params": "ParamsDict"}
    return_annotation = annotations_with_params.pop("return")

    @with_signature(args=annotations_with_params, return_annotation=return_annotation)
    @functools.wraps(func)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:
        return func(*args, **kwargs, **params[name])

    return cast("InternalUserFunction", processed_func)


def _add_dummy_params_argument(func: UserFunction) -> InternalUserFunction:
    annotations = get_annotations(func) | {"params": "ParamsDict"}
    return_annotation = annotations.pop("return")

    @with_signature(args=annotations, return_annotation=return_annotation)
    @functools.wraps(func)
    def processed_func(*args: Array, params: ParamsDict, **kwargs: Array) -> Array:  # noqa: ARG001
        return func(*args, **kwargs)

    return cast("InternalUserFunction", processed_func)


def _get_stochastic_next_function(raw_func: UserFunction, grid: Int1D) -> UserFunction:
    annotations = get_annotations(raw_func)
    annotations.pop("return")

    @with_signature(args=annotations, return_annotation="Int1D")
    @functools.wraps(raw_func)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return grid

    return next_func


def _get_stochastic_weight_function(
    raw_func: UserFunction, name: str, variable_info: pd.DataFrame
) -> InternalUserFunction:
    """Get a function that returns the transition weights of a stochastic variable.

    Example:
    Consider a stochastic variable 'health' that takes two values {0, 1}. The transition
    matrix is thus 2x2. We create the weighting function and then select the weights
    that correspond to the case where 'health' is 0.

    >>> from lcm.mark import StochasticInfo
    >>> def next_health(health):
    >>>     pass
    >>> next_health._stochastic_info = StochasticInfo()
    >>> params = {"shocks": {"health": np.arange(4).reshape(2, 2)}}
    >>> weight_func = _get_stochastic_weight_function(
    >>>     raw_func=next_health,
    >>>     name="health"
    >>>     variable_info=variable_info,
    >>>     grids=grids,
    >>> )
    >>> weight_func(health=0, params=params)
    >>> array([0, 1])


    Args:
        raw_func: The raw next function of the stochastic variable.
        name: The name of the stochastic variable.
        variable_info: A table with information about model variables.

    Returns:
        A function that returns the transition weights of the stochastic variable.

    """
    function_parameters = list(inspect.signature(raw_func).parameters)

    # Assert that stochastic next function only depends on discrete variables or period
    invalid = {
        arg
        for arg in function_parameters
        if arg != "_period" and not variable_info.loc[arg, "is_discrete"]
    }

    if invalid:
        raise ValueError(
            "Stochastic variables can only depend on discrete variables and '_period', "
            f"but {name} depends on {invalid}.",
        )

    annotations = get_annotations(raw_func) | {"params": "ParamsDict"}
    annotations.pop("return")

    @with_signature(args=annotations, return_annotation="FloatND")
    def weight_func(
        params: ParamsDict, **kwargs: DiscreteState | DiscreteAction | int
    ) -> FloatND:
        args = convert_kwargs_to_args(kwargs, parameters=function_parameters)
        return params["shocks"][name][*args]

    return cast("InternalUserFunction", weight_func)
