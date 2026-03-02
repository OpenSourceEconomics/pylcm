import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from dags.signature import rename_arguments, with_signature
from dags.tree import QNAME_DELIMITER, flatten_to_qnames, unflatten_from_qnames
from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.grid_helpers import get_irreg_coordinate
from lcm.grids import DiscreteMarkovGrid, Grid
from lcm.input_processing.create_regime_params_template import (
    add_runtime_grid_params,
    create_regime_params_template,
)
from lcm.input_processing.params_processing import validate_params_template
from lcm.input_processing.process_transitions import collect_all_regime_functions
from lcm.input_processing.regime_components import (
    build_argmax_and_max_Q_over_a_functions,
    build_max_Q_over_a_functions,
    build_next_state_simulation_functions,
    build_Q_and_F_functions,
    build_regime_transition_probs_functions,
)
from lcm.input_processing.util import (
    get_gridspecs,
    get_variable_info,
)
from lcm.interfaces import InternalFunctions, InternalRegime
from lcm.ndimage import map_coordinates
from lcm.regime import MarkovRegimeTransition, Regime
from lcm.shocks import _ShockGrid
from lcm.state_action_space import create_state_action_space, create_state_space_info
from lcm.typing import (
    REGIME_PAIR_SEPARATOR,
    Float1D,
    Int1D,
    InternalUserFunction,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    UserFunction,
)
from lcm.utils import ensure_containers_are_immutable, ensure_containers_are_mutable


def process_regimes(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> tuple[MappingProxyType[RegimeName, InternalRegime], ParamsTemplate]:
    """Process user regimes into internal regimes.

    Extracts state transitions from grid `transition` attributes and
    regime transitions from `regime.transition`. For fixed states (grids
    without a transition), an identity transition is auto-generated. ShockGrid
    transitions are generated from the grid's intrinsic transition logic.

    Args:
        regimes: Mapping of regime names to Regime instances.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping from regime names to integer indices.
        enable_jit: Whether to jit the functions of the internal regime.

    Returns:
        Tuple of the processed regimes and the parameter template.

    """
    # ----------------------------------------------------------------------------------
    # Extract transitions and collect all functions per regime
    # ----------------------------------------------------------------------------------
    all_regime_functions, transition_keys_per_regime = collect_all_regime_functions(
        regimes
    )

    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    variable_info = MappingProxyType(
        {
            n: get_variable_info(r, user_functions=all_regime_functions[n])
            for n, r in regimes.items()
        }
    )
    gridspecs = MappingProxyType(
        {
            n: get_gridspecs(r, user_functions=all_regime_functions[n])
            for n, r in regimes.items()
        }
    )
    grids = MappingProxyType(
        {
            n: MappingProxyType(
                {name: spec.to_jax() for name, spec in gridspecs[n].items()}
            )
            for n in regimes
        }
    )

    state_space_infos = MappingProxyType(
        {
            n: create_state_space_info(
                variable_info=variable_info[n],
                gridspecs=gridspecs[n],
                terminal=r.terminal,
            )
            for n, r in regimes.items()
        }
    )
    state_action_spaces = MappingProxyType(
        {
            n: create_state_action_space(variable_info=variable_info[n], grids=grids[n])
            for n in regimes
        }
    )
    regime_names_to_active_periods = MappingProxyType(
        {n: ages.get_periods_where(r.active) for n, r in regimes.items()}
    )

    # ----------------------------------------------------------------------------------
    # Stage 2: Build params template with boundary-encoded transition keys
    # ----------------------------------------------------------------------------------
    params_template_dict: dict[str, dict[str, dict[str, type | tuple[int, ...]]]] = {}

    for name, regime in regimes.items():
        all_funcs = all_regime_functions[name]
        trans_keys = transition_keys_per_regime[name]

        # Non-transition functions + next_regime → keyed by regime name
        non_trans = {
            k: v
            for k, v in all_funcs.items()
            if k not in trans_keys or k == "next_regime"
        }
        regime_template = ensure_containers_are_mutable(
            create_regime_params_template(regime, user_functions=non_trans)
        )
        add_runtime_grid_params(regime, regime_template)  # ty: ignore[invalid-argument-type]
        params_template_dict[name] = regime_template  # ty: ignore[invalid-assignment]

        # Boundary transitions → keyed by pair "source_to_target"
        for func_name in trans_keys:
            if QNAME_DELIMITER not in func_name:
                continue  # next_regime handled above
            target, param_key = func_name.split(QNAME_DELIMITER, 1)
            pair = f"{name}{REGIME_PAIR_SEPARATOR}{target}"
            pair_entry = params_template_dict.setdefault(pair, {})
            pair_template = dict(
                create_regime_params_template(
                    regime,
                    user_functions={param_key: all_funcs[func_name]},
                )
            )
            pair_entry.update(pair_template)  # ty: ignore[no-matching-overload]

    params_template: ParamsTemplate = ensure_containers_are_immutable(  # ty: ignore[invalid-assignment]
        params_template_dict
    )

    validate_params_template(params_template)

    # Compute model-level flat param names (shared across all regimes)
    flat_param_names = frozenset(flatten_to_qnames(params_template))

    internal_regimes = {}
    for name, regime in regimes.items():
        internal_functions = _get_internal_functions(
            regime_name=name,
            regime=regime,
            all_functions=all_regime_functions[name],
            transition_keys=transition_keys_per_regime[name],
            grids=grids,
            regime_names_to_ids=regime_names_to_ids,
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            enable_jit=enable_jit,
            params_template=params_template,
            all_flat_param_names=flat_param_names,
        )

        Q_and_F_functions = build_Q_and_F_functions(
            regime_name=name,
            regime=regime,
            regime_names_to_active_periods=regime_names_to_active_periods,
            internal_functions=internal_functions,
            state_space_infos=state_space_infos,
            ages=ages,
            flat_param_names=flat_param_names,
        )
        max_Q_over_a_functions = build_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_functions,
            enable_jit=enable_jit,
        )
        argmax_and_max_Q_over_a_functions = build_argmax_and_max_Q_over_a_functions(
            state_action_space=state_action_spaces[name],
            Q_and_F_functions=Q_and_F_functions,
            enable_jit=enable_jit,
        )
        next_state_simulation_function = build_next_state_simulation_functions(
            internal_functions=internal_functions,
            grids=grids,
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            flat_param_names=flat_param_names,
            enable_jit=enable_jit,
        )

        # ------------------------------------------------------------------------------
        # Collect all components into the internal regime
        # ------------------------------------------------------------------------------
        internal_regimes[name] = InternalRegime(
            name=name,
            terminal=regime.terminal,
            grids=grids[name],
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            functions=MappingProxyType(internal_functions.functions),
            constraints=MappingProxyType(internal_functions.constraints),
            active_periods=tuple(regime_names_to_active_periods[name]),
            regime_transition_probs=internal_functions.regime_transition_probs,
            internal_functions=internal_functions,
            transitions=internal_functions.transitions,
            flat_param_names=flat_param_names,
            state_space_info=state_space_infos[name],
            max_Q_over_a_functions=MappingProxyType(max_Q_over_a_functions),
            argmax_and_max_Q_over_a_functions=MappingProxyType(
                argmax_and_max_Q_over_a_functions
            ),
            next_state_simulation_function=next_state_simulation_function,
            _base_state_action_space=state_action_spaces[name],
        )

    return ensure_containers_are_immutable(internal_regimes), params_template


def _get_internal_functions(
    *,
    regime_name: str,
    regime: Regime,
    all_functions: dict[str, UserFunction],
    transition_keys: frozenset[str],
    grids: MappingProxyType[RegimeName, MappingProxyType[str, Array]],
    regime_names_to_ids: RegimeNamesToIds,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    enable_jit: bool,
    params_template: ParamsTemplate,
    all_flat_param_names: frozenset[str],
) -> InternalFunctions:
    """Process the user provided regime functions.

    Args:
        regime_name: The name of the regime.
        regime: The regime as provided by the user.
        all_functions: Flat mapping of all functions for this regime, including
            boundary-encoded transitions (e.g., `"retired__next_wealth"`).
        transition_keys: Frozenset of keys in `all_functions` that are transitions.
        grids: Immutable mapping of regime names to grid arrays.
        regime_names_to_ids: Mapping from regime names to integer indices.
        gridspecs: The specifications of the current regimes grids.
        variable_info: Variable info of the regime.
        enable_jit: Whether to jit the internal functions.
        params_template: Immutable mapping of all template keys to their
            parameter templates.
        all_flat_param_names: Model-level frozenset of all flat parameter names
            across all regimes.

    Returns:
        The processed regime functions.

    """
    flat_grids = flatten_to_qnames(grids)

    # ==================================================================================
    # Rename function parameters to qualified names
    # ==================================================================================
    # Compute stochastic state names from grid types
    markov_state_names = {
        name for name, grid in gridspecs.items() if isinstance(grid, DiscreteMarkovGrid)
    }
    shock_state_names = set(variable_info.query("is_shock").index.tolist())
    stochastic_transition_names = frozenset(
        f"next_{name}" for name in markov_state_names | shock_state_names
    )

    stochastic_transition_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name in transition_keys
        and func_name.split(QNAME_DELIMITER)[-1] in stochastic_transition_names
        and func_name != "next_regime"
    }

    deterministic_transition_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name in transition_keys
        and func_name not in stochastic_transition_functions
    }

    deterministic_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name not in stochastic_transition_functions
        and func_name not in deterministic_transition_functions
    }

    functions: dict[str, InternalUserFunction] = {}

    for func_name, func in deterministic_functions.items():
        functions[func_name] = _rename_params_to_qnames(
            func=func,
            params_template=params_template,
            param_key=func_name,
            template_key=regime_name,
        )

    for func_name, func in deterministic_transition_functions.items():
        if QNAME_DELIMITER in func_name:
            target = func_name.split(QNAME_DELIMITER, 1)[0]
            param_key = func_name.split(QNAME_DELIMITER, 1)[1]
            pair_key = f"{regime_name}{REGIME_PAIR_SEPARATOR}{target}"
        else:
            param_key = func_name  # "next_regime"
            pair_key = regime_name
        functions[func_name] = _rename_params_to_qnames(
            func=func,
            params_template=params_template,
            param_key=param_key,
            template_key=pair_key,
        )

    for func_name, func in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        if QNAME_DELIMITER in func_name:
            target = func_name.split(QNAME_DELIMITER, 1)[0]
            param_key = func_name.split(QNAME_DELIMITER, 1)[1]
            pair_key = f"{regime_name}{REGIME_PAIR_SEPARATOR}{target}"
        else:
            param_key = func_name
            pair_key = regime_name
        functions[f"weight_{func_name}"] = _rename_params_to_qnames(
            func=func,
            params_template=params_template,
            param_key=param_key,
            template_key=pair_key,
        )
        functions[func_name] = _get_discrete_markov_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")],
        )
    for shock_name in variable_info.query("is_shock").index.tolist():
        relative_name = f"{regime_name}__next_{shock_name}"
        functions[f"weight_{relative_name}"] = _get_weights_func_for_shock(
            regime_name=regime_name,
            name=shock_name,
            gridspec=cast("_ShockGrid", gridspecs[shock_name]),
        )
        functions[relative_name] = _get_stochastic_next_function_for_shock(
            name=shock_name,
            grid=flat_grids[relative_name.replace("next_", "")],
        )

    # Build flat transition keys for unflatten
    flat_transition_keys = {k for k in transition_keys if k != "next_regime"}
    internal_transition = {
        func_name: functions[func_name] for func_name in flat_transition_keys
    }
    internal_constraints = MappingProxyType(
        {func_name: functions[func_name] for func_name in regime.constraints}
    )
    excluded_from_functions = flat_transition_keys | set(regime.constraints)
    internal_functions = MappingProxyType(
        {
            func_name: functions[func_name]
            for func_name in functions
            if func_name not in excluded_from_functions
        }
    )
    is_stochastic_regime_transition = isinstance(
        regime.transition, MarkovRegimeTransition
    )

    if regime.terminal:
        internal_regime_transition_probs = None
    else:
        internal_regime_transition_probs = build_regime_transition_probs_functions(
            internal_functions=internal_functions,
            regime_transition_probs=functions["next_regime"],
            grids=grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=all_flat_param_names,
            is_stochastic=is_stochastic_regime_transition,
            enable_jit=enable_jit,
        )
    return InternalFunctions(
        functions=internal_functions,
        constraints=internal_constraints,
        transitions=ensure_containers_are_immutable(
            unflatten_from_qnames(internal_transition)
        ),
        regime_transition_probs=internal_regime_transition_probs,
        stochastic_transition_names=stochastic_transition_names,
    )


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    params_template: ParamsTemplate,
    param_key: str,
    template_key: str,
) -> InternalUserFunction:
    """Rename function params to regime-prefixed qualified names.

    E.g., `risk_aversion` -> `working__utility__risk_aversion`.

    Args:
        func: The user function.
        params_template: Parameter template for all regimes.
        param_key: The key to look up in the template_key's parameter template
            (e.g., "utility").
        template_key: The top-level template key (e.g., "working" or
            "working_to_retired").

    Returns:
        The function with renamed parameters.

    """
    template_entry = params_template.get(template_key, {})
    param_entry = template_entry.get(param_key, {})
    param_names = list(param_entry)
    if not param_names:
        return cast("InternalUserFunction", func)
    mapper = {
        p: f"{template_key}{QNAME_DELIMITER}{param_key}{QNAME_DELIMITER}{p}"
        for p in param_names
    }
    return cast("InternalUserFunction", rename_arguments(func, mapper=mapper))


def _get_discrete_markov_next_function(
    *, func: UserFunction, grid: Int1D
) -> UserFunction:
    @with_signature(args=None, return_annotation="Int1D")
    @functools.wraps(func)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ANN401, ARG001
        return grid

    return next_func


def _get_stochastic_next_function_for_shock(
    *, name: str, grid: Float1D
) -> UserFunction:
    """Get function that returns the indices in the vf arr of the next shock states."""

    @with_signature(args={f"{name}": "ContinuousState"}, return_annotation="Int1D")
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0])

    return next_func


def _get_weights_func_for_shock(
    *, regime_name: str, name: str, gridspec: _ShockGrid
) -> UserFunction:
    """Get function that uses linear interpolation to calculate the shock weights.

    For shocks whose params are supplied at runtime, the grid points and transition
    probabilities are computed inside JIT from those runtime params.

    """
    if gridspec.params_to_pass_at_runtime:
        n_points = gridspec.n_points
        fixed_params = dict(gridspec.params)
        runtime_param_names = {
            f"{regime_name}{QNAME_DELIMITER}{name}{QNAME_DELIMITER}{p}": p
            for p in gridspec.params_to_pass_at_runtime
        }
        args = {name: "ContinuousState", **dict.fromkeys(runtime_param_names, "float")}

        _compute_gridpoints = gridspec.compute_gridpoints
        _compute_transition_probs = gridspec.compute_transition_probs

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
            shock_kw = {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            grid_points = _compute_gridpoints(n_points, **shock_kw)
            transition_probs = _compute_transition_probs(n_points, **shock_kw)
            coord = get_irreg_coordinate(value=kwargs[name], points=grid_points)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points),
                ],
            )

        return weights_func_runtime

    grid_points = gridspec.get_gridpoints()
    transition_probs = gridspec.get_transition_probs()

    @with_signature(
        args={f"{name}": "ContinuousState"},
        return_annotation="FloatND",
        enforce=False,
    )
    def weights_func(*args: Array, **kwargs: Array) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(value=kwargs[f"{name}"], points=grid_points)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(gridspec.n_points, fill_value=coordinate),
                jnp.arange(gridspec.n_points),
            ],
        )

    return weights_func
