import functools
import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
from dags.signature import rename_arguments, with_signature
from dags.tree import QNAME_DELIMITER
from jax import Array
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.grid_helpers import get_irreg_coordinate
from lcm.grids import DiscreteMarkovGrid, Grid, _DiscreteGridBase
from lcm.input_processing.create_regime_params_template import (
    create_regime_params_template,
)
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
from lcm.regime import MarkovRegimeTransition, Regime, _make_identity_fn
from lcm.shocks import _ShockGrid
from lcm.state_action_space import create_state_action_space, create_state_space_info
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    Float1D,
    Int1D,
    InternalUserFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    TransitionFunctionsMapping,
    UserFunction,
)
from lcm.utils import (
    ensure_containers_are_immutable,
    flatten_regime_namespace,
    unflatten_regime_namespace,
)


def _wrap_transitions(
    transitions: dict[RegimeName, dict[str, InternalUserFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def process_regimes(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, InternalRegime]:
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
        The processed regimes.

    """

    # ----------------------------------------------------------------------------------
    # Extract transitions from grid attributes and regime transition
    # ----------------------------------------------------------------------------------
    states_per_regime: dict[str, set[str]] = {
        name: set(regime.states.keys()) for name, regime in regimes.items()
    }

    nested_transitions = {}
    target_originated_per_regime: dict[str, frozenset[str]] = {}
    for name, regime in regimes.items():
        transitions, target_originated = _extract_transitions_from_regime(
            regime=regime,
            regime_name=name,
            all_regimes=regimes,
            states_per_regime=states_per_regime,
        )
        nested_transitions[name] = transitions
        target_originated_per_regime[name] = target_originated
    # ----------------------------------------------------------------------------------
    # Stage 1: Initialize regime components that do not depend on other regimes
    # ----------------------------------------------------------------------------------
    variable_info = MappingProxyType(
        {n: get_variable_info(r) for n, r in regimes.items()}
    )
    gridspecs = MappingProxyType({n: get_gridspecs(r) for n, r in regimes.items()})
    grids = MappingProxyType(
        {
            n: MappingProxyType(
                {name: spec.to_jax() for name, spec in gridspecs[n].items()}
            )
            for n in regimes
        }
    )

    state_space_infos = MappingProxyType(
        {n: create_state_space_info(r) for n, r in regimes.items()}
    )
    state_action_spaces = MappingProxyType(
        {
            n: create_state_action_space(variable_info=variable_info[n], grids=grids[n])
            for n in regimes
        }
    )
    regimes_to_active_periods = MappingProxyType(
        {n: ages.get_periods_where(r.active) for n, r in regimes.items()}
    )

    # ----------------------------------------------------------------------------------
    # Stage 2: Initialize regime components that depend on other regimes
    # ----------------------------------------------------------------------------------
    all_regime_params_templates = MappingProxyType(
        {
            name: create_regime_params_template(regime)
            for name, regime in regimes.items()
        }
    )

    internal_regimes = {}
    for name, regime in regimes.items():
        regime_params_template = all_regime_params_templates[name]

        internal_functions, cross_boundary_params = _get_internal_functions(
            regime=regime,
            regime_name=name,
            nested_transitions=nested_transitions[name],
            grids=grids,
            regime_params_template=regime_params_template,
            regime_names_to_ids=regime_names_to_ids,
            gridspecs=gridspecs[name],
            variable_info=variable_info[name],
            enable_jit=enable_jit,
            all_regime_params_templates=all_regime_params_templates,
            target_originated_transitions=target_originated_per_regime[name],
        )

        cross_boundary_param_names = frozenset(cross_boundary_params.keys())

        Q_and_F_functions = build_Q_and_F_functions(
            regime_name=name,
            regime=regime,
            regimes_to_active_periods=regimes_to_active_periods,
            internal_functions=internal_functions,
            state_space_infos=state_space_infos,
            ages=ages,
            regime_params_template=regime_params_template,
            cross_boundary_param_names=cross_boundary_param_names,
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
            regime_params_template=regime_params_template,
            enable_jit=enable_jit,
            cross_boundary_param_names=cross_boundary_param_names,
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
            active_periods=tuple(regimes_to_active_periods[name]),
            regime_transition_probs=internal_functions.regime_transition_probs,
            internal_functions=internal_functions,
            transitions=internal_functions.transitions,
            regime_params_template=regime_params_template,
            state_space_info=state_space_infos[name],
            max_Q_over_a_functions=MappingProxyType(max_Q_over_a_functions),
            argmax_and_max_Q_over_a_functions=MappingProxyType(
                argmax_and_max_Q_over_a_functions
            ),
            next_state_simulation_function=next_state_simulation_function,
            _base_state_action_space=state_action_spaces[name],
            cross_boundary_params=cross_boundary_params,
        )

    return ensure_containers_are_immutable(internal_regimes)


def _get_internal_functions(
    *,
    regime: Regime,
    regime_name: str,
    nested_transitions: dict[str, dict[str, UserFunction] | UserFunction],
    grids: MappingProxyType[RegimeName, MappingProxyType[str, Array]],
    regime_params_template: RegimeParamsTemplate,
    regime_names_to_ids: RegimeNamesToIds,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    enable_jit: bool,
    all_regime_params_templates: MappingProxyType[RegimeName, RegimeParamsTemplate],
    target_originated_transitions: frozenset[str],
) -> tuple[InternalFunctions, MappingProxyType[str, tuple[str, str]]]:
    """Process the user provided regime functions.

    Args:
        regime: The regime as provided by the user.
        regime_name: The name of the regime.
        nested_transitions: Nested transitions dict for internal processing.
            Format: {"regime_name": {"next_state": func, ...}, "next_regime": func}
        grids: Immutable mapping of regime names to grid arrays.
        regime_params_template: The regime's parameter template.
        regime_names_to_ids: Mapping from regime names to integer indices.
        gridspecs: The specifications of the current regimes grids.
        variable_info: Variable info of the regime.
        enable_jit: Whether to jit the internal functions.
        all_regime_params_templates: Immutable mapping of all regime names to their
            parameter templates.
        target_originated_transitions: Frozenset of flat function names whose
            transitions were resolved from the target grid's mapping.

    Returns:
        Tuple of the processed regime functions and an immutable mapping from
        cross-boundary qualified param names to `(target_regime, target_qname)`
        tuples.

    """
    flat_grids = flatten_regime_namespace(grids)
    # Flatten nested transitions to get prefixed names like "regime__next_wealth"
    flat_nested_transitions = flatten_regime_namespace(nested_transitions)

    # ==================================================================================
    # Rename function parameters to qualified names
    # ==================================================================================
    # We rename user function parameters to qualified names (e.g.,
    # risk_aversion -> utility__risk_aversion) so they can be matched with flat params.

    # Build all_functions using nested_transitions (to get prefixed names)
    all_functions = {
        **regime.functions,
        **regime.constraints,
        **flat_nested_transitions,
    }

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
        for func_name, func in flat_nested_transitions.items()
        if func_name.split(QNAME_DELIMITER)[-1] in stochastic_transition_names
        and func_name != "next_regime"
    }

    deterministic_transition_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name in flat_nested_transitions
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
            regime_params_template=regime_params_template,
            param_key=func_name,
        )

    cross_boundary_params: dict[str, tuple[str, str]] = {}

    for func_name, func in deterministic_transition_functions.items():
        functions[func_name] = _rename_transition_params(
            func=func,
            func_name=func_name,
            regime_params_template=regime_params_template,
            target_originated_transitions=target_originated_transitions,
            all_regime_params_templates=all_regime_params_templates,
            cross_boundary_params=cross_boundary_params,
        )

    for func_name, func in stochastic_transition_functions.items():
        # The user-specified next function is the weighting function for the
        # stochastic transition. For the solution, we must also define a next function
        # that returns the whole grid of possible values.
        functions[f"weight_{func_name}"] = _rename_transition_params(
            func=func,
            func_name=func_name,
            regime_params_template=regime_params_template,
            target_originated_transitions=target_originated_transitions,
            all_regime_params_templates=all_regime_params_templates,
            cross_boundary_params=cross_boundary_params,
        )
        functions[func_name] = _get_discrete_markov_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")],
        )
    for shock_name in variable_info.query("is_shock").index.tolist():
        relative_name = f"{regime_name}__next_{shock_name}"
        functions[f"weight_{relative_name}"] = _get_weights_func_for_shock(
            name=shock_name,
            gridspec=cast("_ShockGrid", gridspecs[shock_name]),
        )
        functions[relative_name] = _get_stochastic_next_function_for_shock(
            name=shock_name,
            grid=flat_grids[relative_name.replace("next_", "")],
        )
    internal_transition = {
        func_name: functions[func_name]
        for func_name in flat_nested_transitions
        if func_name != "next_regime"
    }
    internal_constraints = MappingProxyType(
        {func_name: functions[func_name] for func_name in regime.constraints}
    )
    excluded_from_functions = set(flat_nested_transitions) | set(regime.constraints)
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
            regime_params_template=regime_params_template,
            is_stochastic=is_stochastic_regime_transition,
            enable_jit=enable_jit,
        )
    return (
        InternalFunctions(
            functions=internal_functions,
            constraints=internal_constraints,
            transitions=_wrap_transitions(
                unflatten_regime_namespace(internal_transition)
            ),
            regime_transition_probs=internal_regime_transition_probs,
            stochastic_transition_names=stochastic_transition_names,
        ),
        MappingProxyType(cross_boundary_params),
    )


def _extract_transitions_from_regime(
    *,
    regime: Regime,
    regime_name: str,
    all_regimes: Mapping[str, Regime],
    states_per_regime: Mapping[str, set[str]],
) -> tuple[dict[str, dict[str, UserFunction] | UserFunction], frozenset[str]]:
    """Extract transitions from grid attributes and auto-generate identity transitions.

    For non-terminal regimes, collects state transitions from source and target grids,
    resolving per-boundary transitions using a hierarchical priority:

    1. Target grid's mapping transition with `(source, target)` key
    2. Source grid's mapping transition with `(source, target)` key
    3. Source grid's single-callable transition
    4. Source grid's `None` transition (identity)
    5. Target grid's single-callable transition
    6. Target grid's `None` transition (identity)
    7. Unlisted boundary in a mapping → identity

    Validates that discrete states with different categories across regimes have
    explicit per-boundary transitions.

    Args:
        regime: The user regime (source).
        regime_name: Name of the source regime.
        all_regimes: Mapping of all regime names to regime instances.
        states_per_regime: Mapping of regime names to their state names.

    Returns:
        Tuple of nested transitions dict and frozenset of flat function names
        (e.g., `"phase2__next_wealth"`) whose transitions were resolved from the
        target grid's mapping (priority 1).

    """
    if regime.terminal:
        return {}, frozenset()

    nested: dict[str, dict[str, UserFunction] | UserFunction] = {}
    target_originated: set[str] = set()
    # Guaranteed non-None: terminal regimes return early.
    assert regime.transition is not None  # noqa: S101
    nested["next_regime"] = regime.transition.func

    for target_name, target_state_names in states_per_regime.items():
        target_regime = all_regimes[target_name]
        boundary_key = (regime_name, target_name)
        boundary_transitions: dict[str, UserFunction] = {}
        missing_states: list[str] = []

        for state_name in target_state_names:
            result = _resolve_state_transition(
                state_name=state_name,
                boundary_key=boundary_key,
                source_regime=regime,
                target_regime=target_regime,
            )
            if isinstance(result, _Unresolved):
                missing_states.append(state_name)
            else:
                resolved, is_target_originated = result
                boundary_transitions[f"next_{state_name}"] = resolved
                if is_target_originated:
                    target_originated.add(f"{target_name}__next_{state_name}")

        if missing_states:
            continue

        _validate_discrete_category_compatibility(
            boundary_key=boundary_key,
            boundary_transitions=boundary_transitions,
            source_regime=regime,
            target_regime=target_regime,
        )
        nested[target_name] = boundary_transitions

    return nested, frozenset(target_originated)


class _Unresolved:
    """Sentinel type for unresolved transitions."""


_UNRESOLVED = _Unresolved()


def _resolve_state_transition(
    *,
    state_name: str,
    boundary_key: tuple[str, str],
    source_regime: Regime,
    target_regime: Regime,
) -> tuple[UserFunction, bool] | _Unresolved:
    """Resolve the transition function for one state in a `(source, target)` boundary.

    Priority (highest to lowest):

    1. Target grid mapping with `(source, target)` key
    2. Source grid mapping with `(source, target)` key
    3. Source grid single-callable
    4. Source grid `None` (identity)
    5. Target grid single-callable
    6. Target grid `None` (identity)
    7. Target or source has mapping but boundary not listed → identity

    Returns:
        Tuple of `(resolved_function, target_originated)` where
        `target_originated` is `True` when the function came from the target
        grid's mapping (priority 1), or the `_UNRESOLVED` sentinel.

    """
    source_grid = source_regime.states.get(state_name)
    target_grid = target_regime.states.get(state_name)

    # ShockGrids have intrinsic transitions handled separately by
    # _get_internal_functions; return a placeholder so the target stays reachable.
    if isinstance(source_grid, _ShockGrid) or isinstance(target_grid, _ShockGrid):
        return (lambda: None, False)

    source_trans = _get_grid_transition(source_grid)
    target_trans = _get_grid_transition(target_grid)

    identity = _make_identity_for_target(state_name, target_regime)

    # Priority 1-2: Mapping with this boundary key (target wins over source).
    # target_originated is True only when the target grid provided the function.
    for trans, target_originated in ((target_trans, True), (source_trans, False)):
        if isinstance(trans, Mapping) and boundary_key in trans:
            fn = trans[boundary_key]  # ty: ignore[invalid-argument-type]
            # _get_grid_transition is untyped; remove suppression once it returns
            # UserFunction | Mapping | None.
            return (identity if fn is None else fn, target_originated)  # ty: ignore[invalid-return-type]

    # Priority 3-6: Source then target — single-callable or None
    for trans in (source_trans, target_trans):
        if callable(trans):
            return trans, False
        if trans is None:
            return identity, False

    # Priority 7: Mapping exists but boundary not listed → identity
    if isinstance(target_trans, Mapping) or isinstance(source_trans, Mapping):
        return identity, False
    return _UNRESOLVED


def _get_grid_transition(grid: Grid | None) -> object:
    """Extract the raw transition attribute from a grid, or return `_UNRESOLVED`."""
    if grid is None:
        return _UNRESOLVED
    if isinstance(grid, _ShockGrid):
        return _UNRESOLVED  # Handled separately
    return getattr(grid, "transition", _UNRESOLVED)


def _make_identity_for_target(state_name: str, target_regime: Regime) -> UserFunction:
    """Create an identity transition using the target regime's type for a state."""
    ann = (
        DiscreteState
        if state_name in target_regime.states
        and isinstance(target_regime.states[state_name], _DiscreteGridBase)
        else ContinuousState
    )
    return _make_identity_fn(state_name, annotation=ann)


def _validate_discrete_category_compatibility(
    *,
    boundary_key: tuple[str, str],
    boundary_transitions: dict[str, UserFunction],
    source_regime: Regime,
    target_regime: Regime,
) -> None:
    """Validate discrete states with different categories have explicit transitions.

    Raise `ModelInitializationError` if a discrete state has different category sets
    in source and target regimes but no per-boundary transition was provided.

    """

    source_name, target_name = boundary_key
    for state_name in target_regime.states:
        source_grid = source_regime.states.get(state_name)
        target_grid = target_regime.states.get(state_name)

        if not (
            isinstance(source_grid, _DiscreteGridBase)
            and isinstance(target_grid, _DiscreteGridBase)
        ):
            continue

        if source_grid.categories == target_grid.categories:
            continue

        # Categories differ — check if an explicit per-boundary transition was provided
        next_name = f"next_{state_name}"
        transition_fn = boundary_transitions.get(next_name)
        if transition_fn is None or getattr(transition_fn, "_is_auto_identity", False):
            raise ModelInitializationError(
                f"State '{state_name}' has different discrete categories in regimes "
                f"'{source_name}' and '{target_name}' "
                f"({list(source_grid.categories)} vs {list(target_grid.categories)}) "
                f"but no per-boundary transition was provided. "
                f"Use a mapping transition on the target regime's grid, e.g.: "
                f'DiscreteGrid(..., transition={{("{source_name}", "{target_name}"): '
                f"map_fn}})"
            )


def _validate_cross_regime_transition(
    func: UserFunction,
    func_name: str,
    regime_params_template: RegimeParamsTemplate,
) -> None:
    """Validate a cross-regime transition function before casting.

    Verify that the function's parameters don't collide with model parameter
    names that would normally require qualified-name renaming. A collision
    indicates a missing entry in the regime parameter template.

    """
    try:
        func_param_names = set(inspect.signature(func).parameters)
    except ValueError, TypeError:
        return

    template_param_names = {
        p for entry in regime_params_template.values() for p in entry
    }
    overlap = func_param_names & template_param_names
    if overlap:
        msg = (
            f"Cross-regime transition '{func_name}' has parameter(s) "
            f"{sorted(overlap)} that collide with model parameter names in the "
            f"regime template. This may indicate a missing template entry."
        )
        raise ModelInitializationError(msg)


def _extract_param_key(func_name: str) -> str:
    """Extract the param template key from a possibly prefixed function name.

    For prefixed names like "work__next_wealth", returns "next_wealth".
    For unprefixed names like "next_regime", returns the name unchanged.

    """
    if QNAME_DELIMITER in func_name:
        return func_name.split(QNAME_DELIMITER, 1)[1]
    return func_name


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
) -> InternalUserFunction:
    """Rename function params to qualified names using dags.signature.rename_arguments.

    E.g., risk_aversion -> utility__risk_aversion.

    Args:
        func: The user function.
        regime_params_template: The parameter template for the regime.
        param_key: The key to look up in regime_params_template (e.g., "utility").

    Returns:
        The function with renamed parameters.

    """
    param_names = list(regime_params_template[param_key])
    if not param_names:
        return cast("InternalUserFunction", func)
    mapper = {p: f"{param_key}{QNAME_DELIMITER}{p}" for p in param_names}
    return cast("InternalUserFunction", rename_arguments(func, mapper=mapper))


def _rename_transition_params(
    *,
    func: UserFunction,
    func_name: str,
    regime_params_template: RegimeParamsTemplate,
    target_originated_transitions: frozenset[str],
    all_regime_params_templates: MappingProxyType[RegimeName, RegimeParamsTemplate],
    cross_boundary_params: dict[str, tuple[str, str]],
) -> InternalUserFunction:
    """Rename a transition function's parameters to qualified names.

    Dispatches to target-originated or source-regime renaming based on whether
    the transition was resolved from the target grid's mapping.

    """
    if func_name in target_originated_transitions:
        return _rename_target_originated_transition(
            func=func,
            func_name=func_name,
            all_regime_params_templates=all_regime_params_templates,
            cross_boundary_params=cross_boundary_params,
        )
    param_key = _extract_param_key(func_name)
    if param_key in regime_params_template:
        return _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=param_key,
        )
    _validate_cross_regime_transition(func, func_name, regime_params_template)
    return cast("InternalUserFunction", func)


def _rename_target_originated_transition(
    *,
    func: UserFunction,
    func_name: str,
    all_regime_params_templates: MappingProxyType[RegimeName, RegimeParamsTemplate],
    cross_boundary_params: dict[str, tuple[str, str]],
) -> InternalUserFunction:
    """Rename parameters of a target-originated transition to cross-boundary qnames.

    For transitions resolved from the target grid's mapping, rename the function's
    parameters using a target-prefixed qualified name (e.g.,
    `growth_rate` -> `phase2__next_wealth__growth_rate`) and record the mapping
    from cross-boundary qname to `(target_regime, target_qname)` so the values
    can be resolved from the target regime's params at runtime.

    Args:
        func: The user transition function.
        func_name: The flat function name (e.g., `"phase2__next_wealth"`).
        all_regime_params_templates: All regime parameter templates.
        cross_boundary_params: Mutable dict to populate with cross-boundary mappings.

    Returns:
        The function with renamed parameters.

    """
    target_name = func_name.split(QNAME_DELIMITER, 1)[0]
    param_key = _extract_param_key(func_name)
    target_template = all_regime_params_templates[target_name]

    assert param_key in target_template, (  # noqa: S101
        f"Target-originated transition '{func_name}' has no matching entry "
        f"'{param_key}' in the target regime's parameter template. "
        f"This indicates a bug in _discover_mapping_transition_params."
    )
    param_names = list(target_template[param_key])
    if not param_names:
        return cast("InternalUserFunction", func)
    mapper = {p: f"{func_name}{QNAME_DELIMITER}{p}" for p in param_names}
    renamed = cast("InternalUserFunction", rename_arguments(func, mapper=mapper))
    for p in param_names:
        src_qname = f"{func_name}{QNAME_DELIMITER}{p}"
        tgt_qname = f"{param_key}{QNAME_DELIMITER}{p}"
        cross_boundary_params[src_qname] = (target_name, tgt_qname)
    return renamed


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


def _get_weights_func_for_shock(*, name: str, gridspec: _ShockGrid) -> UserFunction:
    """Get function that uses linear interpolation to calculate the shock weights.

    For shocks whose params are supplied at runtime, the grid points and transition
    probabilities are computed inside JIT from those runtime params.

    """
    if gridspec.params_to_pass_at_runtime:
        n_points = gridspec.n_points
        fixed_params = dict(gridspec.params)
        runtime_param_names = {
            f"{name}{QNAME_DELIMITER}{p}": p for p in gridspec.params_to_pass_at_runtime
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
