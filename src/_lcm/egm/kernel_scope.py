"""Build-time scope checks for the DC-EGM kernel.

A validated DC-EGM regime always builds a `Model` successfully; features the
kernel does not cover yet are reported here, per period and target, so the
kernel build can install a step that raises `NotImplementedError` at solve
time with a precise message. These checks read the processed kernel-build
context (period carry targets, qualified flat params, the regime's
`VInterpolationInfo`, the processed transition functions), which the
model-time `validation` module does not have — so they live alongside the
kernel build rather than with the model-construction validators.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from _lcm.egm.regime_introspection import (
    _concatenate_regime_function,
    _get_child_discrete_actions,
    _get_child_resources_arg_names,
    _get_child_state_name,
    _get_process_state_names,
)
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.functools import get_union_of_args
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM


def _find_unsupported_feature(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
    asset_row_mode: bool,
) -> str | None:
    """Return a message naming the first feature outside the kernel's scope.

    Returns `None` when the configuration is fully supported.
    """
    message: str | None = None
    for target in carry_targets:
        message = _find_unsupported_target_feature(
            target=target,
            user_regimes=user_regimes,
            functions=functions,
            transitions=transitions,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            own_discrete_state_names=own_discrete_state_names,
            allowed_param_names=flat_param_names | regime_to_flat_param_names[target],
        )
        if message is not None:
            break

    if message is None:
        message = _find_unsupported_function_args(
            solver=solver,
            functions=functions,
            constraints=constraints,
            carry_targets=carry_targets,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_passive_state_names=own_passive_state_names,
            own_discrete_action_names=own_discrete_action_names,
            asset_row_mode=asset_row_mode,
        )

    if message is None:
        return None
    return (
        f"The DC-EGM solver cannot solve regime '{regime_name}' yet: {message} "
        "This configuration is outside the DC-EGM kernel's current scope."
    )


def _find_unsupported_target_feature(
    *,
    target: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    own_discrete_state_names: tuple[StateName, ...],
    allowed_param_names: frozenset[str],
) -> str | None:
    """Return a message naming the first unsupported feature of one target.

    `allowed_param_names` is the union of the source regime's flat params and
    the target regime's flat params: a cross-regime carry evaluates the
    target's resources / transition functions, which read the *target's*
    params (e.g. a pension factor the source never reads), so admitting the
    target's params mirrors the kernel's runtime reach.
    """
    target_info = regime_to_v_interpolation_info[target]
    target_process_states = _get_process_state_names(v_interpolation_info=target_info)
    if user_regimes[target].terminal:
        terminal_message = _find_unsupported_terminal_target_feature(
            target=target,
            user_regime=user_regimes[target],
            target_info=target_info,
            target_transitions=transitions[target],
            own_discrete_state_names=own_discrete_state_names,
        )
        if terminal_message is not None:
            return terminal_message
    for process_name in target_process_states:
        # The child's node distribution comes from the intrinsic transition
        # of the shared process state; without it (the source regime does
        # not carry the process) there is nothing to weight the child's
        # node axis with.
        has_transition = f"next_{process_name}" in transitions[target]
        has_weights = f"weight_{target}__next_{process_name}" in functions
        if not (has_transition and has_weights):
            return (
                f"the process state '{process_name}' of target regime "
                f"'{target}' has no intrinsic transition from this regime "
                "(both regimes must carry the same process state)."
            )
    child_state_name = _get_child_state_name(user_regime=user_regimes[target])
    resources_arg_names = _get_child_resources_arg_names(
        user_regime=user_regimes[target]
    )
    child_action_names, _ = _get_child_discrete_actions(
        user_regime=user_regimes[target]
    )
    # Mirror the function-args allowance (`_find_unsupported_function_args`):
    # beyond the child's states and discrete actions, the resources function
    # may read the regime's flat params and the lifecycle `age` / `period`.
    # Solve-phase imputed intermediates (a carried state's solve law) are
    # baked into the resources DAG, so their leaf inputs (the passive states
    # and params they read) are what surfaces here — the imputed output is
    # computed, never demanded as a leaf.
    allowed_resources_args = (
        {child_state_name}
        | set(target_info.state_names)
        | set(child_action_names)
        | set(allowed_param_names)
        | {"age", "period"}
    )
    extra_resources_args = sorted(resources_arg_names - allowed_resources_args)
    if extra_resources_args:
        return (
            f"the resources function of target regime '{target}' depends on "
            f"{extra_resources_args}; beyond the Euler state it may read "
            "only the child's states and discrete actions, the regime's "
            "params, and age/period."
        )
    for process_name in target_process_states:
        grid = target_info.discrete_states[process_name]
        if (
            process_name in resources_arg_names
            and not cast("_ContinuousStochasticProcess", grid).is_fully_specified
        ):
            return (
                f"the resources function of target regime '{target}' reads "
                f"the process state '{process_name}', whose grid points are "
                "supplied at runtime."
            )
    return None


def _find_unsupported_terminal_target_feature(
    *,
    target: RegimeName,
    user_regime: UserRegime,
    target_info: VInterpolationInfo,
    target_transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    own_discrete_state_names: tuple[StateName, ...],
) -> str | None:
    """Return a message naming the first unsupported feature of a terminal target.

    A terminal carry covers exactly one continuous state and no actions. It
    may additionally carry discrete states, but only those that are *shared*
    with the parent's own discrete combo axes and reached by the *identity*
    (fixed) transition: at a parent combo with the state at value $k$ the
    terminal carry row is its utility evaluated at $k$, so the parent selects
    its own combo by integer indexing the carry's leading axis. A terminal
    discrete state that the parent does not carry has no parent axis to align
    to; one reached by a stochastic or non-identity transition is a separate,
    unsupported gap.
    """
    own_discrete = set(own_discrete_state_names)
    for name, grid in target_info.discrete_states.items():
        if isinstance(grid, _ContinuousStochasticProcess):
            return (
                f"its terminal target regime '{target}' has the process state "
                f"'{name}'; terminal carries cover discrete states shared with "
                "the parent via an identity transition only."
            )
        if name not in own_discrete:
            return (
                f"its terminal target regime '{target}' has the discrete state "
                f"'{name}', which the parent does not carry; a terminal carry's "
                "discrete states must be shared with the parent's own discrete "
                "combo axes."
            )
        transition = target_transitions.get(f"next_{name}")
        if not getattr(transition, "_is_auto_identity", False):
            return (
                f"its terminal target regime '{target}' is reached by a "
                f"non-identity transition into the discrete state '{name}'; a "
                "terminal carry's discrete states must be shared with the "
                "parent via an identity (fixed) transition."
            )
    n_continuous = len(target_info.continuous_states)
    if n_continuous != 1:
        return (
            f"its terminal target regime '{target}' has continuous states "
            f"{list(target_info.continuous_states)}; exactly one continuous "
            "state is supported."
        )
    if user_regime.actions:
        return (
            f"its terminal target regime '{target}' has actions "
            f"{list(user_regime.actions)}, so its carry is not "
            "its utility on the state grid."
        )
    return None


def _find_unsupported_function_args(
    *,
    solver: DCEGM,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
    asset_row_mode: bool,
) -> str | None:
    """Return a message naming the first function with out-of-scope arguments."""
    # Combo inputs are bound per (discrete state, passive node, discrete
    # action) combination, so any of them may feed these functions.
    allowed_combo_inputs = (
        set(own_discrete_state_names)
        | set(own_passive_state_names)
        | set(own_discrete_action_names)
    )
    # In asset-row mode the combo pool carries the Euler node's value, so
    # savings-stage functions (regime transition probabilities, transition
    # weights) may read the Euler state; the single-post-state kernel has no
    # Euler value in the pool.
    allowed_savings_stage_inputs = allowed_combo_inputs | (
        {solver.continuous_state} if asset_row_mode else set()
    )
    allowed_params = flat_param_names | {"age", "period"}
    utility_func = _concatenate_regime_function(functions=functions, target="utility")
    arg_requirements: list[tuple[str, frozenset[str], set[str]]] = [
        (
            "the utility function",
            frozenset(get_union_of_args([utility_func])),
            {solver.continuous_action} | allowed_combo_inputs | allowed_params,
        ),
        (
            "the regime transition probability function",
            frozenset(get_union_of_args([compute_regime_transition_probs])),
            allowed_savings_stage_inputs | allowed_params,
        ),
    ]
    # Intrinsic process-weight functions are evaluated per combo at the
    # savings-node stage, mirroring the savings-stage independence the
    # validation requires of every other stochastic weight function.
    for target in carry_targets:
        target_process_states = _get_process_state_names(
            v_interpolation_info=regime_to_v_interpolation_info[target]
        )
        for process_name in target_process_states:
            weight_key = f"weight_{target}__next_{process_name}"
            if weight_key not in functions:
                continue
            weight_func = _concatenate_regime_function(
                functions=functions, target=weight_key
            )
            arg_requirements.append(
                (
                    f"the transition-weight function '{weight_key}'",
                    frozenset(get_union_of_args([weight_func])),
                    allowed_savings_stage_inputs | allowed_params,
                )
            )
    for constraint_name in constraints:
        constraint_func = _concatenate_regime_function(
            functions=MappingProxyType({**dict(functions), **dict(constraints)}),
            target=constraint_name,
        )
        arg_requirements.append(
            (
                f"the constraint '{constraint_name}'",
                frozenset(get_union_of_args([constraint_func])),
                allowed_combo_inputs | allowed_params,
            )
        )
    for label, needed, allowed in arg_requirements:
        extra = sorted(needed - allowed)
        if extra:
            return f"{label} depends on {extra}."
    return None
