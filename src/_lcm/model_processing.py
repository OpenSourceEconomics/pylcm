"""Model initialization helpers: validation, template creation, fixed-param handling.

Extracted from `model.py` to keep the `Model` class focused on its public API.

"""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

from dags import get_ancestors
from dags.tree import QNAME_DELIMITER, qname_from_tree_path
from jax import Array

from _lcm.egm.validation import validate_dcegm_regimes
from _lcm.grids import DiscreteGrid
from _lcm.pandas_utils import convert_series_in_params, has_series
from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
    create_params_template,
    materialize_granular_transition_params,
)
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.h_dag import get_dag_targets_consumed_by_H
from _lcm.regime_building.processing import (
    Regime,
    process_regimes,
)
from _lcm.typing import (
    FlatParams,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
)
from _lcm.utils.containers import get_field_names_and_values
from _lcm.utils.error_messages import format_messages
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, ModelInitializationError
from lcm.params import MappingLeaf
from lcm.regime import Regime as UserRegime
from lcm.typing import UserParams


def build_regimes_and_template(
    *,
    ages: AgeGrid,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, Regime], ParamsTemplate]:
    """Build canonical regimes and params template in a single pass.

    Compose regime processing, template creation, and optional fixed-param partialling
    so that each result is computed exactly once.

    Args:
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to finalized regimes.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.

    Returns:
        Tuple of (regimes, params_template).

    """
    if not fixed_params:
        regimes = process_regimes(
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            enable_jit=enable_jit,
        )
        params_template = create_params_template(regimes)
    else:
        regimes, params_template = _build_regimes_and_template_with_fixed_params(
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=fixed_params,
        )

    return regimes, params_template


def _build_regimes_and_template_with_fixed_params(
    *,
    ages: AgeGrid,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, Regime], ParamsTemplate]:
    """Build canonical regimes and template, then partial in fixed params.

    Args:
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to finalized regimes.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.

    Returns:
        Tuple of regimes and params_template with fixed params
        partialled in.

    """
    raw_regimes = process_regimes(
        ages=ages,
        user_regimes=user_regimes,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=enable_jit,
    )
    raw_params_template = create_params_template(raw_regimes)

    fixed_flat_params = _resolve_fixed_params(
        fixed_params=dict(fixed_params), template=raw_params_template
    )
    if has_series(fixed_flat_params):
        fixed_flat_params = convert_series_in_params(
            flat_params=fixed_flat_params,
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
        )
    fixed_flat_params = cast_params_to_canonical_dtypes(fixed_flat_params)
    _validate_param_types(fixed_flat_params)

    # The template trim works on the template-shaped (user-coarse) form;
    # partialling needs the granular form the compiled functions bind.
    granular_fixed_flat_params = materialize_granular_transition_params(
        flat_params=fixed_flat_params,
        expansions={
            regime_name: regime.granular_param_expansions
            for regime_name, regime in raw_regimes.items()
        },
    )

    return (
        _partial_fixed_params_into_regimes(
            raw_regimes=raw_regimes,
            fixed_flat_params=granular_fixed_flat_params,
        ),
        _remove_fixed_params_from_template(
            template=raw_params_template,
            fixed_flat_params=fixed_flat_params,
        ),
    )


def validate_model_inputs(
    *,
    n_periods: int,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_id_class: type,
    n_subjects: int | None = None,
    broadcast_variables: Mapping[RegimeName, frozenset[str]] | None = None,
) -> None:
    """Validate model constructor inputs.

    `n_periods` is derived from `Model.__init__`'s `ages: AgeGrid` and
    `regimes` is typed via beartype on `Model.__init__`; both reach this
    function with their declared types. This function focuses on value and
    cross-field rules.

    """
    _fail_if_invalid_n_subjects(n_subjects=n_subjects)

    # DC-EGM contract checks run before the generic checks below: a contract
    # violation (e.g. a missing resources function) typically also leaves
    # variables unused, and the contract-specific message is the actionable
    # one.
    validate_dcegm_regimes(user_regimes=user_regimes)

    error_messages: list[str] = []

    if n_periods <= 1:
        error_messages.append("n_periods must be at least 2.")

    if not user_regimes:
        error_messages.append(
            "At least one non-terminal and one terminal regime must be provided."
        )

    # Validate regime names don't contain separator
    invalid_names = [name for name in user_regimes if QNAME_DELIMITER in name]
    if invalid_names:
        error_messages.append(
            f"Regime names cannot contain the separator character "
            f"'{QNAME_DELIMITER}'. The following names are invalid: {invalid_names}."
        )

    # Assume all items in regimes are lcm.Regime instances beyond this point
    terminal_regimes = [name for name, r in user_regimes.items() if r.terminal]
    if len(terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one terminal regime.")

    non_terminal_regimes = {
        name: r for name, r in user_regimes.items() if not r.terminal
    }
    if len(non_terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one non-terminal regime.")

    regime_id_fields = sorted(get_field_names_and_values(regime_id_class).keys())
    regime_names = sorted(user_regimes.keys())
    if regime_id_fields != regime_names:
        error_messages.append(
            f"regime_id_cls fields must match regime names.\nGot:\n"
            "regime_id_cls fields:\n"
            f"    {regime_id_fields}\n"
            "regime names:\n"
            f"    {regime_names}."
        )
    error_messages.extend(
        _validate_all_variables_used(
            user_regimes, broadcast_variables=broadcast_variables
        )
    )

    for name, user_regime in user_regimes.items():
        if user_regime.taste_shocks is not None and not any(
            isinstance(grid, DiscreteGrid) for grid in user_regime.actions.values()
        ):
            error_messages.append(
                f"Regime '{name}' declares taste_shocks but has no discrete "
                f"action. EV1 taste shocks are drawn per discrete-action "
                f"combination, so at least one discrete action is required."
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _fail_if_invalid_n_subjects(*, n_subjects: int | None) -> None:
    """Raise TypeError if non-int, ValueError if non-positive."""
    if n_subjects is None:
        return
    # `bool` is a subclass of `int`; reject explicitly so True/False don't slip through.
    if not isinstance(n_subjects, int) or isinstance(n_subjects, bool):
        msg = f"n_subjects must be an int or None, got {type(n_subjects).__name__}."
        raise TypeError(msg)
    if n_subjects <= 0:
        msg = f"n_subjects must be a positive integer, got {n_subjects}."
        raise ValueError(msg)


def _validate_all_variables_used(
    user_regimes: Mapping[RegimeName, UserRegime],
    *,
    broadcast_variables: Mapping[RegimeName, frozenset[str]] | None = None,
) -> list[str]:
    """Validate that all states and actions are used somewhere in each regime.

    Each state or action must appear in at least one of:
    - The concurrent valuation (utility or constraints)
    - A transition function
    - A regime function whose output H consumes at the Bellman step

    Broadcast variables are exempt: DAG pruning already weeded the unused
    ones, and a retained broadcast variable may be used only through a law
    of motion toward a reachable target (which this per-regime check cannot
    see).

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        broadcast_variables: Per regime, the model-level broadcast state and
            action names to exempt.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages = []

    for regime_name, user_regime in user_regimes.items():
        variable_names = set(user_regime.states) | set(user_regime.actions)
        if broadcast_variables is not None:
            variable_names -= broadcast_variables.get(regime_name, frozenset())
        user_functions = dict(user_regime.get_all_functions(phase="solve"))

        targets = [
            "utility",
            *list(user_regime.constraints),
            *(
                name
                for name in user_functions
                if name.startswith("next_")
                and not getattr(user_functions[name], "_is_auto_identity", False)
            ),
            *get_dag_targets_consumed_by_H(user_functions),
        ]
        reachable = get_ancestors(
            user_functions, targets=targets, include_targets=False
        )
        unused_variables = sorted(variable_names - reachable)

        if unused_variables:
            unused_states = [v for v in unused_variables if v in user_regime.states]
            unused_actions = [v for v in unused_variables if v in user_regime.actions]

            msg_parts = []
            if unused_states:
                state_word = "state" if len(unused_states) == 1 else "states"
                msg_parts.append(f"{state_word} {unused_states}")
            if unused_actions:
                action_word = "action" if len(unused_actions) == 1 else "actions"
                msg_parts.append(f"{action_word} {unused_actions}")

            error_messages.append(
                f"The following variables are defined but never used in regime "
                f"'{regime_name}': {' and '.join(msg_parts)}. "
                f"Each state and action must be used in at least one of: "
                f"utility, constraints, or transition functions."
            )

    return error_messages


def _resolve_fixed_params(
    *,
    fixed_params: dict[str, object],
    template: ParamsTemplate,
) -> FlatParams:
    """Resolve fixed_params against the params template.

    Like `process_params`, support model/regime/function level specification, but
    do NOT require all template keys to be present — only match what's provided.

    """
    return broadcast_to_template(
        params=fixed_params,
        template=template,
        required=False,
    )


def _remove_fixed_params_from_template(
    *,
    template: ParamsTemplate,
    fixed_flat_params: FlatParams,
) -> ParamsTemplate:
    """Remove fixed params from the params template.

    After partialling fixed params into compiled functions, remove them from the
    template so users don't need to supply them at solve/simulate time.

    """

    def _trim(
        *, branch: Mapping[str, object], prefix: tuple[str, ...], fixed: Mapping
    ) -> dict[str, object]:
        trimmed: dict[str, object] = {}
        for key, value in branch.items():
            if isinstance(value, Mapping):
                inner = _trim(
                    branch=cast("Mapping[str, object]", value),
                    prefix=(*prefix, key),
                    fixed=fixed,
                )
                if inner:
                    trimmed[key] = MappingProxyType(inner)
            elif qname_from_tree_path((*prefix, key)) not in fixed:
                trimmed[key] = value
        return trimmed

    return cast(
        "ParamsTemplate",
        MappingProxyType(
            {
                regime_name: MappingProxyType(
                    _trim(
                        branch=regime_template,
                        prefix=(),
                        fixed=fixed_flat_params.get(regime_name, MappingProxyType({})),
                    )
                )
                for regime_name, regime_template in template.items()
            }
        ),
    )


def _partial_fixed_params_into_regimes(
    *,
    raw_regimes: MappingProxyType[RegimeName, Regime],
    fixed_flat_params: FlatParams,
) -> MappingProxyType[RegimeName, Regime]:
    """Partial fixed params into all compiled functions on each Regime."""
    result: dict[RegimeName, Regime] = {}
    for regime_name, regime in raw_regimes.items():
        regime_fixed = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))
        # A DC-EGM source carrying into a *different* target regime evaluates
        # that target's resources / transition functions in its per-asset-node
        # solve, reading the target's fixed params (e.g. a pension factor the
        # source never reads). These are model-level shared values, so the
        # target's `fixed_flat_params` entry carries the right value; union
        # them into the params bound into the source's `egm_step` kernel. The
        # kernel threads its `**kwargs` into the per-combo pool, and the
        # captured functions read only the keys they need, so the extra
        # carry-target params are harmless to the functions that don't.
        egm_fixed = dict(regime_fixed)
        for target_name in regime.solution.transitions:
            for key, value in fixed_flat_params.get(
                target_name, MappingProxyType({})
            ).items():
                egm_fixed.setdefault(key, value)
        if not regime_fixed and not egm_fixed:
            result[regime_name] = regime
            continue

        # Build new solution phase with partialled functions. The resolved
        # fixed params also land on the phase itself — its
        # `state_action_space` consults them for runtime grid substitution.
        solution = regime.solution
        new_solve = dataclasses.replace(
            solution,
            resolved_fixed_params=MappingProxyType(regime_fixed),
            max_Q_over_a=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in solution.max_Q_over_a.items()
                }
            ),
            compute_regime_transition_probs=(
                functools.partial(
                    solution.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=solution.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if solution.compute_regime_transition_probs is not None
                else None
            ),
            # The DC-EGM kernels are prebuilt closures that capture the
            # regime's savings-stage functions (regime-transition
            # probabilities, transition weights, the child resources and
            # next-state maps) before fixed params are partialled. The kernel
            # threads its `**kwargs` straight into the per-combo pool those
            # captured functions read, so binding the union of the regime's
            # and its carry targets' fixed params here restores the params
            # removed from `flat_params` for every one of them at once —
            # matching what the live params supply.
            egm_step=(
                MappingProxyType(
                    {
                        period: functools.partial(func, **egm_fixed)
                        for period, func in solution.egm_step.items()
                    }
                )
                if solution.egm_step is not None
                else None
            ),
        )

        # Build new simulation phase with partialled functions
        simulation = regime.simulation
        new_simulate = dataclasses.replace(
            simulation,
            argmax_and_max_Q_over_a=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in simulation.argmax_and_max_Q_over_a.items()
                }
            ),
            next_state=functools.partial(simulation.next_state, **regime_fixed),
            compute_regime_transition_probs=(
                functools.partial(
                    simulation.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=simulation.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if simulation.compute_regime_transition_probs is not None
                else None
            ),
        )

        result[regime_name] = dataclasses.replace(
            regime,
            solution=new_solve,
            simulation=new_simulate,
            resolved_fixed_params=MappingProxyType(regime_fixed),
        )
    return MappingProxyType(result)


def _filter_kwargs_for_func(
    *, func: Callable, kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """Filter kwargs to only those accepted by func's signature."""
    try:
        sig = inspect.signature(func)
    except ValueError, TypeError:
        # If we can't inspect the signature, pass all kwargs through
        return kwargs
    params = sig.parameters
    # If the function accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _validate_param_types(flat_params: FlatParams) -> None:
    """Raise if any param leaf is not a JAX `Array` or container leaf.

    Defense-in-depth check after `cast_params_to_canonical_dtypes`: by the
    time this runs, every leaf must be a JAX `Array`, or a `MappingLeaf` /
    `SequenceLeaf` whose contents recursively satisfy the same rule.
    """
    for regime_name, regime_params in flat_params.items():
        for key, value in regime_params.items():
            _check_leaf(value, f"{regime_name}__{key}")


def _check_leaf(value: object, path: str) -> None:
    """Check a single leaf, recursing into `MappingLeaf` / `SequenceLeaf`."""
    if isinstance(value, MappingLeaf):
        for k, v in value.data.items():
            _check_leaf(v, f"{path}.{k}")
        return
    if isinstance(value, SequenceLeaf):
        for i, v in enumerate(value.data):
            _check_leaf(v, f"{path}[{i}]")
        return
    if isinstance(value, Array):
        return
    type_name = type(value).__module__ + "." + type(value).__name__
    msg = f"Parameter {path!r} is a {type_name}, expected a JAX Array."
    raise InvalidParamsError(msg)
