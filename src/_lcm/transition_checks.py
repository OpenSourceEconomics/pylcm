"""Pre-flight numerical checks on user-supplied transition functions.

Called from `Model.solve()` and `Model.simulate()` before backward induction
runs. Two families:

- **Regime transition probability check** keyed on
  `validate_regime_transitions_all_periods`. Iterates active non-terminal
  regimes across periods, evaluates the regime transition function on the
  Cartesian product of its accepted grid variables, and verifies finiteness,
  [0, 1] range, sum-to-1, and no probability mass to inactive regimes.
  The construction-time graph supplies the allowed target set; state-law
  coverage of retained targets is validated at model build.
- **State transition probability check** keyed on
  `validate_state_transitions_all_periods`. Sweeps every `MarkovTransition`
  state transition (incl. per-target dict entries), evaluates the user
  function on the Cartesian product of the function's accepted grid
  variables, and verifies outcome-axis size, [0, 1] range, and sum-to-1.

Both checks read their policy off the `logger`: `log_level="off"` skips the
check, `"warning"` / `"progress"` log each failure and let the run continue,
`"debug"` raises on the first failure.

These are runtime checks: they need a fully-built `Regime` plus user
`flat_params` and evaluate the transition functions numerically. The
construction-time regime-spec validators (`Regime.__post_init__`, which
inspect grids, signatures, and Python source) are a separate concern.

"""

import inspect
import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from dags.tree import tree_path_from_qname

from _lcm.engine import Regime, StateActionSpace, _StochasticStateTransition
from _lcm.regime_building.next_state import get_next_stochastic_weights_function
from _lcm.transition_plans import LotteryLifetime
from _lcm.typing import FlatParams, FlatRegimeParams, RegimeName, StateOrActionName
from _lcm.utils.logging import raise_or_warn, validation_enabled
from _lcm.utils.namespace import ParamsQnameDepth
from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarFloat, ScalarInt

_NO_EXTRA_GRIDS: Mapping[StateOrActionName, FloatND | IntND] = MappingProxyType({})


def validate_transitions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
) -> None:
    """Validate regime and state transition probabilities before solve / simulate.

    Runs the regime-transition check then the state-transition check. Both
    self-gate on the logger's runtime-validation policy (`log_level="off"`
    skips, `"warning"` / `"progress"` warn, `"debug"` raises).

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy.

    """
    validate_regime_transitions_all_periods(
        regimes=regimes, flat_params=flat_params, ages=ages, logger=logger
    )
    validate_state_transitions_all_periods(
        regimes=regimes, flat_params=flat_params, ages=ages, logger=logger
    )
    validate_joint_transitions_all_periods(
        regimes=regimes, flat_params=flat_params, ages=ages, logger=logger
    )


def _params_callable_for_state_transition(
    *,
    regime: Regime,
    flat_params_for_regime: FlatRegimeParams,
    transition: _StochasticStateTransition,
) -> FlatRegimeParams:
    """Return un-qualified params for calling a state-transition function.

    Both `regime.resolved_fixed_params` and `flat_params_for_regime` key
    every transition-law param granularly (`<target>__next_<state>__<param>`),
    matching the engine's target-prefixed function qnames:

    - per-target dicts ⇒ one entry per target, possibly distinct values
    - coarse laws      ⇒ one entry per reachable carrying target, all
      sharing the same leaf — any target's binding yields the law's params

    The `MarkovTransition`'s user function is called with the raw
    parameter names from its signature, so the validator must strip
    the same qualifier before lookup. Without the strip, every
    transition-function parameter that isn't a grid axis falls through
    to the "not numerically validated" skip branch and the
    per-transition numerical check never runs.
    """
    merged = {**regime.resolved_fixed_params, **flat_params_for_regime}

    if transition.target_regime_name is None:
        # Coarse law: prefer any target's shared-leaf granular binding.
        law_name = f"next_{transition.state_name}"
        parts_by_name = {name: tree_path_from_qname(name) for name in merged}
        granular = {
            parts[2]: merged[name]
            for name, parts in parts_by_name.items()
            if len(parts) == ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
            and parts[1] == law_name
        }
        if granular:
            return MappingProxyType(granular)

        # A law with no temporally retained carrying target keeps its original
        # template-qualified binding. It may still be checked as a user declaration.
        prefix = f"{law_name}__"
        return MappingProxyType(
            {
                name.removeprefix(prefix): value
                for name, value in merged.items()
                if name.startswith(prefix)
            }
        )

    prefix = f"{transition.target_regime_name}__next_{transition.state_name}__"
    return MappingProxyType(
        {
            name.removeprefix(prefix): value
            for name, value in merged.items()
            if name.startswith(prefix)
        }
    )


def validate_regime_transitions_all_periods(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
) -> None:
    """Validate regime transition probabilities for all periods before solve.

    For each period (except the last), for each active non-terminal regime, evaluate
    the regime transition function on all grid points and check that inactive regimes
    receive zero probability.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy. `log_level="off"`
            returns immediately; `"warning"` / `"progress"` log each failure and
            continue; `"debug"` raises on the first failure.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If a regime transition produces
            invalid probabilities and the logger implies raise mode.

    """
    # Skipped entirely at `log_level="off"`. What that costs is the diagnosis
    # rather than the answer: the continuation aggregator measures the mass its
    # retained targets represent and returns NaN unless it is one and no weight
    # is negative, so a misspecification survives as a NaN rather than as a
    # plausible number. These checks name the regime, the period and the
    # offending target instead, which a NaN cannot.
    if not validation_enabled(logger):
        return

    last_period = ages.n_periods - 1
    non_terminal_active_at_last = [
        regime_name
        for regime_name, regime in regimes.items()
        if not regime.terminal and last_period in regime.active_periods
    ]
    if non_terminal_active_at_last:
        raise_or_warn(
            logger=logger,
            error=InvalidRegimeTransitionProbabilitiesError(
                f"Non-terminal regime(s) {non_terminal_active_at_last} are active at "
                f"the last period (age {ages.exact_values[last_period]}). Non-terminal "
                "regimes must not be active at the last period because there is no "
                "next period to transition to. Adjust the 'active' function on these "
                "regimes to exclude the last age."
            ),
        )

    for period in range(ages.n_periods - 1):
        for regime_name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            if regime.terminal:
                continue

            try:
                _validate_regime_transition_single(
                    regimes=regimes,
                    regime_params=flat_params[regime_name],
                    active_regimes_next_period=(
                        regime.solution.reachability.targets(
                            period=period, source=regime_name
                        )
                    ),
                    regime_name=regime_name,
                    period=period,
                    ages=ages,
                )
            except InvalidRegimeTransitionProbabilitiesError as error:
                raise_or_warn(logger=logger, error=error)


def _validate_regime_transition_single(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_params: FlatRegimeParams,
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    period: int,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for a single regime and period.

    Evaluate the regime transition function on the Cartesian product of all grid
    variables it accepts, using `jax.vmap` for vectorised evaluation.

    """
    regime = regimes[regime_name]
    # Non-None guaranteed: only called for non-terminal regimes
    regime_transition_func = regime.solution.validation_regime_transition_probs

    state_action_space = regime.solution.state_action_space(
        regime_params=regime_params,
    )

    # Filter params to only those accepted by the transition function
    accepted_params = set(inspect.signature(regime_transition_func).parameters)  # ty: ignore[invalid-argument-type]
    filtered_params = {k: v for k, v in regime_params.items() if k in accepted_params}

    # Collect only grid variables the transition function accepts
    grids: dict[StateOrActionName, FloatND | IntND] = {
        k: v for k, v in state_action_space.states.items() if k in accepted_params
    } | {k: v for k, v in state_action_space.actions.items() if k in accepted_params}

    # Build flat Cartesian product and vmap over all combinations
    grid_var_names = list(grids.keys())
    grid_arrays = list(grids.values())

    # Pin to int32: a Python-int `period` traced through `jax.vmap` becomes
    # int64 under x64, breaking any int32 `period` contract downstream.
    period_int32 = jnp.int32(period)

    if grid_arrays:
        mesh = jnp.meshgrid(*grid_arrays, indexing="ij")
        flat_arrays = [m.ravel() for m in mesh]

        def _call(
            *args: FloatND | IntND,
            _names: list[str] = grid_var_names,
            _params: dict = filtered_params,
            _func: object = regime_transition_func,
            _period: ScalarInt = period_int32,
            _age: ScalarInt | ScalarFloat = ages.values[period],  # noqa: PD011
        ) -> MappingProxyType[RegimeName, FloatND]:
            kwargs = dict(zip(_names, args, strict=True))
            return _func(  # ty: ignore[call-non-callable]
                **kwargs, **_params, period=_period, age=_age
            )

        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = jax.vmap(
            _call
        )(*flat_arrays)
        point = dict(zip(grid_var_names, flat_arrays, strict=True))
    else:
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            regime_transition_func(  # ty: ignore[call-non-callable]
                **filtered_params,
                period=period_int32,
                age=ages.values[period],  # noqa: PD011
            )
        )
        point: dict[StateOrActionName, FloatND | IntND] = {}

    _validate_regime_transition_probs(
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
        regime_name=regime_name,
        age=ages.values[period],  # noqa: PD011
        next_age=ages.values[period + 1],  # noqa: PD011
        period=period,
        state_action_values=MappingProxyType(point),
    )


def _validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[RegimeName, FloatND],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    next_age: float | ScalarInt | ScalarFloat,
    period: int | None = None,
    state_action_values: MappingProxyType[StateOrActionName, FloatND | IntND]
    | None = None,
) -> None:
    """Validate regime transition probabilities.

    Check that probabilities are finite, sum to 1 across all regimes, and that
    inactive regimes have zero probability.

    Args:
        regime_transition_probs: Immutable mapping of regime names to probability
            arrays.
        active_regimes_next_period: Tuple of regime names active in the next period.
        regime_name: Name of the source regime (for error messages).
        age: Current age (for error messages).
        next_age: Next age (for error messages).
        period: Optional source-period index for graph diagnostics.
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in error messages to help diagnose which inputs cause violations.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If probabilities are non-finite,
            outside [0, 1], don't sum to 1, or assign positive probability to inactive
            regimes.

    """
    all_probs = jnp.stack(list(regime_transition_probs.values()))

    if jnp.any(~jnp.isfinite(all_probs)):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-finite values in regime transition probabilities from "
            f"'{regime_name}' between ages {age} and {next_age}. Check the "
            f"'next_regime' function of the '{regime_name}' regime."
        )

    if jnp.any(all_probs < 0) or jnp.any(all_probs > 1):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} contain values outside [0, 1]. Check the 'next_regime' "
            f"function of the '{regime_name}' regime."
        )

    sum_all = jnp.sum(all_probs, axis=0)
    if jnp.any(_unit_mass_violations(sum_all)):
        detail = _format_sum_violation(
            sum_all=sum_all,
            state_action_values=state_action_values,
        )
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} do not sum to 1.0. {detail}\n"
            f"Check the 'next_regime' function of the '{regime_name}' regime."
        )

    inactive = set(regime_transition_probs) - set(active_regimes_next_period)
    for r in inactive:
        if jnp.any(regime_transition_probs[r] > 0):
            period_detail = "" if period is None else f" in period {period}"
            raise InvalidRegimeTransitionProbabilitiesError(
                f"Regime '{r}' is inactive at age {next_age} but has positive "
                f"transition probability from '{regime_name}' between ages {age} and "
                f"{next_age}{period_detail}. Its mass is not represented in the "
                f"continuation, so what the remaining targets carry is less than "
                f"unit mass and the solve returns NaN rather than a value that "
                f"does not depend on '{r}' at all. Either make '{r}' active at "
                f"that age or give it probability 0 there."
            )


def _format_sum_violation(
    *,
    sum_all: FloatND,
    state_action_values: MappingProxyType[StateOrActionName, FloatND | IntND]
    | None = None,
) -> str:
    """Format a human-readable description of probability sum violations.

    Args:
        sum_all: Array of probability sums (per-subject).
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in the output to show which inputs cause violations.

    Returns:
        Formatted string describing which sums violate the sum-to-1 constraint.

    """
    sum_all = jnp.atleast_1d(sum_all)
    if state_action_values is not None:
        state_action_values = MappingProxyType(
            {name: jnp.atleast_1d(arr) for name, arr in state_action_values.items()}
        )
    failing_mask = _unit_mass_violations(sum_all)
    failing_indices = jnp.where(failing_mask)[0].astype(jnp.int32)
    failing_sums = sum_all[failing_mask]
    n_failing = int(failing_indices.shape[0])
    n_show = min(n_failing, 5)
    data: dict[str, list[float]] = {
        "subject": failing_indices[:n_show].tolist(),
    }
    if state_action_values is not None:
        for name, arr in state_action_values.items():
            data[name] = [float(arr[i]) for i in failing_indices[:n_show]]
    data["sum"] = failing_sums[:n_show].tolist()
    df = pd.DataFrame(data)
    return (
        f"{n_failing} of {sum_all.shape[0]} probability vectors do not sum to 1.0.\n"
        f"First failing entries:\n{df.to_string(index=False)}"
    )


def validate_state_transitions_all_periods(  # noqa: C901
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
) -> None:
    """Validate every `MarkovTransition` state transition before solve.

    For each non-terminal active period of each active regime, iterate the
    regime's `stochastic_state_transitions` and evaluate each
    `MarkovTransition` function on the Cartesian product of its accepted
    grid variables. Check:

    - The output's last-axis size matches the state's outcome count.
    - All values lie in [0, 1].
    - Rows along the last axis sum to 1.

    Fast-exits when no regime in the model has any stochastic state
    transitions, so models without `MarkovTransition` states pay no cost.

    Args:
        regimes: Immutable mapping of regime names to canonical regimes.
        flat_params: Immutable mapping of regime names to flat parameter
            mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy. `log_level="off"`
            returns immediately; `"warning"` / `"progress"` log each failure and
            continue; `"debug"` raises on the first failure.

    Raises:
        InvalidStateTransitionProbabilitiesError: If a `MarkovTransition`
            function returns the wrong outcome-axis size, values outside
            [0, 1], or rows that don't sum to 1, and the logger implies raise
            mode.

    """
    if not validation_enabled(logger):
        return
    if not any(r.stochastic_state_transitions for r in regimes.values()):
        return

    for period in range(ages.n_periods - 1):
        for regime_name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            if regime.terminal:
                continue
            if not regime.stochastic_state_transitions:
                continue

            state_action_space = regime.solution.state_action_space(
                regime_params=flat_params[regime_name],
            )
            age = ages.values[period]  # noqa: PD011
            for transition in regime.stochastic_state_transitions.values():
                if _state_transition_unused_in_period(
                    transition=transition,
                    regime=regime,
                    period=period,
                ):
                    continue
                try:
                    _validate_state_transition_single(
                        transition=transition,
                        regime_params=_params_callable_for_state_transition(
                            regime=regime,
                            flat_params_for_regime=flat_params[regime_name],
                            transition=transition,
                        ),
                        state_action_space=state_action_space,
                        regime_name=regime_name,
                        age=age,
                        period=period,
                        logger=logger,
                    )
                except InvalidStateTransitionProbabilitiesError as error:
                    raise_or_warn(logger=logger, error=error)


def validate_joint_transitions_all_periods(  # noqa: C901, PLR0912
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
) -> None:
    """Validate every transition-local lottery before solve or simulation."""
    if not validation_enabled(logger):
        return

    # A callable support is resolved only after params are bound, so its full
    # pytree signature cannot be checked during ``Regime`` construction.  Compare
    # every active period and both phases in this one preflight instead.  Values
    # may differ, but tree structure, event shapes, and dtypes are a static JIT/AOT
    # contract and must not depend on period or perceived/realized phase.
    support_schemas: dict[
        tuple[RegimeName, RegimeName, str],
        tuple[str, int, object, tuple[tuple[tuple[int, ...], str], ...]],
    ] = {}

    for period in range(ages.n_periods - 1):
        period_int32 = jnp.int32(period)
        age = ages.values[period]  # noqa: PD011
        for regime_name, regime in regimes.items():
            if regime.terminal or period not in regime.active_periods:
                continue
            state_action_space = regime.solution.state_action_space(
                regime_params=flat_params[regime_name]
            )
            # A carried state has no solve grid axis, so a simulate-phase law
            # reading one is not resolvable on the solution state-action space.
            # Sweep its simulate-phase domain alongside the solve grids instead.
            carried_only_grids = MappingProxyType(
                {
                    name: regime.simulation.grids[name].to_jax()
                    for name in sorted(regime.simulation.carried_only_state_names)
                }
            )
            for phase_name, phase in (
                ("solve", regime.solution),
                ("simulate", regime.simulation),
            ):
                targets = phase.reachability.targets(period=period, source=regime_name)
                functions = (
                    regime.solution.continuation_functions
                    if phase_name == "solve"
                    else regime.simulation.functions
                )
                for target in targets:
                    plan = phase.transition_plans.get(target)
                    if plan is None:
                        continue
                    joint_laws = {
                        name: lottery
                        for name, lottery in plan.lotteries.items()
                        if lottery.lifetime is LotteryLifetime.TRANSITION_LOCAL
                    }
                    if not joint_laws:
                        continue
                    compute_weights = get_next_stochastic_weights_function(
                        regime_name=target,
                        functions=functions,
                        transitions=phase.transitions[target],
                        transition_plans=phase.transition_plans,
                    )
                    weights = _evaluate_joint_weights(
                        func=compute_weights,
                        state_action_space=state_action_space,
                        extra_grids=(
                            carried_only_grids
                            if phase_name == "simulate"
                            else _NO_EXTRA_GRIDS
                        ),
                        regime_params=flat_params[regime_name],
                        period=period_int32,
                        age=age,
                        regime_name=regime_name,
                        phase_name=phase_name,
                        logger=logger,
                    )
                    if weights is None:
                        continue
                    for kernel_name, law in joint_laws.items():
                        support_provider_name = law.support_provider_name
                        if support_provider_name is None:
                            raise RegimeInitializationError(
                                f"Joint transition {kernel_name!r} has no "
                                "support provider in its canonical plan."
                            )
                        support = _evaluate_joint_support(
                            func=phase.transitions[target][support_provider_name],
                            regime_params=flat_params[regime_name],
                            period=period_int32,
                            age=age,
                            kernel_name=kernel_name,
                            support_size=law.support_signature.size,
                            regime_name=regime_name,
                            phase_name=phase_name,
                            target=target,
                            logger=logger,
                        )
                        if support is not None:
                            leaves, tree = jax.tree_util.tree_flatten(support)
                            leaf_schema = tuple(
                                (tuple(leaf.shape[1:]), str(leaf.dtype))
                                for leaf in leaves
                            )
                            signature_key = (regime_name, target, kernel_name)
                            previous = support_schemas.get(signature_key)
                            if previous is None:
                                support_schemas[signature_key] = (
                                    phase_name,
                                    period,
                                    tree,
                                    leaf_schema,
                                )
                            else:
                                (
                                    previous_phase,
                                    previous_period,
                                    previous_tree,
                                    previous_leaves,
                                ) = previous
                                if (
                                    tree != previous_tree
                                    or leaf_schema != previous_leaves
                                ):
                                    changed_support = (
                                        "Joint transition "
                                        f"{kernel_name}.support changed its "
                                        "static pytree signature between "
                                        f"{previous_phase} period "
                                        f"{previous_period} and {phase_name} "
                                        f"period {period} of regime "
                                        f"{regime_name}, target {target}. "
                                        "Support values may differ, but "
                                        "pytree structure, leaf event shapes, "
                                        "and dtypes must remain identical "
                                        "across periods and phases; got "
                                        f"{previous_leaves} and {leaf_schema}."
                                    )
                                    raise_or_warn(
                                        logger=logger,
                                        error=RegimeInitializationError(
                                            changed_support
                                        ),
                                    )
                        probs = weights[f"weight_{target}__{kernel_name}"]
                        if (
                            probs.ndim == 0
                            or probs.shape[-1] != law.support_signature.size
                        ):
                            length = 1 if probs.ndim == 0 else probs.shape[-1]
                            raise_or_warn(
                                logger=logger,
                                error=InvalidStateTransitionProbabilitiesError(
                                    f"Joint transition {kernel_name}.probabilities "
                                    f"returned length {length}; support_size is "
                                    f"{law.support_signature.size} "
                                    f"({phase_name} phase of regime {regime_name}, "
                                    f"target {target}, age {age})."
                                ),
                            )
                        invalid_values = (
                            jnp.any(~jnp.isfinite(probs))
                            or jnp.any(probs < 0)
                            or jnp.any(probs > 1)
                            or jnp.any(_unit_mass_violations(jnp.sum(probs, axis=-1)))
                        )
                        if invalid_values:
                            raise_or_warn(
                                logger=logger,
                                error=InvalidStateTransitionProbabilitiesError(
                                    f"Joint transition {kernel_name}.probabilities "
                                    "contains nonfinite or out-of-range values, "
                                    "or rows that do not sum to one "
                                    f"({phase_name} phase of regime {regime_name}, "
                                    f"target {target}, age {age})."
                                ),
                            )


def _evaluate_joint_support(
    *,
    func: Callable[..., Any],
    regime_params: FlatRegimeParams,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
    kernel_name: str,
    support_size: int,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    logger: logging.Logger,
) -> Any:  # noqa: ANN401
    """Evaluate and validate one parameter-bound joint-support provider."""
    kwargs: dict[str, object] = {}
    for name in inspect.signature(func).parameters:
        if name == "period":
            kwargs[name] = period
        elif name == "age":
            kwargs[name] = age
        elif name in regime_params:
            kwargs[name] = regime_params[name]
        else:
            raise_or_warn(
                logger=logger,
                error=RegimeInitializationError(
                    f"Joint transition {kernel_name}.support may read only period, "
                    f"age, and parameters; unbound argument {name} appears in the "
                    f"{phase_name} phase of regime {regime_name}, target {target}."
                ),
            )
            return None

    support = func(**kwargs)
    leaves, _ = jax.tree_util.tree_flatten(support)
    invalid_shapes = [
        getattr(leaf, "shape", None)
        for leaf in leaves
        if not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != support_size
    ]
    if not leaves or invalid_shapes:
        raise_or_warn(
            logger=logger,
            error=RegimeInitializationError(
                f"Joint transition {kernel_name}.support must be a nonempty pytree "
                f"whose every leaf has leading axis support_size={support_size}; "
                f"invalid leaf shape(s): {invalid_shapes}."
            ),
        )
        # The caller compares static schemas only for structurally valid
        # supports. In warning mode validation continues, so returning the invalid
        # pytree here would make the comparison itself dereference missing shapes.
        return None

    try:
        has_nonfinite = any(
            not bool(jax.numpy.all(jax.numpy.isfinite(leaf))) for leaf in leaves
        )
    except TypeError:
        has_nonfinite = True
    if has_nonfinite:
        raise_or_warn(
            logger=logger,
            error=RegimeInitializationError(
                f"Joint transition {kernel_name}.support contains nonfinite or "
                f"unsupported leaf values ({phase_name} phase of regime "
                f"{regime_name}, target {target}, age {age})."
            ),
        )
    return support


def _evaluate_joint_weights(
    *,
    func: Callable[..., Mapping[str, FloatND | IntND]],
    state_action_space: StateActionSpace,
    extra_grids: Mapping[StateOrActionName, FloatND | IntND],
    regime_params: FlatRegimeParams,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
    regime_name: RegimeName,
    phase_name: str,
    logger: logging.Logger,
) -> Mapping[str, FloatND | IntND] | None:
    """Evaluate one compiled probability DAG on its accepted grids.

    `extra_grids` carries grid axes the solution state-action space does not
    hold — the simulate-phase domain of each carried-only state. An argument
    that resolves to none of the grids or the regime's parameters leaves the
    lottery unvalidated, which `log_level="debug"` refuses rather than sampling
    from an unexamined law.
    """
    grid_args: dict[StateOrActionName, FloatND | IntND] = {}
    scalar_kwargs: dict[str, object] = {}
    for name in inspect.signature(func).parameters:
        if name == "period":
            scalar_kwargs[name] = period
        elif name == "age":
            scalar_kwargs[name] = age
        elif name in state_action_space.states:
            grid_args[name] = state_action_space.states[name]
        elif name in state_action_space.actions:
            grid_args[name] = state_action_space.actions[name]
        elif name in extra_grids:
            grid_args[name] = extra_grids[name]
        elif name in regime_params:
            scalar_kwargs[name] = regime_params[name]
        else:
            raise_or_warn(
                logger=logger,
                error=InvalidStateTransitionProbabilitiesError(
                    f"Joint transitions in regime {regime_name!r} "
                    f"({phase_name} phase) cannot be validated numerically: "
                    f"argument {name!r} is neither a grid variable of that "
                    "phase nor a parameter of the regime, so the lottery it "
                    "weights is never examined."
                ),
            )
            return None

    if not grid_args:
        return func(**scalar_kwargs)

    grid_var_names = list(grid_args)
    mesh = jnp.meshgrid(*grid_args.values(), indexing="ij")
    flat_arrays = [array.ravel() for array in mesh]

    def _call(*args: FloatND | IntND) -> Mapping[str, FloatND | IntND]:
        kwargs = dict(zip(grid_var_names, args, strict=True))
        return func(**kwargs, **scalar_kwargs)

    return jax.vmap(_call)(*flat_arrays)


def _state_transition_unused_in_period(
    *,
    transition: _StochasticStateTransition,
    regime: Regime,
    period: int,
) -> bool:
    """Return whether a state transition has no retained edge this period.

    A coarse (`target_regime_name is None`) state law applies regardless of
    the regime-transition graph's own target set — an empty `period_targets`
    is a fact about *which regime* is reached, not about whether a coarse
    state law still needs checking. Only a per-target state law can be
    unused, and only when its specific target isn't retained this period.
    """
    if transition.target_regime_name is None:
        return False
    period_targets = regime.solution.reachability.targets(
        period=period, source=regime.name
    )
    return transition.target_regime_name not in period_targets


def _validate_state_transition_single(
    *,
    transition: _StochasticStateTransition,
    regime_params: FlatRegimeParams,
    state_action_space: StateActionSpace,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    period: int,
    logger: logging.Logger,
) -> None:
    """Evaluate one MarkovTransition on its grid args and validate the output."""
    func = transition.func
    sig_params = tuple(inspect.signature(func).parameters)

    grid_args: dict[StateOrActionName, FloatND | IntND] = {}
    scalar_kwargs: dict[str, object] = {}
    period_int32 = jnp.int32(period)

    for name in sig_params:
        if name == "period":
            scalar_kwargs["period"] = period_int32
        elif name == "age":
            scalar_kwargs["age"] = age
        elif name in state_action_space.states:
            grid_args[name] = state_action_space.states[name]
        elif name in state_action_space.actions:
            grid_args[name] = state_action_space.actions[name]
        elif name in regime_params:
            scalar_kwargs[name] = regime_params[name]
        else:
            # An indexing param the function expects is neither a regime
            # grid nor a param. Skip numerical validation for this
            # transition rather than raising — a raise here would conceal
            # the real error the solve step surfaces. Warn so the skip is
            # not silent. Name the phase: a `Phased` law has two variants under
            # one state name, and only one of them may be hitting this branch.
            phase_suffix = (
                f" ({transition.phase} phase)" if transition.phase is not None else ""
            )
            logger.warning(
                "MarkovTransition for state '%s' in regime '%s'%s not numerically "
                "validated: parameter '%s' is not a recognized grid or model "
                "parameter.",
                transition.state_name,
                regime_name,
                phase_suffix,
                name,
            )
            return

    if grid_args:
        grid_var_names = list(grid_args.keys())
        grid_arrays = list(grid_args.values())
        mesh = jnp.meshgrid(*grid_arrays, indexing="ij")
        flat_arrays = [m.ravel() for m in mesh]

        def _call(
            *args: FloatND | IntND,
            _names: list[str] = grid_var_names,
            _scalar: dict[str, object] = scalar_kwargs,
            _func: object = func,
        ) -> FloatND:
            kwargs = dict(zip(_names, args, strict=True))
            return _func(**kwargs, **_scalar)  # ty: ignore[call-non-callable]

        probs = jax.vmap(_call)(*flat_arrays)
    else:
        probs = func(**scalar_kwargs)

    _check_state_probs(
        probs=probs,
        transition=transition,
        regime_name=regime_name,
        age=age,
    )


def _check_state_probs(
    *,
    probs: FloatND,
    transition: _StochasticStateTransition,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
) -> None:
    """Assert outcome-axis size, [0, 1] range, and sum-to-1 on a probs array."""
    qualifiers = []
    if transition.target_regime_name is not None:
        qualifiers.append(f"target regime '{transition.target_regime_name}'")
    if transition.phase is not None:
        # A `Phased` law has two variants under one state name; without the phase the
        # message would not say which of them is malformed.
        qualifiers.append(f"{transition.phase} phase")
    state_label = f"state '{transition.state_name}'"
    if qualifiers:
        state_label += f" ({', '.join(qualifiers)})"

    if probs.shape[-1] != transition.n_outcomes:
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned an outcome axis of size "
            f"{probs.shape[-1]}; expected {transition.n_outcomes} from the "
            f"state's DiscreteGrid."
        )

    if jnp.any(probs < 0) or jnp.any(probs > 1):
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned values outside [0, 1]."
        )

    row_sums = jnp.sum(probs, axis=-1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-6):
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned rows that do not sum to 1 along the "
            f"outcome axis."
        )


def _unit_mass_violations(sum_all: FloatND) -> BoolND:
    """Return the mask of total regime masses that are not unit mass.

    The single criterion for "does not sum to 1", shared by the check that
    raises and the formatter that reports which entries failed — a formatter
    with a looser criterion of its own reports an empty table alongside a
    raised error.

    Tight by design. A tolerance wide enough to admit a mass that changes the
    Bellman `argmax` is not a guard: at float32 a total mass of `1.000005` is
    enough to reverse a decision, and `jnp.allclose`'s default `rtol` of `1e-5`
    admits it. Sixteen epsilons leaves room for the rounding of a handful of
    summed probabilities and nothing else.
    """
    tolerance = 16.0 * float(jnp.finfo(sum_all.dtype).eps)
    return jnp.abs(sum_all - 1.0) > tolerance
