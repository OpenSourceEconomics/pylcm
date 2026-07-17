"""Pre-flight numerical checks on user-supplied transition functions.

Called from `Model.solve()` and `Model.simulate()` before backward induction
runs. Two families:

- **Regime transition probability check** keyed on
  `validate_regime_transitions_all_periods`. Iterates active non-terminal
  regimes across periods, evaluates the regime transition function on the
  Cartesian product of its accepted grid variables, and verifies finiteness,
  [0, 1] range, sum-to-1, and no probability mass to inactive regimes.
  (Per-target regime transitions make omitted targets structurally
  unreachable; state-law coverage of reachable targets is validated at
  model build.)
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
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pandas as pd
from dags.tree import tree_path_from_qname

from _lcm.engine import Regime, StateActionSpace, _StochasticStateTransition
from _lcm.typing import FlatParams, FlatRegimeParams, RegimeName, StateOrActionName
from _lcm.utils.logging import raise_or_warn, validation_enabled
from _lcm.utils.namespace import ParamsQnameDepth
from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionProbabilitiesError,
)
from lcm.typing import FloatND, IntND, ScalarFloat, ScalarInt


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
        # Coarse law: read any target's (shared-leaf) binding.
        law_name = f"next_{transition.state_name}"
        parts_by_name = {name: tree_path_from_qname(name) for name in merged}
        return MappingProxyType(
            {
                parts[2]: merged[name]
                for name, parts in parts_by_name.items()
                if len(parts) == ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
                and parts[1] == law_name
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
        active_regimes_next_period = tuple(
            regime_name
            for regime_name, regime in regimes.items()
            if period + 1 in regime.active_periods
        )

        for regime_name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            if regime.terminal:
                continue

            try:
                _validate_regime_transition_single(
                    regimes=regimes,
                    regime_params=flat_params[regime_name],
                    active_regimes_next_period=active_regimes_next_period,
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
    regime_transition_func = regime.solution.compute_regime_transition_probs

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
        state_action_values=MappingProxyType(point),
    )


def _validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[RegimeName, FloatND],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    next_age: float | ScalarInt | ScalarFloat,
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
    if not jnp.allclose(sum_all, 1.0):
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
            raise InvalidRegimeTransitionProbabilitiesError(
                f"Regime '{r}' is inactive at age {next_age} but has positive "
                f"transition probability from '{regime_name}' between ages {age} and "
                f"{next_age}. Either make '{r}' active or ensure its probability is 0."
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
    failing_mask = ~jnp.isclose(sum_all, 1.0)
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
                if _per_target_unreachable_at_next_period(
                    transition=transition, regimes=regimes, period=period
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


def _per_target_unreachable_at_next_period(
    *,
    transition: _StochasticStateTransition,
    regimes: MappingProxyType[RegimeName, Regime],
    period: int,
) -> bool:
    """Return True when a per-target transition's target deactivates before reach.

    `solve()` and `simulate()` only dispatch a per-target MarkovTransition
    for targets in `active_regimes_next_period` at the source's period;
    targets that deactivate before the source can reach them never fire at
    runtime. The pre-solve validator mirrors that gate so a per-target
    function whose output shape only needs to match the (always-zero-
    weighted) target's outcome grid in principle is not numerically
    evaluated against the source's state grid.
    """
    if transition.target_regime_name is None:
        return False
    target = regimes[transition.target_regime_name]
    return period + 1 not in target.active_periods


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
            # not silent.
            logger.warning(
                "MarkovTransition for state '%s' in regime '%s' not numerically "
                "validated: parameter '%s' is not a recognized grid or model "
                "parameter.",
                transition.state_name,
                regime_name,
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
