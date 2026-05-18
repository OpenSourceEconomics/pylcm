"""Runtime checks on JAX arrays produced during solve / simulate.

Three families of defensive checks fire during `model.solve()` and
`model.simulate()`:

- **Value-function NaN check** (`validate_V`). Fires after each
  backward-induction period; on failure, runs the diagnostic
  intermediates closure to pinpoint which intermediate
  (`U`, `F`, `E[V]`, `Q`) produced the NaN.
- **Regime transition probability check** keyed on
  `validate_regime_transitions_all_periods`. Pre-solve sweep:
  iterate active non-terminal regimes across periods, evaluate the regime
  transition function on the Cartesian product of its accepted grid
  variables, and verify finiteness, [0, 1] range, sum-to-1, no
  probability mass to inactive regimes, and no positive probability
  to a target with incomplete stochastic transitions.
- **State transition probability check** keyed on
  `validate_state_transitions_all_periods`. Pre-solve sweep over every
  `MarkovTransition` state transition (incl. per-target dict entries):
  evaluate the user function on the Cartesian product of the function's
  accepted grid variables, and verify outcome-axis size, [0, 1] range,
  and sum-to-1. State validation is gated by `log_level != "off"` because
  state transitions commonly depend on more variables than regime
  transitions, so the Cartesian product can blow up.

"""

import inspect
import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionProbabilitiesError,
    InvalidValueFunctionError,
)
from lcm.interfaces import Regime, StateActionSpace, _StochasticStateTransition
from lcm.typing import (
    FlatParams,
    FlatRegimeParams,
    FloatND,
    IntND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    StateOrActionName,
)


def validate_V(
    *,
    V_arr: FloatND,
    age: float | ScalarInt | ScalarFloat,
    regime_name: RegimeName | None = None,
    partial_solution: object = None,
    compute_intermediates: Callable | None = None,
    state_action_space: StateActionSpace | None = None,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND] | None = None,
    flat_params: FlatRegimeParams | None = None,
    period: int | None = None,
) -> None:
    """Validate the value function array for NaN values.

    When `compute_intermediates` is provided, NaN detection triggers a
    diagnostic run of the (already productmapped + JIT-compiled) closure to
    pinpoint which intermediate (U, F, E[V], Q) contains NaN.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.
        regime_name: Name of the regime (for error messages).
        partial_solution: Value function arrays for periods completed before
            the error. Attached to the exception for debug snapshots.
        compute_intermediates: Productmap + reduction closure (already
            JIT-compiled by `_build_compute_intermediates_per_period`)
            for the regime/period whose V array is being validated.
        state_action_space: StateActionSpace for the current regime/period.
        next_regime_to_V_arr: Next-period value function arrays.
        flat_params: Flat regime parameters.
        period: The current period index (forwarded to diagnostic closure).

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if not jnp.any(jnp.isnan(V_arr)):
        return

    n_nan = int(jnp.sum(jnp.isnan(V_arr)))
    total = int(V_arr.size)
    regime_part = f" in regime '{regime_name}'" if regime_name else ""
    all_nan = n_nan == total
    fraction_hint = "all" if all_nan else f"{n_nan} of {total}"
    exc = InvalidValueFunctionError(
        f"Value function at age {age}{regime_part}: {fraction_hint} values "
        f"are NaN.\n\n"
        "NaN propagates through Q = U + beta * E[V]. Common causes:\n"
        "- A missing feasibility constraint (e.g. negative leisure passed "
        "to a fractional exponent).\n"
        "- A regime parameter is NaN.\n"
        "- The utility function returned NaN (e.g. log of a non-positive "
        "argument).\n"
        "- The regime transition function returned NaN probabilities "
        "(e.g. from a NaN survival probability or a NaN fixed param).\n"
        "- A per-target state_transitions dict omits a reachable target "
        "(non-zero transition probability to an incomplete target).\n\n"
        "See the [NOTE] below for the per-intermediate / per-axis "
        "breakdown produced by `compute_intermediates`. When `log_path` "
        "is configured, an additional [NOTE] points to the on-disk "
        "snapshot directory written before this exception was raised. "
        "Debugging guide:\n"
        "https://pylcm.readthedocs.io/en/latest/user_guide/debugging/"
    )
    exc.partial_solution = partial_solution

    if compute_intermediates is not None and state_action_space is not None:
        try:
            _enrich_with_diagnostics(
                exc=exc,
                compute_intermediates=compute_intermediates,
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                flat_params=flat_params,
                regime_name=regime_name or "",
                age=float(age),
                period=period,
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("lcm").warning(
                "Diagnostic enrichment failed; raising original NaN error",
                exc_info=True,
            )

    raise exc


def _enrich_with_diagnostics(
    *,
    exc: InvalidValueFunctionError,
    compute_intermediates: Callable,
    state_action_space: StateActionSpace,
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND] | None,
    flat_params: FlatRegimeParams | None,
    regime_name: RegimeName,
    age: float,
    period: int | None,
) -> None:
    """Run diagnostic intermediates and attach summary to exception.

    `compute_intermediates` is productmap-wrapped over the full state-action
    space (same structure as `max_Q_over_a`) and fused with an on-device
    reduction step in a single JIT region — so the full-shape U/F/E/Q
    arrays never materialise in host-visible memory. It returns a flat
    dict of scalars + per-dimension vectors.

    Args:
        exc: The `InvalidValueFunctionError` to enrich with a diagnostic
            note and a `diagnostics` attribute.
        compute_intermediates: Fused productmap + reduction closure for the
            regime/period whose V array contained NaN.
        state_action_space: State-action space for the regime/period; used
            to build call kwargs and label per-dimension reductions.
        next_regime_to_V_arr: Immutable mapping of next-period value
            function arrays per regime (or `None`).
        flat_params: Optional mapping of flat regime parameter values.
        regime_name: Name of the regime whose V array failed validation.
        age: Age at which the V array failed validation.

    """
    all_names = (*state_action_space.state_names, *state_action_space.action_names)
    state_action_kwargs: dict[str, Any] = {
        **state_action_space.states,
        **state_action_space.actions,
    }
    # Drop any flat regime params that collide with state/action names so
    # they don't silently overwrite the grids.
    param_kwargs = (
        {k: v for k, v in flat_params.items() if k not in state_action_kwargs}
        if flat_params
        else {}
    )
    # Wrap Python scalars as JAX arrays so the call matches the dtype used
    # at trace time in `_build_compute_intermediates_per_period`; avoids a
    # retrace for the diagnostic invocation.
    call_kwargs: dict[str, Any] = {
        **state_action_kwargs,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **param_kwargs,
        "age": jnp.asarray(age),
        "period": jnp.int32(period) if period is not None else None,
    }

    reductions = compute_intermediates(**call_kwargs)
    exc.diagnostics = _summarize_diagnostics(
        reductions=reductions,
        variable_names=all_names,
        regime_name=regime_name,
        age=age,
    )
    exc.add_note(_format_diagnostic_summary(exc.diagnostics))


def _summarize_diagnostics(
    *,
    reductions: Mapping[str, Any],
    variable_names: tuple[str, ...],
    regime_name: RegimeName,
    age: float,
) -> dict[str, Any]:
    """Restructure the flat reduction pytree into the summary dict shape.

    Pure host-side — no device computation. Consumes the output of the
    fused compute-and-reduce function built in
    `_build_compute_intermediates_per_period`.

    Args:
        reductions: Flat mapping of reduction keys (`{metric}_overall`,
            `{metric}_by_{name}`, and `regime_probs`) to device arrays.
        variable_names: Tuple of state + action names in the order that
            matches the productmap axes.
        regime_name: Name of the regime for the summary header.
        age: Age for the summary header.

    Returns:
        Dict with per-metric `"overall"` and `"by_dim"` entries plus a
        `"regime_probs"` mapping, suitable for `_format_diagnostic_summary`.

    """
    summary: dict[str, Any] = {"regime_name": regime_name, "age": age}

    for key_out, key_in in [
        ("U_nan_fraction", "U_nan"),
        ("E_nan_fraction", "E_nan"),
        ("Q_nan_fraction", "Q_nan"),
        ("F_feasible_fraction", "F_feasible"),
    ]:
        by_dim: dict[str, list[float]] = {}
        for name in variable_names:
            k = f"{key_in}_by_{name}"
            if k in reductions:
                by_dim[name] = reductions[k].tolist()
        summary[key_out] = {
            "overall": float(reductions[f"{key_in}_overall"]),
            "by_dim": by_dim,
        }

    summary["regime_probs"] = {
        k: float(v) for k, v in reductions["regime_probs"].items()
    }
    return summary


def _format_diagnostic_summary(summary: dict[str, Any]) -> str:
    """Format diagnostic summary for exception note.

    Args:
        summary: Nested summary dict as produced by `_summarize_diagnostics`.

    Returns:
        Human-readable multi-line string suitable for `Exception.add_note`.

    """
    lines = [
        f"\nDiagnostics for regime '{summary['regime_name']}' at age {summary['age']}:",
    ]

    u_frac = summary.get("U_nan_fraction", {}).get("overall", 0)
    e_frac = summary.get("E_nan_fraction", {}).get("overall", 0)
    f_feas = summary.get("F_feasible_fraction", {}).get("overall", 0)
    lines.append(f"  F: {f_feas:.4f} feasible")
    lines.append(
        f"  Among feasible state-action pairs:  "
        f"U: {u_frac:.4f} NaN  |  E[V]: {e_frac:.4f} NaN"
    )

    probs = summary.get("regime_probs", {})
    if probs:
        prob_parts = [f"{t}: {p:.4f}" for t, p in probs.items()]
        lines.append(f"  Regime probs: {' | '.join(prob_parts)}")

    for label, key in (("U", "U_nan_fraction"), ("E[V]", "E_nan_fraction")):
        info = summary.get(key, {})
        frac = info.get("overall", 0)
        by_dim = info.get("by_dim", {})
        if frac > 0 and by_dim:
            lines.append(
                f"  {label} NaN fraction by state (among feasible state-action pairs):"
            )
            for dim_name, values in by_dim.items():
                formatted = ", ".join(f"{v:.2f}" for v in values)
                lines.append(f"    {dim_name:24s} [{formatted}]")

    return "\n".join(lines)


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


def validate_regime_transitions_all_periods(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for all periods before solve.

    For each period (except the last), for each active non-terminal regime, evaluate
    the regime transition function on all grid points and check that inactive regimes
    receive zero probability.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If any inactive regime receives
            positive transition probability.

    """
    last_period = ages.n_periods - 1
    non_terminal_active_at_last = [
        regime_name
        for regime_name, regime in regimes.items()
        if not regime.terminal and last_period in regime.active_periods
    ]
    if non_terminal_active_at_last:
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-terminal regime(s) {non_terminal_active_at_last} are active at the "
            f"last period (age {ages.exact_values[last_period]}). Non-terminal regimes "
            "must not be active at the last period because there is no next period to "
            "transition to. Adjust the 'active' function on these regimes to exclude "
            "the last age."
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

            _validate_regime_transition_single(
                regimes=regimes,
                regime_params=flat_params[regime_name],
                active_regimes_next_period=active_regimes_next_period,
                regime_name=regime_name,
                period=period,
                ages=ages,
            )


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
    regime_transition_func = regime.solve_functions.compute_regime_transition_probs

    state_action_space = regime.state_action_space(
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

    _validate_no_reachable_incomplete_targets(
        regimes=regimes,
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
        regime_name=regime_name,
        age=ages.values[period],  # noqa: PD011
    )


def _validate_no_reachable_incomplete_targets(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_transition_probs: MappingProxyType[RegimeName, FloatND],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
) -> None:
    """Check that targets with incomplete stochastic transitions are unreachable.

    A target is "incomplete" from the source regime if the source's
    `transitions[target_regime_name]` does not cover all of the target's
    stochastic state needs. Such targets must have zero transition
    probability, otherwise the continuation value cannot be computed. This
    includes self-transitions (regime reaches itself): omitting the
    self-entry in a per-target dict is a common user error.

    """
    solve_functions = regimes[regime_name].solve_functions
    transitions = solve_functions.transitions
    stochastic_names = solve_functions.stochastic_transition_names

    for target_regime_name in active_regimes_next_period:
        target_regime = regimes[target_regime_name]
        needs = {
            f"next_{s}"
            for s in target_regime.variables.state_names
            if f"next_{s}" in stochastic_names
        }
        if not needs:
            continue
        if target_regime_name in transitions and needs.issubset(
            transitions[target_regime_name]
        ):
            continue
        if not jnp.any(regime_transition_probs[target_regime_name] > 0):
            continue
        missing = sorted(needs - set(transitions.get(target_regime_name, {})))
        if target_regime_name not in transitions:
            missing = sorted(f"next_{s}" for s in target_regime.variables.state_names)
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime '{regime_name}' at age {age} has positive transition "
            f"probability to '{target_regime_name}', but '{regime_name}' "
            f"does not provide state transition(s) for: {missing}. Extend "
            f"`state_transitions` in '{regime_name}' to cover "
            f"'{target_regime_name}' (via a per-target dict if the "
            f"transition differs by target), or ensure "
            f"'{target_regime_name}' is unreachable."
        )


def validate_state_transitions_all_periods(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
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

    Raises:
        InvalidStateTransitionProbabilitiesError: If a `MarkovTransition`
            function returns the wrong outcome-axis size, values outside
            [0, 1], or rows that don't sum to 1.

    """
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

            state_action_space = regime.state_action_space(
                regime_params=flat_params[regime_name],
            )
            age = ages.values[period]  # noqa: PD011
            for transition in regime.stochastic_state_transitions.values():
                _validate_state_transition_single(
                    transition=transition,
                    regime_params=flat_params[regime_name],
                    state_action_space=state_action_space,
                    regime_name=regime_name,
                    age=age,
                    period=period,
                )


def _validate_state_transition_single(
    *,
    transition: _StochasticStateTransition,
    regime_params: FlatRegimeParams,
    state_action_space: StateActionSpace,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    period: int,
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
            # grid nor a param. Leave the validation to whichever solve
            # step surfaces the real error — raising here would conceal it.
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
    state_label = (
        f"state '{transition.state_name}'"
        if transition.target_regime_name is None
        else (
            f"state '{transition.state_name}' (target regime "
            f"'{transition.target_regime_name}')"
        )
    )

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
