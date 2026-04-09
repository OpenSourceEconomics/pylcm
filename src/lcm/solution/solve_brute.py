import logging
import time
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.interfaces import InternalRegime
from lcm.typing import FloatND, InternalParams, RegimeName
from lcm.utils.error_handling import validate_V
from lcm.utils.logging import (
    format_duration,
    log_nan_in_V,
    log_period_timing,
    log_V_stats,
)


def solve(
    *,
    internal_params: InternalParams,
    ages: AgeGrid,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    logger: logging.Logger,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Solve a model using grid search.

    Args:
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        internal_regimes: The internal regimes, that contain all necessary functions
            to solve the model.
        logger: Logger that logs to stdout.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    """
    solution: dict[int, MappingProxyType[RegimeName, FloatND]] = {}
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND] = MappingProxyType(
        {name: jnp.empty(0) for name in internal_regimes}
    )

    logger.info("Starting solution")
    total_start = time.monotonic()

    # backwards induction loop
    for period in reversed(range(ages.n_periods)):
        period_start = time.monotonic()
        period_solution: dict[RegimeName, FloatND] = {}

        active_regimes = {
            regime_name: regime
            for regime_name, regime in internal_regimes.items()
            if period in regime.active_periods
        }

        for name, internal_regime in active_regimes.items():
            state_action_space = internal_regime.state_action_space(
                regime_params=internal_params[name],
            )
            max_Q_over_a = internal_regime.solve_functions.max_Q_over_a[period]

            # evaluate Q-function on states and actions, and maximize over actions
            V_arr = max_Q_over_a(
                **state_action_space.states,
                **state_action_space.actions,
                next_regime_to_V_arr=next_regime_to_V_arr,
                **internal_params[name],
            )

            log_nan_in_V(
                logger=logger,
                regime_name=name,
                age=ages.values[period],
                V_arr=V_arr,
            )
            log_V_stats(logger=logger, regime_name=name, V_arr=V_arr)

            try:
                validate_V(V_arr=V_arr, age=ages.values[period], regime_name=name)
            except InvalidValueFunctionError as exc:
                exc.partial_solution = MappingProxyType(solution)
                diag_funcs = internal_regime.solve_functions.diagnostic_Q_and_F.get(
                    period
                )
                if diag_funcs is not None:
                    # Move inputs to CPU — the GPU may be out of memory
                    # for compiling new diagnostic kernels after the solve.
                    cpu = jax.devices("cpu")[0]
                    call_kwargs = jax.device_put(
                        {
                            **state_action_space.states,
                            **state_action_space.actions,
                            "next_regime_to_V_arr": next_regime_to_V_arr,
                            **internal_params[name],
                        },
                        cpu,
                    )
                    diag_results: dict[str, Any] = {}
                    for diag_name, diag_func in diag_funcs.items():
                        diag_results[diag_name] = np.asarray(diag_func(**call_kwargs))
                    exc.diagnostics = _summarize_diagnostics(
                        diag_results,
                        state_action_space.state_names,
                        regime_name=name,
                        age=ages.values[period],
                    )
                    exc.add_note(_format_diagnostic_summary(exc.diagnostics))
                raise
            period_solution[name] = V_arr

        next_regime_to_V_arr = MappingProxyType(period_solution)
        solution[period] = next_regime_to_V_arr

        elapsed = time.monotonic() - period_start
        log_period_timing(
            logger=logger,
            age=ages.values[period],
            n_active_regimes=len(active_regimes),
            elapsed=elapsed,
        )

    total_elapsed = time.monotonic() - total_start
    logger.info("Solution complete  (%s)", format_duration(seconds=total_elapsed))

    return MappingProxyType(solution)


def _summarize_diagnostics(
    diag: dict[str, Any],
    state_names: tuple[str, ...],
    *,
    regime_name: str,
    age: float,
) -> dict[str, Any]:
    """Reduce state-grid diagnostic arrays to marginal fractions by state dim.

    The diagnostic returns scalars (NaN fractions) that were productmapped
    over states, giving state-grid-shaped arrays. Compute marginals by
    reducing along individual state dimensions.
    """
    summary: dict[str, Any] = {"regime_name": regime_name, "age": age}

    main_keys = (
        "U_nan_fraction",
        "E_nan_fraction",
        "Q_nan_fraction",
        "F_feasible_fraction",
    )
    for key in main_keys:
        arr = np.asarray(diag[key])
        summary[key] = {
            "overall": float(np.mean(arr)),
            "by_dim": {
                name: np.mean(
                    arr, axis=tuple(j for j in range(arr.ndim) if j != i)
                ).tolist()
                for i, name in enumerate(state_names)
                if i < arr.ndim
            },
        }

    summary["regime_probs"] = {}
    summary["per_target_E_nan"] = {}
    for key, val in diag.items():
        if key.startswith("regime_prob__"):
            target = key.removeprefix("regime_prob__")
            summary["regime_probs"][target] = float(np.mean(np.asarray(val)))
        elif key.startswith("target_E_nan__"):
            target = key.removeprefix("target_E_nan__")
            summary["per_target_E_nan"][target] = float(np.mean(np.asarray(val)))

    return summary


def _format_diagnostic_summary(summary: dict[str, Any]) -> str:
    """Format diagnostic summary for exception note."""
    lines = [
        f"\nDiagnostics for regime '{summary['regime_name']}' at age {summary['age']}:",
    ]

    u = summary.get("U_nan_fraction", {}).get("overall", 0)
    e = summary.get("E_nan_fraction", {}).get("overall", 0)
    f_feas = summary.get("F_feasible_fraction", {}).get("overall", 0)
    lines.append(f"  U: {u:.4f} NaN  |  E[V]: {e:.4f} NaN  |  F: {f_feas:.4f} feasible")

    probs = summary.get("regime_probs", {})
    if probs:
        prob_parts = [f"{t}: {p:.4f}" for t, p in probs.items()]
        lines.append(f"  Regime probs: {' | '.join(prob_parts)}")

    per_target = summary.get("per_target_E_nan", {})
    nan_targets = {t: f for t, f in per_target.items() if f > 0}
    if nan_targets:
        parts = [f"{t}: {f:.4f}" for t, f in nan_targets.items()]
        lines.append(f"  Per-target E[V] NaN: {' | '.join(parts)}")

    # Show marginals for the first intermediate with any NaN
    for label, key in (("U", "U_nan_fraction"), ("E[V]", "E_nan_fraction")):
        info = summary.get(key, {})
        frac = info.get("overall", 0)
        by_dim = info.get("by_dim", {})
        if frac > 0 and by_dim:
            lines.append(f"  {label} NaN fraction by state:")
            for dim_name, values in by_dim.items():
                max_shown = 8
                formatted = ", ".join(f"{v:.2f}" for v in values[:max_shown])
                suffix = ", ..." if len(values) > max_shown else ""
                lines.append(f"    {dim_name:24s} [{formatted}{suffix}]")
            break

    return "\n".join(lines)
