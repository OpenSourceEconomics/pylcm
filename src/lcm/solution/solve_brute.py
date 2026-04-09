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
                diag_func = internal_regime.solve_functions.diagnostic_Q_and_F.get(
                    period
                )
                if diag_func is not None:
                    # Disable JIT to avoid XLA compilation OOM — the
                    # diagnostic returns multiple full-grid arrays, making
                    # the XLA computation graph much larger than max_Q_over_a.
                    with jax.disable_jit():
                        diag = diag_func(
                            **state_action_space.states,
                            **state_action_space.actions,
                            next_regime_to_V_arr=next_regime_to_V_arr,
                            **internal_params[name],
                        )
                    dim_names = (
                        *state_action_space.state_names,
                        *state_action_space.action_names,
                    )
                    exc.diagnostics = _summarize_diagnostics(
                        diag,
                        dim_names,
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
    dim_names: tuple[str, ...],
    *,
    regime_name: str,
    age: float,
) -> dict[str, Any]:
    """Reduce full-grid diagnostic arrays to marginal NaN fractions."""
    summary: dict[str, Any] = {"regime_name": regime_name, "age": age}

    for key in ("U_arr", "F_arr", "E_next_V", "Q_arr"):
        arr = np.asarray(diag[key])
        nan_mask = np.isnan(arr)
        all_nan = bool(nan_mask.all())
        summary[key] = {
            "nan_fraction": float(np.mean(nan_mask)),
            "min": None if all_nan else float(np.nanmin(arr)),
            "max": None if all_nan else float(np.nanmax(arr)),
            "by_dim": {
                name: np.mean(
                    nan_mask, axis=tuple(j for j in range(arr.ndim) if j != i)
                ).tolist()
                for i, name in enumerate(dim_names)
                if i < arr.ndim
            },
        }

    summary["F_feasible_fraction"] = float(np.mean(np.asarray(diag["F_arr"])))
    summary["regime_transition_probs"] = {
        target: float(np.mean(np.asarray(prob)))
        for target, prob in diag.get("regime_transition_probs", {}).items()
    }
    summary["per_target_E_next_V"] = {
        target: float(np.mean(np.isnan(np.asarray(arr))))
        for target, arr in diag.get("per_target_E_next_V", {}).items()
    }
    return summary


def _format_diagnostic_summary(summary: dict[str, Any]) -> str:
    """Format diagnostic summary for exception note."""
    lines = [
        f"\nDiagnostics for regime '{summary['regime_name']}' at age {summary['age']}:",
    ]

    u = summary.get("U_arr", {})
    e = summary.get("E_next_V", {})
    f_feas = summary.get("F_feasible_fraction", 0.0)
    lines.append(
        f"  U_arr: {u.get('nan_fraction', 0):.2f} NaN  |  "
        f"E_next_V: {e.get('nan_fraction', 0):.2f} NaN  |  "
        f"F_arr: {f_feas:.2f} feasible"
    )

    probs = summary.get("regime_transition_probs", {})
    if probs:
        prob_parts = [f"{t}: {p:.4f}" for t, p in probs.items()]
        lines.append(f"  Regime transition probs: {' | '.join(prob_parts)}")

    per_target = summary.get("per_target_E_next_V", {})
    if per_target:
        nan_targets = {t: f for t, f in per_target.items() if f > 0}
        if nan_targets:
            parts = [f"{t}: {f:.2f} NaN" for t, f in nan_targets.items()]
            lines.append(f"  Per-target E_next_V NaN: {' | '.join(parts)}")

    # Show marginals for the first intermediate that has partial NaN
    for key in ("U_arr", "E_next_V", "Q_arr"):
        info = summary.get(key, {})
        frac = info.get("nan_fraction", 0)
        by_dim = info.get("by_dim", {})
        if 0 < frac < 1 and by_dim:
            lines.append(f"  NaN fraction of {key} by dim:")
            for dim_name, values in by_dim.items():
                max_shown = 8
                formatted = ", ".join(f"{v:.2f}" for v in values[:max_shown])
                suffix = ", ..." if len(values) > max_shown else ""
                lines.append(f"    {dim_name:20s} [{formatted}{suffix}]")
            break
        if frac == 1.0 and by_dim:
            lines.append(f"  {key} is all NaN across all dims")
            break

    return "\n".join(lines)
