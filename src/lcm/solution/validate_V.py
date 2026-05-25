"""Value-function NaN check fired during solve and simulate.

`validate_V` runs after each backward-induction period in `solve_brute.py`
and once on the V handed to `simulate.py`. On NaN it invokes the
diagnostic-intermediates closure (built during regime canonicalization in
`regime_building/diagnostics.py`) to pinpoint which intermediate
(`U`, `F`, `E[V]`, `Q`) produced the NaN, then raises an
`InvalidValueFunctionError` enriched with that breakdown.

"""

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp

from lcm.exceptions import InvalidValueFunctionError
from lcm.interfaces import StateActionSpace
from lcm.typing import (
    FlatRegimeParams,
    FloatND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
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
        dict with per-metric `"overall"` and `"by_dim"` entries plus a
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
