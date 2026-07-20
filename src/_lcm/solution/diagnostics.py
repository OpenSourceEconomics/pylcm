"""Failure-path and logging diagnostics for backward induction.

The hot loop folds two async NaN/Inf reductions (plus, at debug, a
min/max/mean trio) into the accumulators initialized here; after the loop,
`emit_post_loop_diagnostics` decides on one host scalar per flag whether to
enter the per-row failure path — raising on the first NaN row with the
enriched `validate_V` breakdown, warning per Inf row, and logging the debug
stats. A healthy solve materialises nothing per row.
"""

import logging
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp

from _lcm.engine import Regime
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
)
from _lcm.solution.validate_V import validate_V
from _lcm.typing import FlatParams, RegimeName
from _lcm.utils.logging import v_array_has_inf, v_array_has_nan
from lcm.ages import AgeGrid
from lcm.typing import BoolND, FloatND


@dataclass(frozen=True)
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata — no device-array references — so
    every (regime, period) row stays at a few bytes regardless of grid
    size. State-action space, next-period V mapping, regime params, and
    the `compute_intermediates` closure are reconstructed lazily on the
    failure path from `regimes`, `flat_params`, and the
    partial `solution` built up to that point.
    """

    regime_name: RegimeName
    """Name of the regime whose V-array this row summarises."""
    period: int
    """Period index in the backward-induction loop."""
    age: float
    """Age corresponding to `period` (pulled off `AgeGrid.values`)."""


def _init_diagnostic_accumulators() -> tuple[
    list[_DiagnosticRow],
    list[FloatND],
    list[FloatND],
    list[FloatND],
    BoolND,
    BoolND,
]:
    """Initialize the per-period async diagnostics accumulators.

    Returns the empty diagnostic-row, min, max, and mean lists, and the two
    running NaN/Inf flag scalars (folded into across the backward-induction
    loop). The two flags share the same immutable zero scalar initially; each
    is reassigned independently inside the loop.
    """
    zero: BoolND = jnp.zeros((), dtype=bool)
    rows: list[_DiagnosticRow] = []
    mins: list[FloatND] = []
    maxs: list[FloatND] = []
    means: list[FloatND] = []
    return rows, mins, maxs, means, zero, zero


def _fold_period_diagnostics(
    *,
    V_arr: FloatND,
    regime_name: RegimeName,
    period: int,
    ages: AgeGrid,
    diagnostics_enabled: bool,
    stats_enabled: bool,
    diagnostic_rows: list[_DiagnosticRow],
    diagnostic_min: list[FloatND],
    diagnostic_max: list[FloatND],
    diagnostic_mean: list[FloatND],
    running_any_nan: BoolND,
    running_any_inf: BoolND,
) -> tuple[BoolND, BoolND]:
    """Fold one regime-period's V array into the diagnostics accumulators.

    Async reductions: gated on the public log level.

    - validation `"off"` (`diagnostics_enabled=False`) ⇒ nothing; the flags pass
      through unchanged
    - `"warning"` / `"progress"` ⇒ folds two cheap isnan/isinf reductions into
      the running scalars
    - `"debug"` (`stats_enabled`) ⇒ adds the min/max/mean trio

    Each extra full-V read is a memory-bandwidth tax on the larger models, so
    the default keeps it to two reductions per (regime, period).

    Returns:
        Tuple of the updated running NaN and Inf flag scalars.

    """
    if not diagnostics_enabled:
        return running_any_nan, running_any_inf
    if stats_enabled:
        diagnostic_min.append(jnp.min(V_arr))
        diagnostic_max.append(jnp.max(V_arr))
        diagnostic_mean.append(jnp.mean(V_arr))
    diagnostic_rows.append(
        _DiagnosticRow(
            regime_name=regime_name,
            period=period,
            age=float(ages.values[period]),
        )
    )
    return (
        running_any_nan | v_array_has_nan(V_arr),
        running_any_inf | v_array_has_inf(V_arr),
    )


def _emit_post_loop_diagnostics(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    running_any_nan: BoolND,
    running_any_inf: BoolND,
    diagnostic_min: list[FloatND] | None,
    diagnostic_max: list[FloatND] | None,
    diagnostic_mean: list[FloatND] | None,
) -> None:
    """Flush async diagnostics: raise on NaN, warn on Inf, log debug stats.

    Only enters the per-row failure path when the running NaN or Inf
    accumulators are set, so a healthy solve incurs no host-side scalar
    materialisation here.
    """
    if running_any_nan.item():
        _raise_first_nan_row(
            diagnostic_rows=diagnostic_rows,
            solution=solution,
            regimes=regimes,
            flat_params=flat_params,
        )
    if running_any_inf.item():
        _warn_inf_rows(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            solution=solution,
        )
    if diagnostic_min is not None and diagnostic_max is not None and diagnostic_mean:
        _log_per_period_stats(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            mins=jnp.stack(diagnostic_min),
            maxs=jnp.stack(diagnostic_max),
            means=jnp.stack(diagnostic_mean),
        )


def _raise_first_nan_row(
    *,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Find the first NaN-bearing (regime, period) and raise.

    Failure-path only — walks rows until the first NaN hit.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isnan(V_arr)).item():
            _raise_at(
                row=row,
                solution=solution,
                regimes=regimes,
                flat_params=flat_params,
            )


def _raise_at(
    *,
    row: _DiagnosticRow,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Run the enriched NaN diagnostic on a single offending row and raise."""
    regime = regimes[row.regime_name]
    regime_params = flat_params[row.regime_name]
    # `compute_intermediates` was built from the regime's full `flat_param_names`
    # (per-iteration params + fixed params); the live solve loop merges
    # `resolved_fixed_params` into `regime_params` implicitly via the partialled
    # closures, but we have to do it by hand here to call the diagnostic
    # directly. Same merge order as `engine.state_action_space` and
    # `simulation.result`.
    effective_regime_params = MappingProxyType(
        {**regime.resolved_fixed_params, **regime_params}
    )
    state_action_space = regime.solution.state_action_space(regime_params=regime_params)
    next_regime_to_V_arr = _reconstruct_next_regime_to_V_arr(
        period=row.period,
        regimes=regimes,
        flat_params=flat_params,
        solution=solution,
    )
    # The intermediates closure mirrors the brute-force Q evaluation; for a
    # regime solved from interpolated continuations it cannot reproduce the
    # failing computation, so the error is raised without the U/F/E/Q
    # breakdown.
    compute_intermediates = (
        None
        if regime.solution.solves_from_continuation
        else regime.solution.compute_intermediates.get(row.period)
    )
    V_arr = solution[row.period][row.regime_name]
    validate_V(
        V_arr=V_arr,
        age=row.age,
        regime_name=row.regime_name,
        partial_solution=solution,
        compute_intermediates=compute_intermediates,
        state_action_space=state_action_space,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=effective_regime_params,
        period=row.period,
    )


def _reconstruct_next_regime_to_V_arr(
    *,
    period: int,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> MappingProxyType[RegimeName, FloatND]:
    """Recreate the rolling `next_regime_to_V_arr` that was used at `period`.

    The hot loop rolls the per-regime V forward via `period_solution.get(name,
    next_regime_to_V_arr[name])`, so at iteration `period` each regime's slot
    holds its V from the smallest later period where it was active, falling
    back to a zeros template otherwise.

    Rebuild the same mapping post-hoc from `solution`. Shape and device
    sharding both come from `_get_regime_V_shapes_and_shardings` so the
    reconstructed templates have the same pytree structure and placement as
    the live ones in `solve()`.
    """
    regime_V_topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
    )
    later_periods = sorted(p for p in solution if p > period)
    result: dict[RegimeName, FloatND] = {}
    for regime_name, topology in regime_V_topology.items():
        rolled: FloatND | None = None
        for q in later_periods:
            if regime_name in solution[q]:
                rolled = solution[q][regime_name]
                break
        result[regime_name] = (
            rolled if rolled is not None else _build_zero_V_arr(topology=topology)
        )
    return MappingProxyType(result)


def _warn_inf_rows(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> None:
    """Emit a warning per (regime, period) with Inf values.

    Only invoked on the failure path (`running_any_inf` was True).
    Materialises one host-side bool per row.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isinf(V_arr)).item():
            logger.warning(
                "Inf in V_arr for regime '%s' at age %s",
                row.regime_name,
                row.age,
            )


def _log_per_period_stats(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    mins: FloatND,
    maxs: FloatND,
    means: FloatND,
) -> None:
    """Emit one debug log line per (regime, period) with V min/max/mean."""
    for row, V_min, V_max, V_mean in zip(
        diagnostic_rows, mins.tolist(), maxs.tolist(), means.tolist(), strict=True
    ):
        logger.debug(
            "  %s  age %s   V min=%.3g  max=%.3g  mean=%.3g",
            row.regime_name,
            row.age,
            V_min,
            V_max,
            V_mean,
        )
