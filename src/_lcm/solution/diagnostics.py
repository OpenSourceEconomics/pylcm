"""Failure-path and logging diagnostics for backward induction.

The hot loop folds two async NaN/Inf reductions (plus, at debug, a
min/max/mean trio) into the accumulators initialized here and keeps each row's
own pair of flags, which become host booleans at the period barrier. At the
first period with a NaN row, the enriched `validate_V` breakdown runs right
there, while that period's values and the next-period values it read are still
on the device. After the loop, `emit_post_loop_diagnostics` raises that error,
warns per Inf row from the row flags, and logs the debug stats, so it reads no
value back, wherever the solve stores them.
"""

import dataclasses
import logging
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp

from _lcm.engine import Regime, StateActionSpace
from _lcm.params.edges import regime_kernel_params
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.solution.feasibility import ContinuationReadCounts, FeasibilityCounts
from _lcm.solution.host_values import device_read_view
from _lcm.solution.validate_V import validate_V
from _lcm.time import TimeAxis, coordinate_kind
from _lcm.transition_plans import InterpolationBasisInfo
from _lcm.typing import FlatParams, PeriodToRegimeToVArr, RegimeName
from _lcm.utils.logging import v_array_has_inf, v_array_has_nan
from lcm._solver_api.entries import _LazyEntry
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import BoolND, FloatND, IntND, StateName


@dataclass(frozen=True)
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Native scalar counters and the NaN/Inf flags become host metadata at the
    period barrier, so completed rows retain no device buffers. State-action
    space, regime params and the `compute_intermediates` closure are
    reconstructed on the failure path from `regimes` and `flat_params`, at the
    barrier of the first period with a NaN row.
    """

    regime_name: RegimeName
    """Name of the regime whose V-array this row summarises."""
    period: int
    """Period index in the backward-induction loop."""
    age: float
    """Internal time coordinate, read from exact host values."""
    time_kind: Literal["age", "period"] = "age"
    """Meaning of the coordinate in reports."""

    feasibility_counts: FeasibilityCounts | tuple[int, int] | None = None
    """Native counts, moved to host metadata at the period's existing barrier."""
    continuation_read_counts: ContinuationReadCounts | tuple[int, int] | None = None
    """One paired-read summary, copied once at the period barrier."""
    has_nan: BoolND | bool | None = None
    """Whether the row's V holds a NaN; a host bool once the barrier has passed."""
    has_inf: BoolND | bool | None = None
    """Whether the row's V holds an Inf; a host bool once the barrier has passed."""


type FeasibilityWarnings = dict[_DiagnosticRow, tuple[int, int]]

_component_feasibility_warnings: ContextVar[FeasibilityWarnings | None] = ContextVar(
    "component_feasibility_warnings", default=None
)


@contextmanager
def collect_component_feasibility_warnings(
    *,
    counts: FeasibilityWarnings,
) -> Generator[None]:
    """Combine host counts over a block-major schedule without retaining values."""
    token = _component_feasibility_warnings.set(counts)
    try:
        yield
    finally:
        _component_feasibility_warnings.reset(token)


def emit_component_feasibility_warnings(
    *, logger: logging.Logger, counts: FeasibilityWarnings, scope: str = ""
) -> None:
    """Report each completed regime-period once, then release host metadata.

    A non-empty `scope` is appended to every message to name the part of the
    model the counts cover.
    """
    if logger.isEnabledFor(logging.WARNING):
        _warn_inf_rows(
            logger=logger,
            diagnostic_rows=[
                dataclasses.replace(row, feasibility_counts=value)
                for row, value in counts.items()
            ],
            scope=scope,
        )
    counts.clear()


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
    ages: TimeAxis,
    diagnostics_enabled: bool,
    stats_enabled: bool,
    diagnostic_rows: list[_DiagnosticRow],
    diagnostic_min: list[FloatND],
    diagnostic_max: list[FloatND],
    diagnostic_mean: list[FloatND],
    running_any_nan: BoolND,
    running_any_inf: BoolND,
    feasibility_counts: FeasibilityCounts | None = None,
    continuation_read_counts: ContinuationReadCounts | None = None,
) -> tuple[BoolND, BoolND]:
    """Fold one regime-period's V array into the diagnostics accumulators.

    Runtime checks fold NaN/Inf reductions into the running scalars. Debug
    output independently adds the min/max/mean reductions. Each extra full-V
    read is a memory-bandwidth tax on larger models.

    The row's age is read from the grid's exact host-side values: a read off
    the device array here would be a blocking device-to-host copy queued
    behind the period's dispatch, charged to the host loop.

    Returns:
        Tuple of the updated running NaN and Inf flag scalars.

    """
    if not diagnostics_enabled and not stats_enabled:
        return running_any_nan, running_any_inf
    if stats_enabled:
        diagnostic_min.append(
            _place_like(scalar=jnp.min(V_arr), running=running_any_nan)
        )
        diagnostic_max.append(
            _place_like(scalar=jnp.max(V_arr), running=running_any_nan)
        )
        diagnostic_mean.append(
            _place_like(scalar=jnp.mean(V_arr), running=running_any_nan)
        )
    has_nan = has_inf = None
    if diagnostics_enabled:
        has_nan = _place_like(scalar=v_array_has_nan(V_arr), running=running_any_nan)
        has_inf = _place_like(scalar=v_array_has_inf(V_arr), running=running_any_inf)
    diagnostic_rows.append(
        _DiagnosticRow(
            regime_name=regime_name,
            period=period,
            age=float(ages.exact_values[period]),
            time_kind=coordinate_kind(ages),
            feasibility_counts=feasibility_counts,
            continuation_read_counts=continuation_read_counts,
            has_nan=has_nan,
            has_inf=has_inf,
        )
    )
    if has_nan is None or has_inf is None:
        return running_any_nan, running_any_inf
    observed_inf = has_inf
    if feasibility_counts is not None:
        # Collective dissolution can produce finite fallback values.
        observed_inf = _fold_flag(
            running=observed_inf, observed=feasibility_counts[0] > 0
        )
    return running_any_nan | has_nan, running_any_inf | observed_inf


def _fold_flag(*, running: BoolND, observed: BoolND) -> BoolND:
    """OR one observed flag into the running flag regardless of placement.

    A value array is born wherever its program's planned output layout puts it: a
    terminal regime's value on one committed device, a distributed regime's value
    sharded across the mesh. Two scalars committed to different device sets cannot
    be combined directly, so the observed flag is moved onto the running flag's
    placement first. That is one scalar transfer per (regime, period); the
    reduction over the value array itself stays partitioned across its devices.
    """
    return running | _place_like(scalar=observed, running=running)


def _materialize_period_feasibility_counts(
    *, diagnostic_rows: list[_DiagnosticRow], period: int, enabled: bool
) -> None:
    """Move a period's counters and NaN/Inf flags to host metadata.

    Every row of the period crosses to the host in one transfer. Releases the
    native counter roots before another period starts.
    """
    indices = []
    for index in range(len(diagnostic_rows) - 1, -1, -1):
        if diagnostic_rows[index].period != period:
            break
        indices.append(index)
    on_device = [
        (
            row.feasibility_counts
            if row.feasibility_counts is not None and enabled
            else None,
            row.continuation_read_counts,
            None
            if row.has_nan is None or row.has_inf is None
            else (row.has_nan, row.has_inf),
        )
        for row in (diagnostic_rows[index] for index in indices)
    ]
    on_host = jax.device_get(on_device) if jax.tree.leaves(on_device) else on_device
    for index, (counts, reads, flags) in zip(indices, on_host, strict=True):
        diagnostic_rows[index] = dataclasses.replace(
            diagnostic_rows[index],
            feasibility_counts=None
            if counts is None
            else (int(counts[0]), int(counts[1])),
            continuation_read_counts=None
            if reads is None
            else (int(reads[0]), int(reads[1])),
            has_nan=None if flags is None else bool(flags[0]),
            has_inf=None if flags is None else bool(flags[1]),
        )


def _place_like(*, scalar: FloatND | BoolND, running: BoolND) -> FloatND | BoolND:
    """Move one reduced scalar onto the running-flag placement.

    Every per-period scalar the diagnostics keep — the NaN/Inf flags and the
    min/max/mean stats — is later combined across regimes and periods, so all of
    them share the running flag's placement rather than the placement of the
    value array they were reduced from.
    """
    return jax.device_put(scalar, running.sharding)


def _emit_post_loop_diagnostics(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    nan_error: InvalidValueFunctionError | None,
    running_any_inf: BoolND,
    diagnostic_min: list[FloatND] | None,
    diagnostic_max: list[FloatND] | None,
    diagnostic_mean: list[FloatND] | None,
) -> None:
    """Flush async diagnostics: raise on NaN, warn on Inf, log debug stats.

    The NaN error was built at the barrier of its period. Inf warnings are only
    entered when the running Inf accumulator is set, and read the row flags, so
    a healthy solve incurs no host-side scalar materialisation here.

    Raises:
        InvalidValueFunctionError: `nan_error`, when the solve found a NaN.

    """
    if nan_error is not None:
        raise nan_error
    if logger.isEnabledFor(logging.WARNING):
        for row in diagnostic_rows:
            if row.continuation_read_counts is not None:
                below, eligible = row.continuation_read_counts
                if below:
                    logger.warning(
                        "Regime '%s', period %s: %s of %s eligible continuation reads "
                        "lie below their first supported coordinate.",
                        row.regime_name,
                        row.period,
                        below,
                        eligible,
                    )
    if running_any_inf.item() and logger.isEnabledFor(logging.WARNING):
        _warn_inf_rows(logger=logger, diagnostic_rows=diagnostic_rows)
    if diagnostic_min is not None and diagnostic_max is not None and diagnostic_mean:
        _log_per_period_stats(
            logger=logger,
            diagnostic_rows=diagnostic_rows,
            mins=jnp.stack(diagnostic_min),
            maxs=jnp.stack(diagnostic_max),
            means=jnp.stack(diagnostic_mean),
        )


def period_nan_error(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    period: int,
    period_values: MappingProxyType[RegimeName, FloatND],
    next_values: MappingProxyType[RegimeName, FloatND],
    stored_values: Mapping[int, Mapping[RegimeName, FloatND | _LazyEntry]],
    store_on_host: bool,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> InvalidValueFunctionError | None:
    """Return the enriched error of the period's first NaN row, if it has one.

    Runs at the period barrier, after the row flags became host booleans, while
    the period's values and the next-period values its programs read are still
    on the device. The error's partial solution is the values solved so far,
    read back from value storage only if someone reads it.
    """
    rows = []
    # The period's rows are the last ones; walk back only through them.
    for row in reversed(diagnostic_rows):
        if row.period != period:
            break
        if row.has_nan:
            rows.insert(0, row)
    if not rows:
        return None
    partial = {**stored_values, period: period_values}
    try:
        _raise_first_nan_row(
            logger=logger,
            diagnostic_rows=rows,
            solution=MappingProxyType({period: period_values, period + 1: next_values}),
            partial_solution=(
                device_read_view(values=partial)
                if store_on_host
                else cast("PeriodToRegimeToVArr", MappingProxyType(partial))
            ),
            regimes=regimes,
            flat_params=flat_params,
            process_grid_resolver=process_grid_resolver,
        )
    except InvalidValueFunctionError as error:
        return error
    return None


def _raise_first_nan_row(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    partial_solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> None:
    """Find the first NaN-bearing (regime, period) and raise.

    Failure-path only — walks rows until the first NaN hit.
    """
    for row in diagnostic_rows:
        V_arr = solution[row.period][row.regime_name]
        if jnp.any(jnp.isnan(V_arr)).item():
            _raise_at(
                logger=logger,
                row=row,
                solution=solution,
                partial_solution=partial_solution,
                regimes=regimes,
                flat_params=flat_params,
                process_grid_resolver=process_grid_resolver,
            )


def _raise_at(
    *,
    logger: logging.Logger,
    row: _DiagnosticRow,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    partial_solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> None:
    """Run the enriched NaN diagnostic on a single offending row and raise."""
    regime = regimes[row.regime_name]
    regime_params = regime_kernel_params(flat_params, regime_name=row.regime_name)
    # `compute_intermediates` was built from the regime's full `flat_param_names`
    # (per-iteration params + fixed params); the live solve loop merges
    # `resolved_fixed_params` into `regime_params` implicitly via the partialled
    # closures, but we have to do it by hand here to call the diagnostic
    # directly. Same merge order as `engine.state_action_space` and
    # `simulation.result`.
    effective_regime_params = MappingProxyType(
        {**regime.resolved_fixed_params, **regime_params}
    )
    state_action_space = regime.solution.state_action_space(
        regime_params=regime_params, process_grid_resolver=process_grid_resolver
    )
    # The live solve tabulated period t on period t's grid, so the diagnostic has to as
    # well — otherwise a moving current-state grid reports U/F/E/Q fractions on state
    # values the failing solve never saw.
    state_action_space = dataclasses.replace(
        state_action_space,
        states=MappingProxyType(
            dict(
                _states_for_period(
                    regime=regime,
                    state_action_space=state_action_space,
                    period=row.period,
                )
            )
        ),
    )
    next_regime_to_V_arr = _reconstruct_next_regime_to_V_arr(
        period=row.period, solution=solution
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
        logger=logger,
        V_arr=V_arr,
        age=row.age,
        regime_name=row.regime_name,
        partial_solution=partial_solution,
        compute_intermediates=compute_intermediates,
        state_action_space=state_action_space,
        next_regime_to_V_arr=next_regime_to_V_arr,
        flat_params=effective_regime_params,
        period=row.period,
        time_kind=row.time_kind,
        entered_process_names=tuple(
            law.next_state_name.removeprefix("next_")
            for plan in regime.solution.transition_plans.values()
            for law in plan.outputs.values()
            if isinstance(law.continuation_coordinate, InterpolationBasisInfo)
        ),
    )


def _states_for_period(
    *, regime: Regime, state_action_space: StateActionSpace, period: int
) -> Mapping[StateName, FloatND | IntND]:
    """Current-period state axes, overriding age-varying states with period-t nodes.

    For a regime with `AgeSpecializedGrid` states, replace the representative base
    axis with this period's grid nodes so period-t's value function is tabulated on
    period-t's grid (consistent with the continuation interpolation, which reads
    V_{t+1} on period-(t+1)'s grid). Same shape as the base, so the shared compiled
    kernel is not retraced. Age-invariant regimes return the base axis unchanged.
    """
    # getattr (not direct access) so a duck-typed mock regime without the field works.
    axes = getattr(regime.solution, "period_state_axes", None)
    if axes is not None and period in axes:
        return {**state_action_space.states, **axes[period]}
    return state_action_space.states


def _reconstruct_next_regime_to_V_arr(
    *,
    period: int,
    solution: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
) -> MappingProxyType[RegimeName, FloatND]:
    """Recreate the rolling `next_regime_to_V_arr` that was used at `period`.

    The solve hands a period's programs the values of the regimes solved at
    `period + 1`, which are the only ones those programs read; the last period
    reads none.
    """
    return MappingProxyType(dict(solution.get(period + 1, {})))


def _warn_inf_rows(
    *,
    logger: logging.Logger,
    diagnostic_rows: list[_DiagnosticRow],
    scope: str = "",
) -> None:
    """Emit a warning per (regime, period) with Inf values.

    Only invoked on the failure path (`running_any_inf` was True). Reads the
    row flags the period barrier made host booleans.
    """
    for row in diagnostic_rows:
        if row.feasibility_counts is not None:
            empty_count, other_inf_count = (
                int(count) for count in row.feasibility_counts
            )
            collected = _component_feasibility_warnings.get()
            if collected is not None:
                key = dataclasses.replace(
                    row, feasibility_counts=None, has_nan=None, has_inf=None
                )
                empty, other = collected.get(key, (0, 0))
                collected[key] = empty + empty_count, other + other_inf_count
                continue
            if empty_count:
                logger.warning(
                    "%s state(s) with no feasible action for regime '%s', "
                    "period %s (%s %s)%s",
                    empty_count,
                    row.regime_name,
                    row.period,
                    row.time_kind,
                    row.age if row.time_kind == "age" else row.period,
                    scope,
                )
            if other_inf_count:
                logger.warning(
                    "Inf in V_arr for %s other state(s) for regime '%s', "
                    "period %s (%s %s)%s",
                    other_inf_count,
                    row.regime_name,
                    row.period,
                    row.time_kind,
                    row.age if row.time_kind == "age" else row.period,
                    scope,
                )
            continue
        if row.has_inf:
            logger.warning(
                "Inf in V_arr for regime '%s' at %s %s",
                row.regime_name,
                row.time_kind,
                row.age if row.time_kind == "age" else row.period,
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
            "  %s  %s %s   V min=%.3g  max=%.3g  mean=%.3g",
            row.regime_name,
            row.time_kind,
            row.age if row.time_kind == "age" else row.period,
            V_min,
            V_max,
            V_mean,
        )
