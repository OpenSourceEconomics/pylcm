"""Call-local output and runtime-validation policies, plus numerical log records."""

import logging
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from _lcm.typing import RegimeIdsToNames, RegimeName
from lcm.typing import (
    BoolND,
    FloatND,
    Int1D,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


@jax.jit
def v_array_has_nan(V_arr: FloatND) -> ScalarBool:
    """Return whether `V_arr` contains any NaN, sharded-safe.

    Putting the reduction inside `@jax.jit` keeps it in the XLA compiled graph, so
    GSPMD partitions it across the V-array's devices (per-device `any` → all-reduce
    → replicated scalar) and XLA fuses `isnan`+`any` into one pass. The
    eager-dispatch alternative `jnp.any(jnp.isnan(V))` materialises a full V-shaped
    bool intermediate and, on a sharded V-array, can fall back to gathering V onto
    the default device before reducing — a path that exhausts device memory at
    production grid sizes.
    """
    return jnp.any(jnp.isnan(V_arr))


@jax.jit
def v_array_has_inf(V_arr: FloatND) -> ScalarBool:
    """Return whether `V_arr` contains any +/-Inf, sharded-safe.

    Same compiled-graph rationale as `v_array_has_nan` — keeps the reduction
    partitioned across the V-array's devices instead of falling through a gather.
    """
    return jnp.any(jnp.isinf(V_arr))


@jax.jit
def non_finite_by_regime(
    *, values: tuple[FloatND, ...], in_regime: tuple[BoolND, ...]
) -> BoolND:
    """Return two flag rows over the regimes: NaN in row 0, NaN or Inf in row 1.

    The shape is `(2, n_regimes)`. Row 0 selects the regimes whose values need
    the enriched value-function report, which speaks only about NaN; row 1
    selects the regimes to warn about, since an Inf is worth reporting too.

    Out-of-regime rows carry placeholder values — possibly `-inf`, because the
    subject's state is infeasible under another regime's problem — so each
    array is masked by its own ownership flags before the reductions.

    Masking and reducing inside one compiled function keeps a whole period's
    check in a single program: every active regime is reduced on the device and
    both flag rows cross to the host together, instead of once per regime. On a
    sharded value array the reductions stay partitioned, for the same reason
    `v_array_has_nan` is jit-wrapped.
    """
    owned = [
        _owned_values(value=value, in_regime=mask)
        for value, mask in zip(values, in_regime, strict=True)
    ]
    return jnp.stack(
        [
            jnp.stack([jnp.any(jnp.isnan(value)) for value in owned]),
            jnp.stack([~jnp.all(jnp.isfinite(value)) for value in owned]),
        ]
    )


def _owned_values(*, value: FloatND, in_regime: BoolND) -> FloatND:
    """Replace the rows a regime does not own with zero.

    A collective regime's value carries a trailing stakeholder axis, so the
    per-subject ownership flags gain trailing singleton axes to broadcast
    against it; a singleton regime's value is already per-subject and the
    reshape is a no-op.
    """
    owned = in_regime.reshape(in_regime.shape + (1,) * (value.ndim - in_regime.ndim))
    return jnp.where(owned, value, 0.0)


type LogLevel = Literal["off", "warning", "progress", "debug"]

_LOG_LEVEL_MAP: dict[str, int] = {
    "off": logging.CRITICAL,
    "warning": logging.WARNING,
    "progress": logging.INFO,
    "debug": logging.DEBUG,
}


class _RunLogger(logging.Logger):
    """Call-local output level and independent numerical-validation policy.

    Keeping the policy on the existing logger seam preserves the solver ABI;
    separate instances prevent one call from changing another call's policy.
    """

    def __init__(self, *, log_level: LogLevel, runtime_checks: bool) -> None:
        super().__init__(name="lcm", level=_LOG_LEVEL_MAP[log_level])
        self.runtime_checks = runtime_checks
        self.parent = logging.getLogger("lcm")


def validation_enabled(logger: logging.Logger) -> bool:
    """Return the call's explicit numerical-validation policy."""
    return logger.runtime_checks if isinstance(logger, _RunLogger) else True


def validation_raises(logger: logging.Logger) -> bool:
    """Enabled numerical validation raises independently of output verbosity."""
    return validation_enabled(logger)


def raise_or_warn(*, logger: logging.Logger, error: Exception) -> None:
    """Raise an enabled validation failure; disabled checks publish nothing."""
    if validation_enabled(logger):
        raise error


def get_logger(*, log_level: LogLevel, runtime_checks: bool = True) -> logging.Logger:
    """Build a call-local logger sharing the application's output handlers.

    Args:
        log_level: Console verbosity only.
        runtime_checks: Whether numerical validation runs and raises on failure.

    Returns:
        Logger carrying independent output and numerical-validation controls.
    """
    parent = logging.getLogger("lcm")
    if not parent.handlers:
        parent.addHandler(logging.StreamHandler())
    # Planner emitters share this output logger; numerical policy stays call-local.
    parent.setLevel(_LOG_LEVEL_MAP[log_level])
    return _RunLogger(log_level=log_level, runtime_checks=runtime_checks)


def format_duration(*, seconds: float) -> str:
    """Format a duration in human-readable form.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string, e.g. "1.2ms", "3.4s", "2.1min", "1.5h".

    """
    _seconds_per_minute = 60
    _seconds_per_hour = 3600
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < _seconds_per_minute:
        return f"{seconds:.1f}s"
    if seconds < _seconds_per_hour:
        return f"{seconds / _seconds_per_minute:.1f}min"
    return f"{seconds / _seconds_per_hour:.1f}h"


def log_non_finite_values(
    *,
    logger: logging.Logger,
    age: float | ScalarInt | ScalarFloat,
    regime_names: tuple[RegimeName, ...],
    flags: tuple[bool, ...],
    time_kind: Literal["age", "period"] = "age",
) -> None:
    """Warn once for each regime holding a NaN or an Inf among the values it owns.

    Args:
        logger: Logger instance.
        age: Age corresponding to the current period.
        regime_names: Names of the regimes the flags belong to, in flag order.
        flags: Whether each regime owns a non-finite value, already on the host
            so that reporting a whole period costs no further device transfer.
        time_kind: Whether the coordinate is an age or a computational period.

    """
    for regime_name, flag in zip(regime_names, flags, strict=True):
        if flag:
            logger.warning(
                "NaN/Inf in V_arr for regime '%s' at %s %s", regime_name, time_kind, age
            )


def log_period_header(
    *,
    logger: logging.Logger,
    age: float | ScalarInt | ScalarFloat,
    n_active_regimes: int,
    time_kind: Literal["age", "period"] = "age",
) -> None:
    """Log the start of a period.

    Args:
        logger: Logger instance.
        age: Age corresponding to the current period.
        n_active_regimes: Number of active regimes in the period.
        time_kind: Whether the coordinate is an age or a computational period.

    """
    logger.info("%s %s (%d regimes):", time_kind.capitalize(), age, n_active_regimes)


def log_period_timing(
    *,
    logger: logging.Logger,
    elapsed: float,
) -> None:
    """Log period elapsed time.

    Args:
        logger: Logger instance.
        elapsed: Elapsed time in seconds.

    """
    logger.info("  finished in %s", format_duration(seconds=elapsed))


def log_regime_transition_counts(
    *,
    logger: logging.Logger,
    prev_regime_ids: Int1D,
    new_regime_ids: Int1D,
    regime_ids_to_names: RegimeIdsToNames,
    counts_factory: Callable[[], list[list[int]]] | None = None,
) -> None:
    """Log regime transition counts at debug level.

    Builds the full `(n_regimes, n_regimes)` transition count matrix in a
    single fused JAX kernel, then host-transfers the matrix once per period.

    Args:
        logger: Logger instance.
        prev_regime_ids: Regime IDs before the transition.
        new_regime_ids: Regime IDs after the transition.
        regime_ids_to_names: Immutable mapping of regime integer IDs to regime names.
        counts_factory: Optional call-local admitted count operation. Called only
            at debug level; the default retains the ordinary count path.

    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    sorted_ids = sorted(regime_ids_to_names.keys())
    if counts_factory is None:
        id_array = jnp.array(sorted_ids)
        from_one_hot = prev_regime_ids[:, None] == id_array[None, :]
        to_one_hot = new_regime_ids[:, None] == id_array[None, :]
        counts = (from_one_hot[:, :, None] & to_one_hot[:, None, :]).sum(axis=0)
        counts_host = counts.tolist()
    else:
        counts_host = counts_factory()

    parts: list[str] = []
    for i, from_id in enumerate(sorted_ids):
        for j, to_id in enumerate(sorted_ids):
            count = counts_host[i][j]
            if count > 0:
                parts.append(
                    f"  - {regime_ids_to_names[from_id]} \u2192 "
                    f"{regime_ids_to_names[to_id]} = {count}"
                )
    if parts:
        logger.debug("  transitions:\n%s", "\n".join(parts))
