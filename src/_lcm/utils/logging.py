import logging
from typing import Literal

import jax
import jax.numpy as jnp

from _lcm.typing import RegimeIdsToNames
from lcm.typing import FloatND, Int1D, ScalarBool, ScalarFloat, ScalarInt


@jax.jit
def v_array_has_nan(V_arr: FloatND) -> ScalarBool:
    """Return whether `V_arr` contains any NaN, sharded-safe.

    Putting the reduction inside `@jax.jit` keeps it in the XLA compiled
    graph, so GSPMD partitions it across the V-array's devices (per-device
    `any` → all-reduce → replicated scalar) and XLA fuses `isnan`+`any`
    into one pass. The eager-dispatch alternative `jnp.any(jnp.isnan(V))`
    materialises a full V-shaped bool intermediate and, on a sharded
    V-array, can fall back to gathering V onto the default device before
    reducing — a path that exhausts device memory at production grid
    sizes.
    """
    return jnp.any(jnp.isnan(V_arr))


@jax.jit
def v_array_has_inf(V_arr: FloatND) -> ScalarBool:
    """Return whether `V_arr` contains any +/-Inf, sharded-safe.

    Same compiled-graph rationale as `v_array_has_nan` — keeps the
    reduction partitioned across the V-array's devices instead of
    falling through a gather.
    """
    return jnp.any(jnp.isinf(V_arr))


type LogLevel = Literal["off", "warning", "progress", "debug"]

_LOG_LEVEL_MAP: dict[str, int] = {
    "off": logging.CRITICAL,
    "warning": logging.WARNING,
    "progress": logging.INFO,
    "debug": logging.DEBUG,
}


def validation_enabled(logger: logging.Logger) -> bool:
    """Return whether runtime validation runs at all.

    Runtime validation runs unless `log_level="off"`. The logger's level is
    the single source of truth for the runtime policy: `"off"` raises the
    logger to `CRITICAL`, every other level keeps it at `WARNING` or lower.
    """
    return logger.isEnabledFor(logging.WARNING)


def validation_raises(logger: logging.Logger) -> bool:
    """Return whether a validation failure raises (vs. logs a warning).

    A failure raises at `log_level="debug"` and only warns at `"warning"` /
    `"progress"`. `"debug"` is the one level that lowers the logger to
    `DEBUG`, so `isEnabledFor(DEBUG)` is exactly the raise predicate.
    """
    return logger.isEnabledFor(logging.DEBUG)


def raise_or_warn(*, logger: logging.Logger, error: Exception) -> None:
    """Surface a validation failure according to the logger's policy.

    Raises the error when the logger implies raise mode (`log_level="debug"`);
    otherwise logs it as a warning and returns so the run continues. Must not
    be called when validation is disabled (`log_level="off"`).

    Args:
        logger: Logger carrying the runtime-validation policy.
        error: The validation error to raise or describe.

    Raises:
        Exception: The passed `error`, in raise mode.

    """
    if validation_raises(logger):
        raise error
    logger.warning("%s", error)


def get_logger(*, log_level: LogLevel) -> logging.Logger:
    """Get a logger that logs to stdout.

    Args:
        log_level: Verbosity level. `"off"` suppresses all output, `"warning"` shows
            only warnings (e.g. NaN/Inf), `"progress"` adds timing per period,
            `"debug"` adds V_arr stats and feasibility info.

    Returns:
        Logger that logs to stdout.

    """
    logger = logging.getLogger("lcm")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(_LOG_LEVEL_MAP[log_level])
    return logger


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


def log_nan_in_V(
    *,
    logger: logging.Logger,
    regime_name: str,
    age: float | ScalarInt | ScalarFloat,
    V_arr: FloatND,
) -> None:
    """Log a warning if V_arr contains NaN or Inf values.

    The reductions go through `v_array_has_nan` / `v_array_has_inf`
    so they stay sharded on distributed V-arrays. Callers gate this
    function at the `validation_enabled(logger)` level; the helper
    itself trusts the caller and always performs the check.

    Args:
        logger: Logger instance.
        regime_name: Name of the regime.
        age: Age corresponding to the current period.
        V_arr: Value function array to check.

    """
    if bool(v_array_has_nan(V_arr)) or bool(v_array_has_inf(V_arr)):
        logger.warning("NaN/Inf in V_arr for regime '%s' at age %s", regime_name, age)


def log_period_header(
    *,
    logger: logging.Logger,
    age: float | ScalarInt | ScalarFloat,
    n_active_regimes: int,
) -> None:
    """Log the start of a period.

    Args:
        logger: Logger instance.
        age: Age corresponding to the current period.
        n_active_regimes: Number of active regimes in the period.

    """
    logger.info("Age %s (%d regimes):", age, n_active_regimes)


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


def log_regime_transitions(
    *,
    logger: logging.Logger,
    prev_regime_ids: Int1D,
    new_regime_ids: Int1D,
    regime_ids_to_names: RegimeIdsToNames,
) -> None:
    """Log regime transition counts at debug level.

    Args:
        logger: Logger instance.
        prev_regime_ids: Regime IDs before the transition.
        new_regime_ids: Regime IDs after the transition.
        regime_ids_to_names: Immutable mapping of regime integer IDs to regime names.

    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    parts: list[str] = []
    for from_id, from_name in sorted(regime_ids_to_names.items()):
        mask = prev_regime_ids == from_id
        if not jnp.any(mask):
            continue
        for to_id, to_name in sorted(regime_ids_to_names.items()):
            count = int(jnp.sum(mask & (new_regime_ids == to_id)))
            if count > 0:
                parts.append(f"  - {from_name} \u2192 {to_name} = {count}")
    if parts:
        logger.debug("  transitions:\n%s", "\n".join(parts))
