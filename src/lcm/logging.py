import logging
from typing import Literal

import jax.numpy as jnp

from lcm.typing import FloatND, Int1D, ScalarFloat

type LogLevel = Literal["off", "warning", "progress", "debug"]

_LOG_LEVEL_MAP: dict[str, int] = {
    "off": logging.CRITICAL,
    "warning": logging.WARNING,
    "progress": logging.INFO,
    "debug": logging.DEBUG,
}


def get_logger(*, log_level: LogLevel) -> logging.Logger:
    """Get a logger that logs to stdout.

    Args:
        log_level: Verbosity level. `"off"` suppresses all output, `"warning"` shows
            only warnings (e.g. NaN/Inf), `"progress"` adds timing per period,
            `"debug"` adds V_arr stats and feasibility info.

    Returns:
        Logger that logs to stdout.

    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("lcm")
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


def log_vf_nan(
    *,
    logger: logging.Logger,
    regime_name: str,
    age: ScalarFloat,
    V_arr: FloatND,
) -> None:
    """Log a warning if V_arr contains NaN or Inf values.

    Args:
        logger: Logger instance.
        regime_name: Name of the regime.
        age: Age corresponding to the current period.
        V_arr: Value function array to check.

    """
    if jnp.any(jnp.isnan(V_arr)) or jnp.any(jnp.isinf(V_arr)):
        logger.warning("NaN/Inf in V_arr for regime '%s' at age %s", regime_name, age)


def log_vf_stats(
    *,
    logger: logging.Logger,
    regime_name: str,
    V_arr: FloatND,
) -> None:
    """Log min/max/mean statistics of a value function array at debug level.

    Args:
        logger: Logger instance.
        regime_name: Name of the regime.
        V_arr: Value function array.

    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "  regime '%s': V min=%.3g max=%.3g mean=%.3g",
        regime_name,
        float(jnp.min(V_arr)),
        float(jnp.max(V_arr)),
        float(jnp.mean(V_arr)),
    )


def log_period_timing(
    *,
    logger: logging.Logger,
    age: ScalarFloat,
    n_active_regimes: int,
    elapsed: float,
) -> None:
    """Log period timing with regime count.

    Args:
        logger: Logger instance.
        age: Age corresponding to the current period.
        n_active_regimes: Number of active regimes in the period.
        elapsed: Elapsed time in seconds.

    """
    logger.info(
        "Age: %s  regimes=%d  (%s)",
        age,
        n_active_regimes,
        format_duration(seconds=elapsed),
    )


def log_regime_transitions(
    *,
    logger: logging.Logger,
    prev_regime_ids: Int1D,
    new_regime_ids: Int1D,
    ids_to_names: dict[int, str],
) -> None:
    """Log regime transition counts at debug level.

    Args:
        logger: Logger instance.
        prev_regime_ids: Regime IDs before the transition.
        new_regime_ids: Regime IDs after the transition.
        ids_to_names: Mapping from regime integer IDs to regime names.

    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    parts: list[str] = []
    for from_id, from_name in sorted(ids_to_names.items()):
        mask = prev_regime_ids == from_id
        if not jnp.any(mask):
            continue
        for to_id, to_name in sorted(ids_to_names.items()):
            count = int(jnp.sum(mask & (new_regime_ids == to_id)))
            if count > 0:
                parts.append(f"{from_name}\u2192{to_name}={count}")
    if parts:
        logger.debug("  transitions: %s", " ".join(parts))
