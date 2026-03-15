import logging
from typing import Literal

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
