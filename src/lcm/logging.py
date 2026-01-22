import logging


def get_logger(*, debug_mode: bool) -> logging.Logger:
    """Get a logger that logs to stdout.

    Args:
        debug_mode: Whether to log debug messages.

    Returns:
        Logger that logs to stdout.

    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("lcm")

    if debug_mode:
        logger.setLevel(logging.DEBUG)

    return logger
