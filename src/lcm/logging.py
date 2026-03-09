import logging


def get_logger(*, debug: bool) -> logging.Logger:
    """Get a logger that logs to stdout.

    Args:
        debug: Whether to log debug messages.

    Returns:
        Logger that logs to stdout.

    """
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("lcm")

    if debug:
        logger.setLevel(logging.DEBUG)

    return logger
