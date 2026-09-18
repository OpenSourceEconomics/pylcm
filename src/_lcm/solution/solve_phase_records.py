"""Host-phase records of one public solve, for a stamped log to time.

A phase is bracketed by two INFO records carrying the public call's id. The
end record states the phase's outcome and its monotonic duration, so a log
read afterwards can reconcile the children of `public_solve` to the whole
call and keep the remainder as an explicit residual.

A phase entered without a call id emits nothing, so an internal entry point
reached outside a public call stays silent while its code path is shared.
"""

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator

type CallId = str


def new_call_id() -> CallId:
    """Return a fresh identifier for one public call."""
    return uuid.uuid4().hex[:12]


@contextlib.contextmanager
def solve_phase(
    *, name: str, logger: logging.Logger, call_id: CallId | None
) -> Iterator[None]:
    """Bracket one host phase of a public solve with a begin and an end record.

    Args:
        name: The phase's name in the record vocabulary.
        logger: Logger carrying the call's verbosity policy.
        call_id: Identifier of the public call, or `None` to emit nothing.

    Yields:
        `None`, for the duration of the phase.

    """
    if call_id is None:
        yield
        return
    logger.info("solve call %s phase %s begin", call_id, name)
    start = time.monotonic()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        logger.info(
            "solve call %s phase %s end status=%s seconds=%.6f",
            call_id,
            name,
            status,
            time.monotonic() - start,
        )
