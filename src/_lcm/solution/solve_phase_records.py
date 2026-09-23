"""Host-phase records of one public solve, for a stamped log to time.

A phase is bracketed by two INFO records carrying the public call's id. The
end record states the phase's outcome and its monotonic duration, so a log
read afterwards can reconcile the children of `public_solve` to the whole
call and keep the remainder as an explicit residual.

A phase entered without a call id emits nothing, so an internal entry point
reached outside a public call stays silent while its code path is shared.

The phases a public solve passes through, in emission order, all nested
inside `public_solve`:

- `params_validation` — processing the user's parameters and the transitions
- `authority_fingerprint` — the declared solution authority and the model's
  durable fingerprint
- `solver_param_checks` — each regime's solver parameters and Pareto weights
- `state_action_spaces` — the base state-action spaces and the collision fence
- `continuation_templates` — the continuation and value templates
- `program_graphs` — every kernel's native program, narrowed to the retention
- `structural_resolution` — output layouts, lowering keys and fallback binding
- `residency_inventory` — what the plan already keeps on each device
- `compilation_waves` — lowering and compiling each wave of candidates
- `workspace_selection` — the planner's width choice per core
- `backward_induction` — the period loop, as host wall including every wait
- `result_assembly` — binding the generated authority and building the result
- `result_readiness` — waiting for the returned values' device work to finish

A reader reconciles the children against `public_solve` and reports what is
left over as an explicit residual rather than normalising it away.
"""

import contextlib
import logging
import time
import uuid
from collections.abc import Iterator

type CallId = str

# The phases nested inside `public_solve`, in emission order.
PHASE_NAMES = (
    "params_validation",
    "authority_fingerprint",
    "solver_param_checks",
    "state_action_spaces",
    "continuation_templates",
    "program_graphs",
    "structural_resolution",
    "residency_inventory",
    "compilation_waves",
    "workspace_selection",
    "backward_induction",
    "result_assembly",
    "result_readiness",
)

# Name of the bracket every other phase is nested inside.
PUBLIC_PHASE = "public_solve"


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
