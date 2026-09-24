"""Host-phase records of one public solve or simulate, for a stamped log to time.

A phase is bracketed by two INFO records carrying the public call's id. The
end record states the phase's outcome and its monotonic duration, so a log
read afterwards can reconcile the top-level children of the public bracket to
the whole call and keep the remainder as an explicit residual. A phase opened
inside another is nested one level deeper and is already part of its parent's
time.

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
- `solve_snapshot` — only when a diagnostic snapshot is written
- `result_assembly` — binding the generated authority and building the result
- `result_readiness` — waiting for the returned values' device work to finish

A public simulate is bracketed by `public_simulate`. Its top-level phases, in
emission order:

- `params_validation` — capturing the entry inputs, processing the user's
  parameters and resolving process grids
- `solution_resolution` — only with a supplied solution: validating and
  projecting it
- `simulation_inputs` — converting, canonicalising, padding and validating the
  initial conditions
- only without a supplied solution: the internal solve's phases,
  `authority_fingerprint` through `result_assembly`, followed by
  `solution_resolution` — resolving the solved result for simulation
- `solution_handover` — checking the resolved inputs and handing them to the
  entry allocations
- `chunk_planning` — cohort and chunk planning, which in a budgeted run
  compiles every profiled simulation candidate
- `simulation_setup` — the simulation memory, call inputs and plan summary
- `simulation_chunk` — one per subject chunk, the forward period loop
- `simulation_completion` — concatenating the chunks and waiting for them
- `simulation_result` — trimming padding and building the result
- `result_finalization` — releasing entry allocations and writing a simulate
  snapshot when one is due

Each simulation executable compiled during the call is bracketed as a nested
`simulation_compilation` phase inside whichever top-level phase compiled it.

A reader reconciles the top-level children against the public bracket and
reports what is left over as an explicit residual rather than normalising it
away.
"""

import contextlib
import contextvars
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

# The innermost open phase's call, so code reached without a call id can still
# bracket a nested phase of the call it runs under.
_OPEN_CALL: contextvars.ContextVar[tuple[CallId, logging.Logger] | None] = (
    contextvars.ContextVar("_OPEN_CALL", default=None)
)


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
    token = _OPEN_CALL.set((call_id, logger))
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        _OPEN_CALL.reset(token)
        logger.info(
            "solve call %s phase %s end status=%s seconds=%.6f",
            call_id,
            name,
            status,
            time.monotonic() - start,
        )


@contextlib.contextmanager
def nested_phase(*, name: str) -> Iterator[None]:
    """Bracket a phase inside whichever call's phase is open in this context.

    Outside any open phase — and on a thread the call's context did not reach —
    it emits nothing.

    Args:
        name: The phase's name in the record vocabulary.

    Yields:
        `None`, for the duration of the phase.

    """
    open_call = _OPEN_CALL.get()
    if open_call is None:
        yield
        return
    call_id, logger = open_call
    with solve_phase(name=name, logger=logger, call_id=call_id):
        yield
