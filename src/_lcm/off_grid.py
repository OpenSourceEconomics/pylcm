"""Raw continuous-state grid diagnostics, independent of clipping and verbosity."""

import logging
from collections.abc import Mapping

import jax

from _lcm.grids import ContinuousGrid, Grid
from _lcm.solution.contract import Solver
from _lcm.solution.dcegm import DCEGM
from _lcm.solution.egm import EGM
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import NEGM
from _lcm.solution.nnbegm import NNBEGM
from _lcm.utils.counts import COUNT_CAPACITY, sum_counts
from _lcm.utils.logging import validation_enabled
from lcm.exceptions import OffGridPolicyError, OffGridStateError
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    OffGrid,
    RegimeName,
    ScalarInt,
    StateName,
)


def fail_if_endogenous_grid_reads_a_clipped_grid(
    *,
    solvers: Mapping[RegimeName, Solver],
    grids: Mapping[RegimeName, tuple[tuple[StateName, Grid], ...]],
    targets_by_regime: Mapping[RegimeName, frozenset[RegimeName]],
) -> None:
    """Refuse a clipped grid that an endogenous-grid solver reads.

    EGM-family carries interpolate the target's continuation on the solver's own
    endogenous grid, and their policy banks read the source's own state grid;
    neither applies a grid's closed-endpoint lookup. Only `GridSearch` reads
    through the per-grid lookup rule, so a clipped grid on the source or on any
    of its targets is rejected here rather than silently extrapolated.
    """
    for source, solver in solvers.items():
        if not isinstance(solver, EGM | DCEGM | NBEGM | NEGM | NNBEGM):
            continue
        for regime in (source, *sorted(targets_by_regime.get(source, ()))):
            for state, grid in grids[regime]:
                rule = grid.off_grid if isinstance(grid, ContinuousGrid) else None
                if rule in ("clip_lookup", "clip_state"):
                    raise OffGridPolicyError(
                        f"The {type(solver).__name__} solver of regime "
                        f"{source!r} reads regime {regime!r}, whose state "
                        f"{state!r} declares off_grid={rule!r}. Endogenous-grid "
                        "solvers do not apply clipped lookup; declare "
                        "off_grid='raise' or 'extrapolate' on that grid, or "
                        f"solve {source!r} with GridSearch."
                    )


@jax.jit
def count_off_grid(*, values: FloatND, points: Float1D, feasible: BoolND) -> ScalarInt:
    """Count strict exterior raw values on the economically reached population."""
    return sum_counts(counts=feasible & ((values < points[0]) | (values > points[-1])))


def report_off_grid(
    *,
    count: int,
    rule: OffGrid,
    regime: RegimeName,
    target: RegimeName,
    period: int,
    state: StateName,
    phase: str,
    logger: logging.Logger,
) -> None:
    """Enforce a declared raise policy and separately emit checked counts."""
    shown = f">={count}" if count >= COUNT_CAPACITY else f"={count}"
    message = (
        f"Off-grid next states: regime={regime!r}, target={target!r}, "
        f"period={period}, state={state!r}, phase={phase!r}, count{shown}"
    )
    if count and rule == "raise":
        raise OffGridStateError(message)
    if validation_enabled(logger):
        if count:
            logger.warning(message)
        else:
            logger.info(message)
