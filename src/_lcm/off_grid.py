"""Raw continuous-state grid diagnostics, independent of clipping and verbosity."""

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from dags import with_signature

from _lcm.grids import ContinuousGrid, Grid
from _lcm.grids.continuous import _clip_grid_value
from _lcm.solution.contract import Solver
from _lcm.solution.dcegm import DCEGM
from _lcm.solution.egm import EGM
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import NEGM
from _lcm.solution.nnbegm import NNBEGM
from _lcm.typing import PytreeValue
from _lcm.utils.counts import COUNT_CAPACITY, sum_counts
from _lcm.utils.logging import validation_enabled
from lcm.exceptions import (
    OffGridPolicyError,
    OffGridStateError,
    RegimeInitializationError,
)
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    OffGrid,
    RegimeName,
    ScalarInt,
    StageName,
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
    stage: StageName | None = None,
) -> None:
    """Enforce a declared raise policy and separately emit checked counts."""
    message = off_grid_message(
        count=count,
        regime=regime,
        target=target,
        period=period,
        state=state,
        phase=phase,
        stage=stage,
    )
    if count and rule == "raise":
        raise OffGridStateError(message)
    if validation_enabled(logger):
        if count:
            logger.warning(message)
        else:
            logger.info(message)


def off_grid_message(
    *,
    count: int,
    regime: RegimeName,
    target: RegimeName,
    period: int,
    state: StateName,
    phase: str,
    stage: StageName | None = None,
) -> str:
    """Describe one target state's raw landings outside, or not finite on, its grid.

    Every check that counts landings words its report and its refusal with this,
    so the same landing reads the same in the solve kernel, the preflight and the
    realized simulation. `stage` names the source's stage in a model with
    stages. A count at `COUNT_CAPACITY` is a lower bound.
    """
    shown = f">={count}" if count >= COUNT_CAPACITY else f"={count}"
    in_stage = "" if stage is None else f", stage={stage!r}"
    return (
        f"Off-grid next states: regime={regime!r}, target={target!r}, "
        f"period={period}{in_stage}, state={state!r}, phase={phase!r}, "
        f"count{shown}"
    )


@dataclass(frozen=True, kw_only=True)
class ProjectedGridRule:
    """The destination-grid rule an exit's projected continuous state follows.

    A mover lands on the destination's grid as a realised next state does, so the
    state's declared `off_grid` rule acts on the projected value:

    - `"clip_state"` caps it to the grid's outer nodes, wherever it is read;
    - `"raise"` leaves it as is, and every active read outside the grid counts as
      a reached landing, which fails the solve or the simulation.

    `"extrapolate"` and `"clip_lookup"` keep the projected value, so they need no
    rule.
    """

    source: RegimeName
    """The regime whose exit projects the state."""
    target: RegimeName
    """The destination regime."""
    state: StateName
    """The destination state."""
    off_grid: Literal["clip_state", "raise"]
    """The destination grid's declared rule."""
    low: float
    """The grid's lowest node in the read's period."""
    high: float
    """The grid's highest node in the read's period."""


def projected_grid_rules(
    *,
    source: RegimeName,
    target: RegimeName,
    states: tuple[StateName, ...],
    grids: Mapping[StateName, Grid],
    nodes: Mapping[StateName, Float1D | IntND],
) -> tuple[ProjectedGridRule, ...]:
    """Return the rule of each projected state whose grid clips or raises.

    Args:
        source: The regime whose exit projects the states.
        target: The destination regime.
        states: The destination states the exit projects.
        grids: The destination's grids.
        nodes: The destination's state nodes in the read's period.

    Returns:
        One rule per projected continuous state whose grid declares
        `off_grid="clip_state"` or `"raise"`.

    Raises:
        RegimeInitializationError: Such a grid takes its points at run time, so
            its outer nodes are not known when the exit is bound.

    """
    rules = []
    for state in states:
        grid = grids.get(state)
        if not isinstance(grid, ContinuousGrid) or grid.off_grid not in (
            "clip_state",
            "raise",
        ):
            continue
        points = np.asarray(nodes[state])
        if not np.all(np.isfinite(points)):
            raise RegimeInitializationError(
                f"The exit of regime {source!r} projects state {state!r} of regime "
                f"{target!r}, whose grid declares off_grid={grid.off_grid!r} on "
                "points supplied at run time. The exit cannot apply that rule "
                "without the grid's outer nodes; fix the points or declare another "
                "off_grid rule."
            )
        rules.append(
            ProjectedGridRule(
                source=source,
                target=target,
                state=state,
                off_grid=grid.off_grid,
                low=float(points[0]),
                high=float(points[-1]),
            )
        )
    return tuple(rules)


def clip_projection(
    *,
    projection: Mapping[StateName, Callable[..., PytreeValue]],
    rules: tuple[ProjectedGridRule, ...],
) -> dict[StateName, Callable[..., PytreeValue]]:
    """Cap each projected state whose destination grid declares `clip_state`.

    Every reader of the projection then takes the capped value: the destination's
    own problem, its shock laws, the reads below it and the simulated landing.
    """
    clipped = dict(projection)
    for rule in rules:
        if rule.off_grid == "clip_state":
            function = projection[rule.state]
            clipped[rule.state] = with_signature(
                _ClippedProjection(function=function, rule=rule),
                args=list(inspect.signature(function).parameters),
                enforce=False,
            )
    return clipped


def count_projected_off_grid(
    *,
    rules: tuple[ProjectedGridRule, ...],
    mapped: Mapping[StateName, FloatND | IntND],
    active: BoolND,
) -> IntND:
    """Count, per `raise` rule, whether an active read lands outside its grid.

    Args:
        rules: The projected states' destination-grid rules.
        mapped: The read's projected destination states.
        active: Whether the read is evaluated at a genuine, available cell.

    Returns:
        One int32 count per `raise` rule, in rule order.

    """
    counts = [
        (
            active
            & ((mapped[rule.state] < rule.low) | (mapped[rule.state] > rule.high))
        ).astype(jnp.int32)
        for rule in rules
        if rule.off_grid == "raise"
    ]
    return jnp.stack(counts) if counts else jnp.zeros((0,), dtype=jnp.int32)


@dataclass(frozen=True, kw_only=True, eq=False)
class _ClippedProjection:
    """A projected state capped to its destination grid's outer nodes."""

    function: Callable[..., PytreeValue]
    rule: ProjectedGridRule

    def __call__(self, **operands: PytreeValue) -> FloatND:
        value = jnp.asarray(cast("FloatND", self.function(**operands)))
        return _clip_grid_value(
            value=value, points=jnp.asarray((self.rule.low, self.rule.high))
        )
