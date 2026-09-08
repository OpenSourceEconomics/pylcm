"""The vocabulary forward simulation declares its programs in.

Separate from the builder so the canonical regime can publish a declaration
without the engine's low layers reaching the solver contract that builds one.
"""

import dataclasses
from collections.abc import Callable, Hashable
from types import MappingProxyType

from _lcm.execution.core_program import CoreProgram, TiledOutputAxis
from _lcm.typing import StateOrActionName

# Planner name of the per-subject axis every simulation program tiles.
SUBJECT_AXIS = "subject"

# Keyword each simulation program body accepts for the planner-bound tile width.
SUBJECT_WIDTH_KEYWORD = "_lcm_subject_width"

# Graph key of the one program each simulation family publishes per period.
DECISION_PROGRAM = "simulate_decision"
TRANSITION_PROGRAM = "simulate_transition"
ROUTE_PROGRAM = "simulate_route"

# What a simulation program publishes, in the order its body returns it.
ACTION_INDEX = "action_index"
DECISION_VALUE = "decision_value"
NEXT_STATES = "next_states"
REGIME_TRANSITION_PROBS = "regime_transition_probs"

# Subject count a declared tile axis carries until the lowering path rebinds it.
#
# The population is a runtime fact — it arrives with the initial conditions and
# is chunked per call — while the declaration is built once, at model build. The
# lowering path replaces the axis with the chunk width it lowers for, so this
# value never reaches a compiled program.
UNRESOLVED_SUBJECT_EXTENT = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PerSubjectFunction:
    """One simulation body at a single subject's state cell."""

    function: Callable[..., object]
    """The body, taking one subject's states and actions as scalars."""

    subject_arg_names: tuple[str, ...]
    """Arguments carrying a per-subject leading axis, which the tile splits."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationPrograms:
    """The programs one regime dispatches per period group."""

    decision: MappingProxyType[Hashable, CoreProgram]
    """Period to the argmax-and-value program that period dispatches."""

    transition: MappingProxyType[Hashable, CoreProgram]
    """Period to the next-state program that period dispatches, over exactly the
    periods the regime publishes a law of motion for."""

    route: MappingProxyType[Hashable, CoreProgram]
    """Graph key to the regime-transition program; empty where the regime draws
    no successor, which is every terminal regime."""

    def __post_init__(self) -> None:
        """Snapshot the caller-owned program mappings."""
        for field in ("decision", "transition", "route"):
            object.__setattr__(
                self, field, MappingProxyType(dict(getattr(self, field)))
            )

    @property
    def declared_axis_names(self) -> frozenset[str]:
        """Return every planner axis name these programs declare."""
        return frozenset(
            name
            for family in (self.decision, self.transition, self.route)
            for program in family.values()
            for name in program.requirements.axis_names
        )


def subject_axis(*, state_names: tuple[StateOrActionName, ...]) -> TiledOutputAxis:
    """Declare the per-subject axis a simulation program tiles."""
    return TiledOutputAxis(
        name=SUBJECT_AXIS,
        state_names=state_names,
        extent=UNRESOLVED_SUBJECT_EXTENT,
        width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
