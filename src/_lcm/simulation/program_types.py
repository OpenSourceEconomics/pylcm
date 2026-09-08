"""The vocabulary forward simulation declares its programs in.

Separate from the builder so the canonical regime can publish a declaration
without the engine's low layers reaching the solver contract that builds one.

`output_roles` follows the same convention every solve program follows: the
declared tree is a pytree of the same structure as the body's own output, one
role leaf per output leaf, so the lowering path can check a lowered signature
against it. A body returning a nested mapping declares the same nesting, and the
role names what that position publishes.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from _lcm.execution.core_program import CoreBuildContext, CoreProgram, TiledOutputAxis
from _lcm.typing import RegimeName, StateOrActionName

# Planner name of the per-subject axis every simulation program tiles.
SUBJECT_AXIS = "subject"

# Keyword each simulation program body accepts for the planner-bound tile width.
SUBJECT_WIDTH_KEYWORD = "__lcm_subject_width__"

# Graph key of the one program each simulation family publishes per period.
DECISION_PROGRAM = "simulate_decision"
TRANSITION_PROGRAM = "simulate_transition"
ROUTE_PROGRAM = "simulate_route"

# What one leaf of a simulation program's output publishes.
ACTION_INDEX = "action_index"
DECISION_VALUE = "decision_value"
NEXT_STATE = "next_state"
REGIME_TRANSITION_PROB = "regime_transition_prob"

# Subject count a declared tile axis carries until the lowering path rebinds it.
#
# The population is a runtime fact — it arrives with the initial conditions and
# is chunked per call — while the declaration is built once, at model build. The
# lowering path replaces the axis with the chunk width it lowers for, so this
# value never reaches a compiled program.
UNRESOLVED_SUBJECT_EXTENT = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationBuildContext(CoreBuildContext):
    """Complete dynamic arguments for one forward program invocation."""

    call_arguments: Mapping[str, object]
    """Subject states, action operands, parameters, keys and addressed value reads."""

    def __post_init__(self) -> None:
        """Snapshot the call arguments together with the common core context."""
        super().__post_init__()
        object.__setattr__(
            self, "call_arguments", MappingProxyType(dict(self.call_arguments))
        )


@runtime_checkable
class SimulationProgramExecutor(Protocol):
    """Dispatch a declared program against one population's live arguments."""

    def dispatch(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
    ) -> object:
        """Select and invoke the executable for this argument signature."""
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PerSubjectFunction:
    """One simulation body at a single subject's state cell."""

    function: Callable[..., object]
    """The body, taking one subject's states and actions as scalars."""

    subject_arg_names: tuple[str, ...]
    """Arguments carrying a per-subject leading axis, which the tile splits."""

    output_roles: object
    """Role tree of the same structure as the body's output, one role per leaf."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationPrograms:
    """The programs one regime dispatches per period group."""

    decision: MappingProxyType[int, CoreProgram]
    """Period to the selected decision: grid maximization or finite-bank ranking."""

    transition: MappingProxyType[int, CoreProgram]
    """Period to the next-state program that period dispatches, over exactly the
    periods the regime publishes a law of motion for."""

    route: MappingProxyType[int, CoreProgram]
    """Period to the regime-transition program that period dispatches; empty
    where the regime draws no successor, which is every terminal regime."""

    policy_prepare: MappingProxyType[int, CoreProgram] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    """Finite candidate reconstruction, before the host diagnostic."""

    policy_rank: MappingProxyType[int, CoreProgram] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    """Canonical ranking of the represented finite candidate bank."""

    executor: SimulationProgramExecutor | None = None
    """Call-local lowering and dispatch owner, absent from model declarations."""

    def __post_init__(self) -> None:
        """Snapshot the caller-owned program mappings."""
        for field in (
            "decision",
            "transition",
            "route",
            "policy_prepare",
            "policy_rank",
        ):
            object.__setattr__(
                self, field, MappingProxyType(dict(getattr(self, field)))
            )

    @property
    def declared_axis_names(self) -> frozenset[str]:
        """Return every planner axis name these programs declare."""
        return frozenset(
            name
            for family in (
                self.decision,
                self.transition,
                self.route,
                self.policy_prepare,
                self.policy_rank,
            )
            for program in family.values()
            for name in program.requirements.axis_names
        )


def transition_output_roles(
    *, target_next_state_names: Mapping[RegimeName, Sequence[str]]
) -> dict[RegimeName, dict[str, str]]:
    """Declare the role of every leaf a law-of-motion body publishes.

    The body returns one inner mapping per target regime, keyed by the next-state
    name that target carries, so the role tree carries the same two levels.
    """
    return {
        target: dict.fromkeys(next_state_names, NEXT_STATE)
        for target, next_state_names in target_next_state_names.items()
    }


def route_output_roles(
    *, target_regime_names: Sequence[RegimeName]
) -> MappingProxyType[RegimeName, str]:
    """Declare the role of every leaf a regime-transition body publishes.

    The body returns one probability per regime it can draw, as an immutable
    mapping whose key order is part of its pytree identity, so the role tree
    repeats that order.
    """
    return MappingProxyType(dict.fromkeys(target_regime_names, REGIME_TRANSITION_PROB))


def subject_axis(*, state_names: tuple[StateOrActionName, ...]) -> TiledOutputAxis:
    """Declare the per-subject axis a simulation program tiles."""
    return TiledOutputAxis(
        name=SUBJECT_AXIS,
        state_names=state_names,
        extent=UNRESOLVED_SUBJECT_EXTENT,
        width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
