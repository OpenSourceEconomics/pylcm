"""Per-target description of how each next-period state obtains its value.

One `TransitionLawInfo` per `(target regime, next-state)` pair, built once after
explicit and intrinsic transition synthesis and read identically by solve,
diagnostics, and simulation.

The description is **target-qualified** because stochasticity belongs to the
`(target, next-state)` pair, not to the name alone. Consulting the source's own
variables cannot decide it: a process the source does not carry is invisible
there, so its entry law reads as deterministic and its weights are built and then
discarded. Keying on the target instead asks the only question that has an
answer -- how does *this* target obtain *this* state.

Stochasticity is read off the synthesized functions rather than re-derived from
the user's declarations: a law is stochastic when a target-qualified weight
function was built for it. That keeps the description in step with what the DAG
actually contains, so a law can never be priced by a weight that was never built,
nor built a weight that nothing consumes.
"""

from dataclasses import dataclass
from types import MappingProxyType

from _lcm.typing import RegimeName, TransitionFunctionName


@dataclass(frozen=True)
class TransitionLawInfo:
    """How one next-period state of one target regime obtains its value."""

    target: RegimeName
    """Regime this law leads into."""

    next_state_name: TransitionFunctionName
    """Unqualified `next_<state>` name, as keyed within the target's bundle."""

    qualified_name: str
    """`<target>__next_<state>`, as keyed in the flat function namespace."""

    stochastic: bool
    """Whether the law realizes a draw rather than a single value."""

    continuous_process: bool
    """Whether the target's grid for this state is a continuous process.

    Read from the *target's* grid, so a process the source does not carry is
    still recognized as continuous.
    """

    intrinsic_entry: bool
    """Whether the law is the target process's own entry law.

    True only when the source neither carries the state nor declares a law for
    it, so the value comes from the process's unconditional distribution.
    """

    weight_name: str | None
    """`weight_<target>__next_<state>`, or `None` for a deterministic law."""


# Immutable mapping of target regime names to their transition laws.
type TransitionLaws = MappingProxyType[
    RegimeName, MappingProxyType[TransitionFunctionName, TransitionLawInfo]
]


def stochastic_next_state_names(
    laws: TransitionLaws, target: RegimeName
) -> tuple[TransitionFunctionName, ...]:
    """Return one target's stochastic `next_<state>` names, in bundle order.

    Args:
        laws: Immutable mapping of target regime names to their transition laws.
        target: Regime whose laws to read.

    Returns:
        Tuple of unqualified transition names whose law is stochastic.

    """
    return tuple(
        name
        for name, info in laws.get(target, MappingProxyType({})).items()
        if info.stochastic
    )


def is_stochastic(
    laws: TransitionLaws, target: RegimeName, next_state_name: TransitionFunctionName
) -> bool:
    """Return whether one target's named law realizes a draw.

    Args:
        laws: Immutable mapping of target regime names to their transition laws.
        target: Regime the law leads into.
        next_state_name: Unqualified `next_<state>` name.

    Returns:
        Whether the law is stochastic. Unknown pairs are deterministic, which
        keeps callers that iterate a bundle wider than the description safe.

    """
    info = laws.get(target, MappingProxyType({})).get(next_state_name)
    return info is not None and info.stochastic
