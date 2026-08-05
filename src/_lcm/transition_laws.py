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

A weight vector alone does not say what the weights *mean*, so the description
separates the two things a target-qualified weight function can be:

- `stochastic` — the weights are probabilities, and the law realizes a draw.
- `interpolation_basis` — the weights are the coefficients that express one
  declared value in the target's node basis. The law names a single value.

Exactly one of the two holds whenever `weight_name` is set, and neither when it
is `None`. Collapsing them into "has weights" would price a declared entry value
as a lottery over the nodes it interpolates between, which every certainty
equivalent but the linear one answers differently.

Weights and *availability* are likewise separate questions. A law's public
`next_<state>` name always produces the physical next-period value, in every
phase, so any other law may consume it — `next_wealth(..., next_aime)` and
`good_probs(next_capital)` are the ordinary cases, and a declared process entry
is no different. Where the target holds its value function on discrete nodes,
the axis the interpolator maps over is a *separate*, privately named object
(`support_axis_name`), never the public name. Only a genuinely stochastic law
has no physical value to publish: a draw is not available while the expectation
over it is still being built, so a dependency on one is rejected at model build
rather than silently resolved.
"""

from dataclasses import dataclass
from types import MappingProxyType

from _lcm.typing import RegimeName, TransitionFunctionName
from lcm.typing import Int1D


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
    """Whether the law's weights are probabilities, so that it realizes a draw."""

    interpolation_basis: bool
    """Whether the law's weights express one declared value in the node basis.

    True for a declared entry law into a target's continuous process: the value
    the source names is off the target's nodes in general, so it reaches the
    engine as the hat weights of linear interpolation over them. Those
    coefficients are not probabilities, and the continuation they describe is the
    single value `Σ_j w_j · V(node_j)` — contracted before any certainty
    equivalent sees it.
    """

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

    emits_support_index: bool
    """Whether the law's *own* output indexes the target's support for this state.

    A continuous stochastic process is stored on a discrete value-function axis,
    so a law that realizes a draw from one emits node indices — there is no
    single physical value it could emit instead. A declared entry law is the
    other case: it names a physical value and keeps emitting it, and the node
    axis it is interpolated over lives separately under `support_axis_name`.
    """

    support_axis_name: str | None
    """Private name of the node axis the interpolator maps over, if any.

    Set exactly when `interpolation_basis` holds. The axis is the target's node
    vector for this state — an engine-internal object, deliberately not the
    public `next_<state>` name, so that publishing it can never shadow the
    physical value a dependent law reads. `None` everywhere else: a stochastic
    law's own output already *is* the axis, and a plain deterministic law has no
    axis at all.
    """

    weight_name: str | None
    """`weight_<target>__next_<state>`, or `None` for a law that carries no weights.

    Set exactly when one of `stochastic` / `interpolation_basis` holds.
    """


# Immutable mapping of target regime names to their transition laws.
type TransitionLaws = MappingProxyType[
    RegimeName, MappingProxyType[TransitionFunctionName, TransitionLawInfo]
]

# Immutable mapping of target regime names to their private node axes, keyed by
# the public `next_<state>` name whose law is interpolated over each.
type SupportAxes = MappingProxyType[
    RegimeName, MappingProxyType[TransitionFunctionName, Int1D]
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


def is_interpolation_basis(
    laws: TransitionLaws, target: RegimeName, next_state_name: TransitionFunctionName
) -> bool:
    """Return whether one target's named law weights a node basis, not a lottery.

    Args:
        laws: Immutable mapping of target regime names to their transition laws.
        target: Regime the law leads into.
        next_state_name: Unqualified `next_<state>` name.

    Returns:
        Whether the law's weights are interpolation coefficients. Unknown pairs
        are not, which keeps callers that iterate a bundle wider than the
        description safe.

    """
    info = laws.get(target, MappingProxyType({})).get(next_state_name)
    return info is not None and info.interpolation_basis


def support_axis_names(
    laws: TransitionLaws, target: RegimeName
) -> MappingProxyType[TransitionFunctionName, str]:
    """Return one target's private support axes, keyed by public next-state name.

    Args:
        laws: Immutable mapping of target regime names to their transition laws.
        target: Regime whose laws to read.

    Returns:
        Immutable mapping from each `next_<state>` whose law is interpolated
        over a node basis to the private name of that basis's axis.

    """
    return MappingProxyType(
        {
            name: info.support_axis_name
            for name, info in laws.get(target, MappingProxyType({})).items()
            if info.support_axis_name is not None
        }
    )
