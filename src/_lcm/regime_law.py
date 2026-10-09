"""A source regime's law between regimes, as the model graph binds it.

`Model(edges=...)` declares every regime transition. The model binds one
`RegimeLaw` per source regime from those edges and keeps it on its graph; a
`Regime` carries none. Every internal stage that needs to know where a regime
can go, or how a gated target routes, reads it from the law keyed by source
regime name.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from beartype.door import is_bearable

from _lcm.gated_edge import GatedEdge
from _lcm.regime_building.schedules import (
    RegimeTransitionLaw,
    declaration_view,
    uses_declaration_vocabulary,
)
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
    _SupportedStochasticTransition,
)
from _lcm.typing import RegimeName
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import (
    ByAge,
    DeterministicTransition,
    StochasticTransition,
    fail_if_phased_wraps_a_schedule,
)
from lcm.typing import UserFunction

type DecomposedTransition = (
    UserFunction
    | StochasticTransition
    | Phased
    | ByAge
    | Mapping[RegimeName, StochasticTransition | UserFunction | Phased]
    | None
)


@dataclass(frozen=True, kw_only=True)
class RegimeLaw:
    """The law that moves a source regime's subjects to their next regime."""

    transition: RegimeTransitionLaw
    """The bound law, `None` for a terminal regime (no outgoing edges).

    Otherwise one of:

    - a regime name, the deterministic destination;
    - a plain function or `DeterministicTransition` returning a regime code;
    - a `StochasticTransition` returning a probability vector over all regimes;
    - a per-target mapping of `StochasticTransition` probability cells;
    - a `ByAge` selecting one of these per source age;
    - a `Phased` pairing the perceived and the realized law.
    """

    gated_edges: MappingProxyType[RegimeName, GatedEdge] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The edges the source's `Transition.gates` declare, by gated target.

    Each routes this regime's continuation into the gate-open target. See
    `GatedEdge`.
    """

    @property
    def terminal(self) -> bool:
        """Whether the regime has no outgoing edges."""
        return self.transition is None

    @property
    def decomposed_transition(self) -> DecomposedTransition:
        """The law in the vocabulary the engine reads.

        A declaration the engine cannot read directly (a `ByAge` schedule, a
        regime name, a `DeterministicTransition`) is read through its
        period-independent `declaration_view`; every other law is returned as
        bound. The gates are not part of it: they are `gated_edges`.
        """
        return _engine_view(self.transition)


type RegimeLaws = Mapping[RegimeName, RegimeLaw]

type RegimeLawDeclaration = (
    RegimeName
    | DeterministicTransition
    | ByAge
    | UserFunction
    | StochasticTransition
    | Phased
    | Mapping[RegimeName, StochasticTransition | UserFunction | Phased]
    | None
)


# keyword-only-exempt: primary-argument=transition
def bind_regime_law(
    # Any value: the declaration check below is what refuses a non-law.
    transition: object,
    *,
    gated_edges: Mapping[RegimeName, GatedEdge] = MappingProxyType({}),
) -> RegimeLaw:
    """Validate a source's law and attach the gated edges that apply to it.

    Args:
        transition: The law, `None` for a terminal regime.
        gated_edges: The source's gated edges, by gated target. A target the
            law can no longer reach — its cells removed in every case — keeps
            no gated edge.

    Returns:
        The law together with its gated edges.

    Raises:
        RegimeInitializationError: If the law is not one of the declaration forms
            or nests a schedule inside `Phased`.
    """
    if not is_bearable(transition, RegimeLawDeclaration):
        raise RegimeInitializationError(
            "A regime transition law is a regime name, a function or "
            "`DeterministicTransition` returning a regime code, a "
            "`StochasticTransition`, a per-target mapping of probabilities, or a "
            f"`ByAge` or `Phased` of these; got {transition!r}."
        )
    law = cast("RegimeTransitionLaw", transition)
    fail_if_phased_wraps_a_schedule(law)
    named = _named_targets(law)
    return RegimeLaw(
        transition=law,
        gated_edges=ensure_containers_are_immutable(
            {
                target: edge
                for target, edge in gated_edges.items()
                if named is None or target in named
            }
        ),
    )


def _named_targets(transition: RegimeTransitionLaw) -> frozenset[RegimeName] | None:
    """The targets a law can reach in any case or phase.

    `None` for a law over all targets that is not yet bound to the graph's
    support.
    """
    if isinstance(transition, ByAge | Phased):
        cases = (
            transition.laws
            if isinstance(transition, ByAge)
            else (transition.solve, transition.simulate)
        )
        named = tuple(
            _named_targets(cast("RegimeTransitionLaw", case)) for case in cases
        )
        if any(case_named is None for case_named in named):
            return None
        return frozenset().union(*cast("tuple[frozenset[RegimeName], ...]", named))
    if transition is None:
        return frozenset()
    if isinstance(transition, str):
        return frozenset({transition})
    if isinstance(
        transition, _SupportedDeterministicTransition | _SupportedStochasticTransition
    ):
        return frozenset(transition.targets)
    return frozenset(transition) if isinstance(transition, Mapping) else None


def _engine_view(transition: RegimeTransitionLaw) -> DecomposedTransition:
    """Read a declaration-vocabulary law through its period-independent view."""
    if uses_declaration_vocabulary(transition):
        return cast("DecomposedTransition", declaration_view(transition))
    return cast("DecomposedTransition", transition)
