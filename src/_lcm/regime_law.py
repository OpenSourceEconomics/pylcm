"""A source regime's law between regimes, as the model graph binds it.

`Model(edges=...)` declares every regime transition. The model binds one
`RegimeLaw` per source regime from those edges and keeps it on its graph; a
`Regime` carries none. Every internal stage that needs to know where a regime
can go, or how a value-dependent cell routes, reads it from the law keyed by
source regime name.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from beartype.door import is_bearable

from _lcm.gated_edge import GatedEdge
from _lcm.regime_building.schedules import declaration_view, uses_declaration_vocabulary
from _lcm.typing import RegimeName
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.collective import ValueDependentTransition
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import (
    ByAge,
    DeterministicTransition,
    StochasticTransition,
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

    transition: object
    """The bound law, `None` for a terminal regime (no outgoing edges).

    Otherwise one of:

    - a regime name, the deterministic destination;
    - a plain function or `DeterministicTransition` returning a regime code;
    - a `StochasticTransition` returning a probability vector over all regimes;
    - a per-target mapping of `StochasticTransition` probability cells or
      `ValueDependentTransition` declarations;
    - a `ByAge` selecting one of these per source age;
    - a `Phased` pairing the perceived and the realized law.
    """

    gated_edges: MappingProxyType[RegimeName, GatedEdge] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The edges the law's `ValueDependentTransition` cells declare, by target.

    Each routes this regime's continuation into the gate-open target. See
    `GatedEdge`.
    """

    @property
    def terminal(self) -> bool:
        """Whether the regime has no outgoing edges."""
        return self.transition is None

    @property
    def decomposed_transition(self) -> DecomposedTransition:
        """The law with every `ValueDependentTransition` taken apart.

        A value-dependent transition carries two facts at once: which target
        the regime selects, and how the household is routed once there. The
        second belongs to `gated_edges`; what stays here is the selection
        probability, in the per-target cell the canonical pipeline reads. A
        declaration the engine cannot read directly (a `ByAge` schedule, a
        regime name, a `DeterministicTransition`) is read through its
        period-independent `declaration_view` first.
        """
        return decompose_transition(_engine_view(self.transition))

    @property
    def validation_view(self) -> RegimeLaw:
        """The law in the period-independent vocabulary regime validation reads."""
        view = _engine_view(self.transition)
        if view is self.transition:
            return self
        return RegimeLaw(transition=view, gated_edges=self.gated_edges)


type RegimeLaws = Mapping[RegimeName, RegimeLaw]

type RegimeLawDeclaration = (
    RegimeName
    | DeterministicTransition
    | ByAge
    | UserFunction
    | StochasticTransition
    | Phased
    | Mapping[
        RegimeName,
        StochasticTransition | UserFunction | Phased | ValueDependentTransition,
    ]
    | ValueDependentTransition
    | None
)


def _unbound_law() -> int:
    """Stand in for the law of a regime no model has bound yet.

    A callable names no target and reads no variable, so the checks that depend
    on the law's targets or inputs are left to the model, which validates the
    regime again with the law it binds from its edges.
    """
    return 0


# The law a regime is validated against before a model binds its own.
UNBOUND_LAW = RegimeLaw(transition=_unbound_law)


def bind_regime_law(transition: object) -> RegimeLaw:
    """Validate a source's law and derive the gated edges it declares.

    Args:
        transition: The law, `None` for a terminal regime.

    Returns:
        The law together with its gated edges.

    Raises:
        RegimeInitializationError: If the law is not one of the declaration forms,
            declares a `ValueDependentTransition` outside a per-target mapping,
            or declares one gated edge differently in the two phases.
    """
    if not is_bearable(transition, RegimeLawDeclaration):
        raise RegimeInitializationError(
            "A regime transition law is a regime name, a function or "
            "`DeterministicTransition` returning a regime code, a "
            "`StochasticTransition`, a per-target mapping of probabilities, or a "
            f"`ByAge` or `Phased` of these; got {transition!r}."
        )
    view = _engine_view(transition)
    gated_edges = _lower_value_dependent_transitions(view)
    return RegimeLaw(
        transition=transition,
        gated_edges=ensure_containers_are_immutable(gated_edges),
    )


def _engine_view(transition: object) -> object:
    """Read a declaration-vocabulary law through its period-independent view."""
    if uses_declaration_vocabulary(transition):
        return declaration_view(transition)
    return transition


def _lower_value_dependent_transitions(
    transition: object,
) -> dict[RegimeName, GatedEdge]:
    """Derive target-local edges from value-dependent transitions."""
    if isinstance(transition, ValueDependentTransition):
        raise RegimeInitializationError(
            "This regime declares a `ValueDependentTransition` as its whole "
            "transition. A gate is a route — it says where a household goes "
            "when consent fails — so it belongs to one target and is written "
            "in a per-target `Transition` law, keyed by the regime the gate "
            "opens onto: `Transition(targets=..., law={'<target>': "
            "ValueDependentTransition(...)})`."
        )
    if isinstance(transition, Phased):
        return _lower_phased_value_dependent_transitions(transition)
    if not isinstance(transition, Mapping):
        return {}
    return _declared_gated_edges(transition=transition)


def _lower_phased_value_dependent_transitions(
    transition: Phased,
) -> dict[RegimeName, GatedEdge]:
    """Derive the one edge a per-phase pair of declarations describes.

    The probability may differ between the phases — a perceived meeting rate
    and a realized one are a legitimate wedge. Everything else is the same edge
    written twice, so the two sides must agree: identical gate callables, and
    routes, references and off-grid contract that compare equal.
    """
    sides = {}
    for phase in ("solve", "simulate"):
        side = getattr(transition, phase)
        if not isinstance(side, Mapping):
            return {}
        sides[phase] = _declared_gated_edges(transition=side)
    if not any(sides.values()):
        return {}

    one_sided = set(sides["solve"]) ^ set(sides["simulate"])
    if one_sided:
        raise RegimeInitializationError(
            f"The transition into {min(one_sided)!r} is declared as a "
            "`ValueDependentTransition` in one phase and as an ordinary "
            "probability in the other. A target is value-dependent in both "
            "phases or in neither: the gate is what the household consents "
            "to, and it cannot consent only while being solved."
        )

    solve_edges = sides["solve"]
    simulate_edges = sides["simulate"]
    for target, solve_edge in solve_edges.items():
        simulate_edge = simulate_edges[target]
        if solve_edge.gate is not simulate_edge.gate:
            raise RegimeInitializationError(
                f"The two phases of the transition into {target!r} declare "
                "different gate callables. A gate is one predicate the "
                "household is held to, so the two phases must name the "
                "very same function. A difference the model needs goes in "
                "a route's `fallback`, which is `Phased` in its own right."
            )
        if solve_edge != simulate_edge:
            raise RegimeInitializationError(
                f"The two phases of the transition into {target!r} declare "
                "the same gate but disagree elsewhere — on the routes, the "
                "gate references or the off-grid contract. An edge is all "
                "of those together, so the two phases must describe one "
                "edge. Only `probability` may differ between them."
            )
    return solve_edges


def _declared_gated_edges(
    *, transition: Mapping[RegimeName, object]
) -> dict[RegimeName, GatedEdge]:
    """Return the edge each value-dependent cell of one phase declares."""
    return {
        target: GatedEdge(
            gate=cell.gate,
            legs=cell.routes,
            gate_refs=cell.gate_references,
            off_grid=cell.off_grid,
        )
        for target, cell in transition.items()
        if isinstance(cell, ValueDependentTransition)
    }


def decompose_transition(transition: object) -> DecomposedTransition:
    """Replace every `ValueDependentTransition` by the probability it declares.

    Args:
        transition: A source's law, including the `Phased` form.

    Returns:
        The same law with each value-dependent cell replaced by its selection
        probability. The routing half of the declaration belongs to
        `gated_edges` and does not appear here.
    """
    if isinstance(transition, Phased):
        solve = _decomposed_transition_side(transition.solve)
        simulate = _decomposed_transition_side(transition.simulate)
        if solve is transition.solve and simulate is transition.simulate:
            return transition
        return Phased(solve=solve, simulate=simulate)
    return _decomposed_transition_side(transition)


def _decomposed_transition_side(transition: object) -> DecomposedTransition:
    """Replace one phase's `ValueDependentTransition` cells by their probabilities.

    Args:
        transition: One phase's law — a per-target mapping, a coarse callable or
            `StochasticTransition`, an age schedule, or `None` for a terminal
            regime.

    Returns:
        The same law with every `ValueDependentTransition` cell replaced by the
        selection probability it declares, wrapped in a `StochasticTransition`
        where the declaration gave a bare callable. An age schedule has each of
        its laws taken apart. Anything else that is not a per-target mapping is
        returned unchanged.
    """
    if isinstance(transition, ByAge):
        return transition.with_mapped_laws(func=decompose_transition)
    if not isinstance(transition, Mapping):
        return cast(
            "UserFunction | StochasticTransition | Phased | ByAge | None",
            transition,
        )
    if not any(
        isinstance(cell, ValueDependentTransition) for cell in transition.values()
    ):
        # Nothing to take apart. Returning the very same mapping keeps the
        # phase-variation scan able to ask whether the author wrote one object
        # for both phases, which a freshly built copy would always deny.
        return cast(
            "Mapping[RegimeName, StochasticTransition | UserFunction | Phased]",
            transition,
        )
    return MappingProxyType(
        {
            target: (
                _as_markov_transition(cell.probability)
                if isinstance(cell, ValueDependentTransition)
                else cast("StochasticTransition | UserFunction | Phased", cell)
            )
            for target, cell in transition.items()
        }
    )


def _as_markov_transition(
    probability: UserFunction | StochasticTransition,
) -> StochasticTransition:
    """Wrap a bare probability callable in the cell grammar's `StochasticTransition`."""
    if isinstance(probability, StochasticTransition):
        return probability
    return StochasticTransition(func=probability)
