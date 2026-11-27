"""The composition action of a regime: the members' decision on the composition."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids.base import Grid
from _lcm.grids.continuous import ContinuousGrid
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import path_segment_name_errors
from lcm.exceptions import GridInitializationError, RegimeInitializationError
from lcm.phased import Phased
from lcm.shocks import MonteCarlo, QuasiMonteCarlo
from lcm.typing import RegimeName, StateName, UserFunction

type MemberName = str


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Destination:
    """A destination regime and the current states at which the member who leaves lands.

    The value of the outside option is the value of the destination's problem at the
    landing states, without the destination's own composition action: an agent decides
    on its composition at most once per period, and the member who leaves acts in that
    problem. Under `Consent` the destination is the formed couple, which applies its
    own rule for staying.
    """

    regime: RegimeName
    """Actual destination regime."""
    projection: Mapping[StateName, UserFunction] | Phased
    """Complete destination-state mapping; a `Phased` one reads no transfer."""
    role: MemberName | None = None
    """Destination member, required for a destination with several members."""
    utility_adjustment: UserFunction | None = None
    """Utility paid once on leaving, also in the value of the outside option."""

    def __post_init__(self) -> None:
        _validate_names(kind="Destination regime", names=(self.regime,))
        object.__setattr__(self, "projection", _freeze_projection(self.projection))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Counterparty:
    """The other agent at a formation: its regime, its member and its landing states."""

    regime: RegimeName
    role: MemberName
    projection: Mapping[StateName, UserFunction] | Phased

    def __post_init__(self) -> None:
        _validate_names(kind="Counterparty regime", names=(self.regime,))
        _validate_names(kind="Counterparty role", names=(self.role,))
        object.__setattr__(self, "projection", _freeze_projection(self.projection))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Consent:
    """Formation by consent: both agents must prefer the formed couple to stay apart."""

    counterparty: Counterparty


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Transfer:
    """The divorce transfer under mutual consent divorce, which the members can shift.

    Destination projections and utility adjustments read the transfer under `name`.
    Each member compares its value of staying at the unconstrained optimum with the
    value of its outside option; the rule's `tie` says which of two equal values it
    prefers, and a member whose destination has no feasible action prefers staying
    wherever the members have a feasible action. If both members want to leave at the
    default, the pair separates there, without a search of the candidates; where a
    destination has no feasible action there, the node has no value. If exactly one
    does, the other member accepts the candidate nearest the default, strictly beyond
    it in that member's favour, at which it prefers the value of its outside option to
    its value of staying at the optimum and every destination has a feasible action.
    The pair separates there if the member who wants to leave prefers the value of its
    outside option there to its value of staying at the optimum, and stays otherwise or
    without such a candidate. Only that one candidate is tested for the member who
    wants to leave.
    """

    name: str
    """Argument name under which projections and utility adjustments read it."""
    candidates: Grid
    """Ordered candidate values; increasing order favours the first member."""
    default: UserFunction
    """The division by law: one number at the source node, before the action.

    It may read states, parameters, the period, functions of the regime and shocks
    the regime itself draws. Its value is clipped to the range of the candidates and
    need not be a candidate.
    """

    def __post_init__(self) -> None:
        _validate_names(kind="Transfer", names=(self.name,))
        if not isinstance(self.candidates, ContinuousGrid):
            raise RegimeInitializationError(
                "Transfer candidates must be an ordered one-dimensional continuous "
                f"grid, got {type(self.candidates).__name__}."
            )
        try:
            self.candidates.to_jax()
        except GridInitializationError as error:
            raise RegimeInitializationError(
                "Transfer candidates need points fixed when the composition action is "
                "declared."
            ) from error


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class PCOnActions:
    """Unilateral divorce with participation constraints on the actions.

    Each member must consent to stay. Each member's value of staying is compared with
    the value of its outside option at every stay action; the members keep their best
    action at which every member consents, and separate if there is none.
    """

    tie: Literal["stay", "leave"] = "stay"
    """What a member exactly indifferent between staying and leaving does.

    A member whose destination has no feasible action is willing to stay at every
    feasible stay action, whatever `tie` says.
    """


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class PCOnWeight:
    """Unilateral divorce with participation constraints on the Pareto weight.

    Each member must consent to stay. Each member's value of staying is compared with
    the value of its outside option at the members' unconstrained optimum. With a
    `state`, the Pareto weight moves first: increasing node order favours the first
    member, decreasing order the second. The weight moves toward the refusing member
    and stops at the first node where both members consent; the members separate when
    the other member refuses there or the path reaches the edge of the declared grid.
    Without a `state`, there is one weight and nothing adjusts: the members separate
    when a member refuses at the optimum.
    """

    state: StateName | None = None
    """Declared discrete grid state that holds the Pareto weight, or `None`.

    The selected node carries forward.
    """
    tie: Literal["stay", "leave"] = "stay"
    """What a member exactly indifferent between staying and leaving does.

    A member whose destination has no feasible action is willing to stay at every
    feasible stay action, whatever `tie` says.
    """

    def __post_init__(self) -> None:
        if self.state is not None:
            _validate_names(kind="Pareto weight state", names=(self.state,))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MutualConsent:
    """Mutual consent divorce: the members separate only if both agree.

    They separate at the default division or at a transfer that one member offers.
    Each member compares at the members' unconstrained optimum.
    """

    tie: Literal["stay", "leave"] = "stay"
    """Which of an equal value of staying and of the outside option a member prefers.

    It holds at every comparison: who wants to leave at the default, whether the other
    member accepts a candidate transfer, and whether the member who wants to leave does
    so there. A member whose destination has no feasible action prefers staying
    wherever the members have a feasible action, whatever `tie` says.
    """
    transfer: Transfer
    """The divorce transfer, which the members can shift before they separate."""


type CompositionRule = PCOnActions | PCOnWeight | MutualConsent | Consent


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class CompositionAction:
    """The members' decision on the composition within the period.

    It is part of the period's choice and is separate from the scheduled next-period
    edges.
    """

    destinations: Mapping[MemberName, Destination]
    """One destination per member of the regime."""
    rule: CompositionRule
    """The rule that decides: one of the divorce rules, or formation by consent."""
    available: UserFunction | None = None
    """State/parameter/observed-shock predicate, evaluated before the comparisons."""
    observes: tuple[str, ...] = ()
    """Destination information observed at this decision, besides the source reads."""
    nodes: MonteCarlo | QuasiMonteCarlo | None = None
    """Draws replacing the product of this decision's rows the stay problem ignores.

    The shock groups the regime's own stay problem reads keep their declared
    nodes. Every other group of the decision, observed destination information
    included, is drawn `n` times from its law at the current state, and the
    same draws serve every kept row. `None` integrates over the full product.

    The rule serves the regime's whole decision on the composition when it is solved.
    """

    def __post_init__(self) -> None:
        if not self.destinations:
            raise RegimeInitializationError(
                "A composition action needs at least one destination."
            )
        _validate_names(
            kind="Composition action member", names=tuple(self.destinations)
        )
        _validate_names(kind="Observed shock", names=self.observes)
        if len(set(self.observes)) != len(self.observes):
            raise RegimeInitializationError(
                "Composition action observes must name each shock once."
            )
        if isinstance(self.rule, Consent) and any(
            destination.utility_adjustment is not None
            for destination in self.destinations.values()
        ):
            raise RegimeInitializationError(
                "A formation by Consent pays nothing: its destination cannot declare a "
                "utility_adjustment."
            )
        object.__setattr__(
            self, "destinations", ensure_containers_are_immutable(self.destinations)
        )
        object.__setattr__(self, "observes", tuple(self.observes))


def _freeze_projection(
    projection: Mapping[StateName, UserFunction] | Phased,
) -> Mapping[StateName, UserFunction] | Phased:
    """Snapshot both phase maps rather than retaining mutable phase leaves."""
    if isinstance(projection, Phased):
        if not isinstance(projection.solve, Mapping) or not isinstance(
            projection.simulate, Mapping
        ):
            raise RegimeInitializationError(
                "Each destination projection phase must be a state-to-callable mapping."
            )
        solve = _freeze_projection(projection.solve)
        return Phased(
            solve=solve,
            simulate=solve
            if projection.solve is projection.simulate
            else _freeze_projection(projection.simulate),
        )
    _validate_names(kind="Projected state", names=tuple(projection))
    if any(not callable(function) for function in projection.values()):
        raise RegimeInitializationError(
            "Destination projection entries must be callable."
        )
    return ensure_containers_are_immutable(projection)


def _validate_names(*, kind: str, names: tuple[str, ...]) -> None:
    errors = path_segment_name_errors(kind=kind, names=names)
    if errors or any(not name.isidentifier() for name in names):
        raise RegimeInitializationError(
            "; ".join(errors) or f"{kind} names must be ordinary identifiers."
        )
