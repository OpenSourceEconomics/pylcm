"""Current-period exits and the mapped policy problems they read."""

from collections.abc import Mapping
from dataclasses import dataclass, field
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

type ExitRole = str


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ExitTarget:
    """A destination regime and the current states a mover lands at.

    The payoff of leaving is the value of the destination's problem at the projected
    states, without the destination's own exit: a subject moves at most once per
    period, and a mover acts in that problem. Under `Consent` the target is the
    formed couple, which applies its own rule for staying.
    """

    regime: RegimeName
    """Actual destination regime."""
    projection: Mapping[StateName, UserFunction] | Phased
    """Complete destination-state mapping; a `Phased` one reads no settlement."""
    role: ExitRole | None = None
    """Destination member, required for a collective destination."""
    utility_adjustment: UserFunction | None = None
    """Utility paid once on exit and in its participation benchmark."""

    def __post_init__(self) -> None:
        _validate_names(kind="Exit target regime", names=(self.regime,))
        object.__setattr__(self, "projection", _freeze_projection(self.projection))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Counterparty:
    """The other party's Raw stay problem used for strict formation consent."""

    regime: RegimeName
    role: ExitRole
    projection: Mapping[StateName, UserFunction] | Phased

    def __post_init__(self) -> None:
        _validate_names(kind="Counterparty regime", names=(self.regime,))
        _validate_names(kind="Counterparty role", names=(self.role,))
        object.__setattr__(self, "projection", _freeze_projection(self.projection))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Consent:
    """Require both parties to prefer the selected target Participating policy."""

    counterparty: Counterparty


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Renegotiate:
    """Move a discrete state toward the refusing member before separation.

    Increasing node order favors the first member; decreasing order favors
    the second. Stop at the first mutually willing node, or separate when the
    other member refuses or the path reaches the edge of the declared grid.
    """

    state: StateName
    """Declared discrete grid state whose selected node carries forward."""

    def __post_init__(self) -> None:
        _validate_names(kind="Renegotiated state", names=(self.state,))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Settlement:
    """A division on separation that the members can shift before a mutual-consent exit.

    Exit-target projections and utility adjustments read the settlement under
    `name`. Each member compares its stay value at the household optimum with its
    outside value; the rule's `tie` says which of two equal values it prefers. If
    both members want to leave at the default, the pair separates there. If exactly
    one does, the other member accepts the candidate nearest the default, strictly
    beyond it in that member's favour, at which it prefers its outside value and
    every destination has a feasible action. The pair separates there if the member
    who wants to leave prefers its outside value there, and stays otherwise or
    without such a candidate.
    """

    name: str
    """Argument name under which projections and utility adjustments read it."""
    candidates: Grid
    """Ordered candidate values; increasing order favours the first member."""
    default: UserFunction
    """The division by law: reads what the exit's `available` may read.

    Its value is clipped to the range of the candidates.
    """

    def __post_init__(self) -> None:
        _validate_names(kind="Settlement", names=(self.name,))
        if not isinstance(self.candidates, ContinuousGrid):
            raise RegimeInitializationError(
                "Settlement candidates must be an ordered one-dimensional continuous "
                f"grid, got {type(self.candidates).__name__}."
            )
        try:
            self.candidates.to_jax()
        except GridInitializationError as error:
            raise RegimeInitializationError(
                "Settlement candidates need points fixed when the exit is declared."
            ) from error


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Unilateral:
    """Any member can end the match: it continues only if every member is willing."""

    tie: Literal["stay", "leave"] = "stay"
    """What a member exactly indifferent between staying and the exit does."""
    compares: Literal["each_action", "optimum"] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Where each member's stay value is compared with its outside value.

    - `"each_action"`: at every stay action; the household keeps its best action
      that passes the rule.
    - `"optimum"`: at the household's unconstrained optimum only.
    - `None`: `"optimum"` with `renegotiate`, `"each_action"` otherwise.
    """
    renegotiate: Renegotiate | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Discrete adjustment attempted before the current-period exit."""

    def __post_init__(self) -> None:
        _fail_if_adjusted_each_action(
            compares=self.compares, adjusted=self.renegotiate is not None
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MutualConsent:
    """The match ends only if both members agree, at the default or a settlement."""

    tie: Literal["stay", "leave"] = "stay"
    """Which of an equal stay and outside value a member prefers, at every comparison.

    It decides who wants to leave at the default, whether the other member accepts a
    candidate settlement, and whether the member who wants to leave does so there.
    """
    compares: Literal["each_action", "optimum"] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Where each member's stay value is compared with its outside value.

    Always the household's unconstrained optimum: `None` and `"optimum"` both mean
    it, and `"each_action"` is refused.
    """
    settle: Settlement
    """Division on separation that the members can shift before the exit."""

    def __post_init__(self) -> None:
        _fail_if_adjusted_each_action(compares=self.compares, adjusted=True)


type ExitRule = Unilateral | MutualConsent | Consent


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Exit:
    """The regime's current-period move, separate from scheduled next-period edges."""

    targets: Mapping[ExitRole, ExitTarget]
    """One destination per source member, or self for a singleton."""
    rule: ExitRule
    """Built-in dissolution rule or strict formation consent."""
    available: UserFunction | None = None
    """State/parameter/observed-shock predicate, evaluated before benchmarks."""
    observes: tuple[str, ...] = ()
    """Target information observed at this choice, in addition to source reads."""
    nodes: MonteCarlo | QuasiMonteCarlo | None = None
    """Draws replacing the product of this choice's rows the stay problem ignores.

    The shock groups the regime's own stay problem reads keep their declared
    nodes. Every other group of the choice, observed target information
    included, is drawn `n` times from its law at the current state, and the
    same draws serve every kept row. `None` integrates over the full product.

    The rule serves the regime's whole exit choice when it is solved.
    """

    def __post_init__(self) -> None:
        if not self.targets:
            raise RegimeInitializationError("An exit needs destination targets.")
        _validate_names(kind="Exit target", names=tuple(self.targets))
        _validate_names(kind="Observed shock", names=self.observes)
        if len(set(self.observes)) != len(self.observes):
            raise RegimeInitializationError("Exit observes must name each shock once.")
        if isinstance(self.rule, Consent) and any(
            target.utility_adjustment is not None for target in self.targets.values()
        ):
            raise RegimeInitializationError(
                "A formation by Consent pays nothing: its target cannot declare a "
                "utility_adjustment."
            )
        object.__setattr__(
            self, "targets", ensure_containers_are_immutable(self.targets)
        )
        object.__setattr__(self, "observes", tuple(self.observes))


def _freeze_projection(
    projection: Mapping[StateName, UserFunction] | Phased,
) -> Mapping[StateName, UserFunction] | Phased:
    """Snapshot both settlement maps rather than retaining mutable phase leaves."""
    if isinstance(projection, Phased):
        if not isinstance(projection.solve, Mapping) or not isinstance(
            projection.simulate, Mapping
        ):
            raise RegimeInitializationError(
                "Each exit projection phase must be a state-to-callable mapping."
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
        raise RegimeInitializationError("Exit projection entries must be callable.")
    return ensure_containers_are_immutable(projection)


def _fail_if_adjusted_each_action(
    *, compares: Literal["each_action", "optimum"] | None, adjusted: bool
) -> None:
    if adjusted and compares == "each_action":
        raise RegimeInitializationError(
            "An adjustment before the exit compares at the optimum; "
            "compares='each_action' cannot be combined with renegotiate or settle."
        )


def _validate_names(*, kind: str, names: tuple[str, ...]) -> None:
    errors = path_segment_name_errors(kind=kind, names=names)
    if errors or any(not name.isidentifier() for name in names):
        raise RegimeInitializationError(
            "; ".join(errors) or f"{kind} names must be ordinary identifiers."
        )
