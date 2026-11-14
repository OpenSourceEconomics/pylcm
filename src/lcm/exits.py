"""Current-period exits and the mapped policy problems they read."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import path_segment_name_errors
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.shocks import MonteCarlo, QuasiMonteCarlo
from lcm.typing import RegimeName, StateName, UserFunction

type ExitRole = str


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ExitTarget:
    """A destination's Participating policy at projected current states."""

    regime: RegimeName
    """Actual destination regime."""
    projection: Mapping[StateName, UserFunction] | Phased
    """Complete destination-state mapping; Phased may change settlements."""
    role: ExitRole | None = None
    """Destination stakeholder, required for a collective destination."""
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
class Unilateral:
    """Any member can end the match: it continues only if every member is willing."""

    tie: Literal["stay", "leave"] = "stay"
    """What a member exactly indifferent between staying and the exit does."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MutualConsent:
    """The match ends only if every member is unwilling to stay."""

    tie: Literal["stay", "leave"] = "stay"
    """What a member exactly indifferent between staying and the exit does."""


type ExitRule = Unilateral | MutualConsent | Consent


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Exit:
    """The regime's current-period move, separate from scheduled next-period edges."""

    targets: Mapping[ExitRole, ExitTarget]
    """One destination per source stakeholder, or self for a singleton."""
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


def _validate_names(*, kind: str, names: tuple[str, ...]) -> None:
    errors = path_segment_name_errors(kind=kind, names=names)
    if errors or any(not name.isidentifier() for name in names):
        raise RegimeInitializationError(
            "; ".join(errors) or f"{kind} names must be ordinary identifiers."
        )
