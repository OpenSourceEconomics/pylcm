"""Current-period exits and the mapped policy problems they read."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import jax
from beartype import beartype
from jax import Array
from jaxtyping import Int

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import path_segment_name_errors
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.typing import RegimeName, StateName, UserFunction

type ExitRole = str
type _RuleAction = Int[Array, "..."] | int


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class RuleResult:
    """A custom rule's selected stable global alternative identifier."""

    action: _RuleAction
    """Integer ID; native consumption checks scalar shape and declared support."""


type ExitRule = Literal["unilateral", "mutual"] | Consent | Callable[..., RuleResult]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Exit:
    """One named current-period move, separate from scheduled next-period edges."""

    targets: Mapping[ExitRole, ExitTarget]
    """One destination per source stakeholder, or self for a singleton."""
    rule: ExitRule
    """Dissolution rule, strict consent, or a traceable declared callable."""
    available: UserFunction | None = None
    """State/parameter/observed-shock predicate, evaluated before benchmarks."""
    observes: tuple[str, ...] = ()
    """Target information observed at this choice, in addition to source reads."""
    _reads_outside_values: bool | None = field(init=False, repr=False)
    """Snapshot the custom rule's explicit DAG declaration into model identity."""

    def __post_init__(self) -> None:
        if not self.targets:
            raise RegimeInitializationError("An exit needs destination targets.")
        _validate_names(kind="Exit target", names=tuple(self.targets))
        _validate_names(kind="Observed shock", names=self.observes)
        if len(set(self.observes)) != len(self.observes):
            raise RegimeInitializationError("Exit observes must name each shock once.")
        object.__setattr__(
            self,
            "_reads_outside_values",
            _rule_reads_outside_values(cast("Callable[..., RuleResult]", self.rule))
            if callable(self.rule)
            else None,
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


def _validate_names(*, kind: str, names: tuple[str, ...]) -> None:
    errors = path_segment_name_errors(kind=kind, names=names)
    if errors or any(not name.isidentifier() for name in names):
        raise RegimeInitializationError(
            "; ".join(errors) or f"{kind} names must be ordinary identifiers."
        )


def _rule_reads_outside_values(rule: Callable[..., RuleResult]) -> bool:
    """Read authored metadata through only the engine's categorical adapters."""
    from _lcm.regime_building.value_adapter import ValueAdapter  # noqa: PLC0415

    while isinstance(rule, ValueAdapter):
        rule = cast("Callable[..., RuleResult]", rule.func)
    declared = getattr(rule, "reads_outside_values", None)
    if type(declared) is not bool:
        raise RegimeInitializationError(
            "A callable exit rule must declare Boolean reads_outside_values."
        )
    return declared
