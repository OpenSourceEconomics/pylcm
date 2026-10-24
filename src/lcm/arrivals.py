"""Typed decisions after a scheduled landing and before its current action."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.collective import ProjectedRegimeValue, StakeholderRoute
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ReferenceName, UserFunction

type ArrivalValueKind = Literal["Raw", "Participating", "IntegratedFull"]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ArrivalRead(ProjectedRegimeValue):
    """Name the represented object read at a projected arrival state.

    Raw and Participating reconstruct the actual canonical action problem.
    IntegratedFull reads the stored, shock-integrated Full value representation;
    it does not claim to reconstruct an off-grid policy or its support.
    """

    kind: ArrivalValueKind
    """Exact policy problem or integrated stored-value object requested."""
    field: Literal["value", "support"] = "value"
    """Policy support is Boolean and is unavailable from IntegratedFull."""
    observes: tuple[str, ...] = ()
    """Physical target information observed by this arrival decision."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind == "IntegratedFull" and (self.field != "value" or self.observes):
            raise RegimeInitializationError(
                "IntegratedFull arrival reads expose only an integrated value, "
                "without policy support or realized physical information."
            )
        if len(set(self.observes)) != len(self.observes):
            raise RegimeInitializationError(
                "An arrival read must name each observed shock once."
            )
        object.__setattr__(self, "observes", tuple(self.observes))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ArrivalDecision:
    """Apply a Boolean rule to declared reads on this scheduled edge's arrival.

    The decision belongs to its incoming source, scheduled target, source role
    and arrival date. It is consumed before the final regime acts; another
    incoming edge or an initial occupant cannot borrow its fallback.
    """

    predicate: UserFunction
    """Exact Boolean comparison of arrival states and explicitly typed reads."""
    routes: Mapping[str, StakeholderRoute]
    """One actual settlement per original source stakeholder."""
    reads: Mapping[ReferenceName, ArrivalRead] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Projected same-date operands named explicitly in the predicate."""
    observes: tuple[str, ...] = ()
    """Target physical information observed before this arrival comparison."""

    def __post_init__(self) -> None:
        if not self.routes:
            raise RegimeInitializationError("An arrival decision needs source routes.")
        if (
            "D_target" in self.reads
            or "D_target" in inspect.signature(self.predicate).parameters
        ):
            raise RegimeInitializationError(
                "ArrivalDecision does not read D_target; declare typed policy support."
            )
        undeclared = {
            name
            for name in inspect.signature(self.predicate).parameters
            if (name == "V_target" or name.startswith("V_target_"))
            and name not in self.reads
        }
        if undeclared:
            raise RegimeInitializationError(
                f"Arrival value operands require explicit ArrivalRead: {undeclared}."
            )
        if len(set(self.observes)) != len(self.observes):
            raise RegimeInitializationError(
                "An arrival decision must name each observed shock once."
            )
        for name in ("routes", "reads"):
            object.__setattr__(
                self, name, ensure_containers_are_immutable(getattr(self, name))
            )
        object.__setattr__(self, "observes", tuple(self.observes))
