"""Select an adjustment before separation from independently optimized values."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp

from _lcm.policy_queries import PolicyAnswer
from _lcm.typing import PytreeValue
from _lcm.utils.functools import call_with_declared_arguments
from lcm.exits import ExitRole
from lcm.typing import (
    Bool1D,
    BoolND,
    Float1D,
    Float2D,
    FloatND,
    IntND,
    ReferenceName,
    ScalarFloat,
    StateName,
)


@dataclass(frozen=True, kw_only=True)
class RenegotiationQueries:
    """Read the first willing node without crossing the other member's refusal."""

    state: StateName
    """Canonical discrete source state."""
    n_nodes: int
    """Canonical code extent, increasing toward the first stakeholder."""
    operands: tuple[ReferenceName, ...]
    """Raw stay policy at each node, with the same other current states."""
    tie: Literal["stay", "leave"]
    """Whether equality to the current source's outside payoff permits staying."""

    def select(
        self,
        *,
        arguments: Mapping[ReferenceName, PytreeValue],
        outside: Float1D,
        active: BoolND,
    ) -> tuple[IntND, BoolND]:
        """Return the selected node index and whether the pair stays."""
        answers = tuple(cast("PolicyAnswer", arguments[name]) for name in self.operands)
        values = jnp.stack(tuple(answer[0] for answer in answers))
        support = jnp.stack(tuple(answer[2] for answer in answers))
        willing = values >= outside if self.tie == "stay" else values > outside
        willing &= support[:, None]
        incumbent = jnp.asarray(arguments[self.state], dtype=jnp.int32)
        current = willing[incumbent]
        direction = jnp.where(current[0], -1, 1)
        path = incumbent + direction * jnp.arange(1, self.n_nodes + 1, dtype=jnp.int32)
        inside = (path >= 0) & (path < self.n_nodes)
        bounded = jnp.clip(path, 0, self.n_nodes - 1)
        both = jnp.all(willing[bounded], axis=1)
        other = jnp.where(direction > 0, willing[bounded, 1], willing[bounded, 0])
        # The first event is acceptance, the other member's veto, or a grid edge.
        first = jnp.argmax(~inside | both | ~other)
        found = inside[first] & both[first]
        move = active & (current[0] ^ current[1]) & found
        selected = jnp.where(move, bounded[first], incumbent).astype(jnp.int32)
        kept = jnp.where(active, jnp.all(current) | move, support[incumbent])
        return selected, kept


@dataclass(frozen=True, kw_only=True)
class SettlementQueries:
    """Read the candidate settlement nearest the default that the stayer accepts."""

    points: tuple[float, ...]
    """Candidate settlements, increasing in favour of the first stakeholder."""
    default: Callable[..., PytreeValue]
    """Source-context default settlement, clipped to the candidates' range."""
    targets: tuple[Mapping[ExitRole, tuple[str, int | None]], ...]
    """Per candidate, each member's outside read and destination stakeholder index."""
    adjustments: tuple[Mapping[ExitRole, Callable[..., PytreeValue]], ...]
    """Per candidate, each member's source-context adjustment at that settlement."""
    tie: Literal["stay", "leave"]
    """Whether the member who wants to leave still leaves when indifferent."""

    def select(
        self,
        *,
        inside: Float1D,
        willing: Bool1D,
        outside: Float2D,
        supported: Bool1D,
        default: ScalarFloat,
    ) -> tuple[IntND, BoolND]:
        """Return the nearest accepted candidate and whether the pair separates there.

        Only a pair in which exactly one member refuses at the default searches.
        The other member, the stayer, accepts separation at a candidate whose
        outside value is at least its stay value, whatever the rule's `tie`.
        """
        points = jnp.asarray(self.points)
        stayer = jnp.where(willing[0], 0, 1)
        leaver = 1 - stayer
        side = jnp.where(stayer == 0, points > default, points < default)
        accepts = side & supported & (outside[:, stayer] >= inside[stayer])
        gains = (
            outside[:, leaver] > inside[leaver]
            if self.tie == "stay"
            else outside[:, leaver] >= inside[leaver]
        )
        nearest = jnp.argmin(jnp.where(accepts, jnp.abs(points - default), jnp.inf))
        settles = (willing[0] ^ willing[1]) & jnp.any(accepts) & gains[nearest]
        return nearest.astype(jnp.int32), settles


@dataclass(frozen=True, kw_only=True)
class ClippedDefault:
    """Evaluate the default settlement and clip it to the candidates' range."""

    function: Callable[..., PytreeValue]
    """Bound source-context default."""
    low: float
    """Smallest candidate."""
    high: float
    """Largest candidate."""

    def __call__(self, **operands: PytreeValue) -> FloatND:
        value = jnp.asarray(
            call_with_declared_arguments(function=self.function, operands=operands)
        )
        if value.shape != ():
            raise ValueError("A settlement default must be scalar.")
        return jnp.clip(value, self.low, self.high)


@dataclass(frozen=True, kw_only=True)
class FixedSettlement:
    """Bind one candidate settlement without a free model parameter."""

    value: float

    def __call__(self) -> FloatND:
        return jnp.asarray(self.value)


@dataclass(frozen=True, kw_only=True)
class FixedNode:
    """Bind one canonical discrete node without a free model parameter."""

    value: int

    def __call__(self) -> IntND:
        return jnp.asarray(self.value, dtype=jnp.int32)
