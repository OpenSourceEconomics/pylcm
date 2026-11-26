"""Select an adjustment before separation from independently optimized values."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp

from _lcm.policy_queries import PolicyAnswer
from _lcm.typing import PytreeValue
from _lcm.utils.functools import call_with_declared_arguments
from lcm.composition_actions import MemberName
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
    """Canonical code extent, increasing toward the first member."""
    operands: tuple[ReferenceName, ...]
    """Raw stay policy at each node, with the same other current states."""
    tie: Literal["stay", "leave"]
    """Whether equality to the current source's outside payoff permits staying."""

    def select(
        self,
        *,
        arguments: Mapping[ReferenceName, PytreeValue],
        outside: Float1D,
        undefined: Bool1D,
        active: BoolND,
    ) -> tuple[IntND, BoolND]:
        """Return the selected node index and whether the pair stays.

        A member whose destination has no feasible action (`undefined`) accepts
        every node with a feasible action, whatever the rule's `tie`.
        """
        answers = tuple(cast("PolicyAnswer", arguments[name]) for name in self.operands)
        values = jnp.stack(tuple(answer[0] for answer in answers))
        support = jnp.stack(tuple(answer[2] for answer in answers))
        willing = values >= outside if self.tie == "stay" else values > outside
        willing |= undefined
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
class TransferQueries:
    """Read the transfer nearest the default that the stayer accepts."""

    points: tuple[float, ...]
    """Candidate transfers, increasing in favour of the first member."""
    default: Callable[..., PytreeValue]
    """Source-context default transfer, clipped to the candidates' range."""
    targets: tuple[Mapping[MemberName, tuple[str, int]], ...]
    """Per candidate, each member's outside read and destination member index."""
    adjustments: tuple[Mapping[MemberName, Callable[..., PytreeValue]], ...]
    """Per candidate, each member's source-context adjustment at that transfer."""
    tie: Literal["stay", "leave"]
    """What a member prefers when its stay value equals the value of its outside
    option."""

    def select(
        self,
        *,
        inside: Float1D,
        willing: Bool1D,
        outside: Float2D,
        resolved: BoolND,
        supported: Bool1D,
        default: ScalarFloat,
    ) -> tuple[IntND, BoolND, BoolND]:
        """Return the accepted transfer and whether the pair separates there.

        Row 0 of `outside` and `supported` is the default and row `1 + k` candidate
        `k`; the returned index counts the same rows. Only a pair in which exactly
        one member refuses at the default separates at a candidate transfer. A
        member prefers its outside option when the value of that option is larger
        than its stay value, or equal under `tie="leave"`. The other member, the
        stayer, accepts separation where it prefers its outside option and every
        destination is supported. Willing to stay at the default, it never accepts
        there, so the transfer is the accepted candidate nearest the default
        strictly on the stayer's side. The pair separates there if the member who
        wants to leave prefers its outside option.

        Also returns whether every value of an outside option the decision reads
        is resolved. `resolved` holds this per candidate and member; the caller
        checks the default's. A pair in which not exactly one member refuses reads
        no candidate. Otherwise the decision reads the stayer's value at every
        supported candidate on its side up to the accepted one, or all of them when
        none accepts, and the leaver's value at the accepted one.
        """
        candidates = jnp.asarray(self.points)
        points = jnp.concatenate((jnp.reshape(default, (1,)), candidates))
        stayer = jnp.where(willing[0], 0, 1)
        leaver = 1 - stayer
        side = jnp.concatenate(
            (
                jnp.ones(1, dtype=bool),
                jnp.where(stayer == 0, candidates > default, candidates < default),
            )
        )
        prefers = outside > inside if self.tie == "stay" else outside >= inside
        accepts = side & supported & prefers[:, stayer]
        gains = prefers[:, leaver]
        distance = jnp.abs(points - default)
        ranked = jnp.where(accepts, distance, jnp.inf)
        nearest = jnp.argmin(ranked)
        one_refuses = willing[0] ^ willing[1]
        settles = one_refuses & jnp.any(accepts) & gains[nearest]
        rows = jnp.concatenate((jnp.ones((1, 2), dtype=bool), resolved))
        compared = side & supported & (distance <= ranked[nearest])
        decided = ~one_refuses | (
            jnp.all(~compared | rows[:, stayer])
            & (~jnp.any(accepts) | rows[nearest, leaver])
        )
        return nearest.astype(jnp.int32), settles, decided


@dataclass(frozen=True, kw_only=True)
class ClippedDefault:
    """Evaluate the default transfer and clip it to the candidates' range."""

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
            raise ValueError("A transfer default must be scalar.")
        return jnp.clip(value, self.low, self.high)


@dataclass(frozen=True, kw_only=True)
class FixedTransfer:
    """Bind one candidate transfer without a free model parameter."""

    value: float

    def __call__(self) -> FloatND:
        return jnp.asarray(self.value)


@dataclass(frozen=True, kw_only=True)
class FixedNode:
    """Bind one canonical discrete node without a free model parameter."""

    value: int

    def __call__(self) -> IntND:
        return jnp.asarray(self.value, dtype=jnp.int32)
