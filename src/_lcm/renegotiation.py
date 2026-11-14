"""Select a discrete adjustment from independently optimized stay policies."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import jax.numpy as jnp

from _lcm.policy_queries import PolicyAnswer
from _lcm.typing import PytreeValue
from lcm.typing import BoolND, Float1D, IntND, ReferenceName, StateName


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
class FixedNode:
    """Bind one canonical discrete node without a free model parameter."""

    value: int

    def __call__(self) -> IntND:
        return jnp.asarray(self.value, dtype=jnp.int32)
