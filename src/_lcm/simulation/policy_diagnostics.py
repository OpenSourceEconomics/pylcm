"""Pure finite replay diagnostics whose complete allocation can be profiled."""

import jax
import jax.numpy as jnp
from jaxtyping import Integer

from lcm.typing import BoolND


def dropped_candidate_counts(
    *, live: BoolND, represented: BoolND
) -> Integer[jax.Array, "counts=2"]:
    """Count the original dropped predicate and its live candidate denominator."""
    return jnp.stack((jnp.sum(live & ~represented), jnp.sum(live)))
