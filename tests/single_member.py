"""A single as the household of its one member `"self"`, for reducer-level tests.

The engine solves every regime as a household. A test that drives a reducer
directly with a single's `Q_and_F` gives that function the trailing member axis
here and passes the one member's Pareto weight.
"""

import functools
from collections.abc import Callable

import jax.numpy as jnp

from _lcm.regime_building.collective import ParetoWeights, build_pareto_weights
from lcm.typing import BoolND, FloatND

# The members of a single.
SELF: tuple[str, ...] = ("self",)

# The weight evaluator the engine builds for a single.
SELF_PARETO_WEIGHTS: ParetoWeights = build_pareto_weights(
    objective=None, members=SELF, state_names=frozenset()
)


def with_member_axis(
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Return `Q_and_F` publishing its value with the trailing member axis.

    The wrapper keeps the signature of `Q_and_F`, which the reducers read.
    """

    @functools.wraps(Q_and_F)
    def single(**kwargs: FloatND) -> tuple[FloatND, BoolND]:
        value, feasible = Q_and_F(**kwargs)
        return jnp.asarray(value)[..., jnp.newaxis], feasible

    return single
