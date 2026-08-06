"""A target reached with probability zero contributes nothing to the continuation.

`-inf` is the ordinary value of a state at which every action is infeasible, and
a regime transition may send zero probability to the regime carrying it. The
term is then `0 · -inf`, which is NaN in floating point but zero as an
expectation: an event of probability zero carries no weight, whatever value sits
on it.

Every target enters the same certainty equivalent, so a NaN from one destroys
the well-specified targets beside it -- the states that *are* reachable lose
their value because of one that is not.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.Q_and_F import _scalar_target_contribution


@pytest.mark.parametrize("unreachable_value", [-np.inf, np.inf])
def test_a_zero_probability_scalar_target_leaves_the_continuation_intact(
    unreachable_value,
):
    """A zero-probability target adds zero, not NaN, beside a reachable one."""
    CE, _, _, mass = _scalar_target_contribution(
        scalar_targets=("reachable", "unreachable"),
        next_regime_to_V_arr={
            "reachable": jnp.asarray(2.0),
            "unreachable": jnp.asarray(unreachable_value),
        },
        active_regime_probs={
            "reachable": jnp.asarray(1.0),
            "unreachable": jnp.asarray(0.0),
        },
        as_lottery=False,
        zero=jnp.asarray(0.0),
    )

    np.testing.assert_allclose(np.asarray(CE), 2.0)
    np.testing.assert_allclose(np.asarray(mass), 1.0)


def test_a_reachable_infeasible_target_still_propagates_its_minus_infinity():
    """A target that is genuinely reached at `-inf` keeps making the value `-inf`."""
    CE, _, _, _ = _scalar_target_contribution(
        scalar_targets=("reachable", "infeasible"),
        next_regime_to_V_arr={
            "reachable": jnp.asarray(2.0),
            "infeasible": jnp.asarray(-np.inf),
        },
        active_regime_probs={
            "reachable": jnp.asarray(0.5),
            "infeasible": jnp.asarray(0.5),
        },
        as_lottery=False,
        zero=jnp.asarray(0.0),
    )

    assert np.isneginf(np.asarray(CE))
