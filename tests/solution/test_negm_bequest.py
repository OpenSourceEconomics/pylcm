"""NEGM supports a terminal bequest over two continuous states.

When the terminal regime values a bequest over both the liquid Euler state and
the durable (the NEGM passive/outer margin), the EGM parent carries a terminal
value with the Euler state plus the durable as a passive leading axis. The
nested-EGM solve reproduces the grid-search optimum within the coarse-grid
band — the Dobrescu-Shanker housing bequest pattern.
"""

import jax.numpy as jnp
import numpy as np

from tests.test_models import negm_bequest_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _solve_period0_alive(model) -> jnp.ndarray:
    return model.solve(params=_PARAMS, log_level="off")[0]["alive"]


def test_negm_solves_with_a_two_continuous_state_terminal_bequest():
    """The bequest NEGM model solves to a finite, well-shaped value array.

    The terminal `dead` regime carries both `wealth` and `illiquid`; the EGM
    parent reads the durable as a passive leading axis of the terminal carry.
    """
    v0 = _solve_period0_alive(negm_bequest_toy.build_negm_model())
    assert v0.shape == (negm_bequest_toy.N_X, negm_bequest_toy.N_Z)
    assert bool(jnp.all(jnp.isfinite(v0)))


def test_negm_value_is_monotone_increasing_in_both_assets():
    """The period-0 value rises with both liquid wealth and the durable stock."""
    v0 = _solve_period0_alive(negm_bequest_toy.build_negm_model())
    assert bool(jnp.all(jnp.diff(v0, axis=0) >= -1e-6))
    assert bool(jnp.all(jnp.diff(v0, axis=1) >= -1e-6))


def test_negm_value_matches_the_dense_brute_optimum():
    """NEGM reproduces the dense grid-search optimum within the grid band."""
    v_negm = np.asarray(_solve_period0_alive(negm_bequest_toy.build_negm_model()))
    v_brute = np.asarray(_solve_period0_alive(negm_bequest_toy.build_brute_model()))
    deviation = np.abs(v_negm - v_brute)
    assert float(deviation.mean()) < 0.06
    assert float(deviation.max()) < 0.15
