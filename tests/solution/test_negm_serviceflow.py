"""NEGM supports `utility` reading the outer post-decision (the new durable).

When the durable's service flow accrues from the newly chosen stock `s'`
(`utility` reads `serviced_durable(next_illiquid)`, where `next_illiquid` is the
`outer_post_decision`), the nested-EGM solve binds that value per outer-grid
node and reproduces the grid-search optimum: the inner consumption margin is
off-grid (exact), so the two methods agree within the coarse-grid
discretization band on this deliberately tiny toy (8x8 states, 3 periods, a
durable withdrawal penalty and a credit-card rate kink). The residual is the
grid band, which tightens as the grids refine — not a bias of either method.
"""

import jax.numpy as jnp
import numpy as np

from tests.test_models import negm_serviceflow_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _solve_period0_alive(model) -> jnp.ndarray:
    return model.solve(params=_PARAMS, log_level="off")[0]["alive"]


def test_negm_solves_when_utility_reads_the_new_durable_stock():
    """The service-flow NEGM model solves to a finite, well-shaped value array.

    `utility` reads `next_illiquid` (the `outer_post_decision`) through the
    `serviced_durable` flow; the solver binds it per outer node, so the inner
    kernel evaluates without falling outside its scope.
    """
    v0 = _solve_period0_alive(negm_serviceflow_toy.build_negm_model())
    assert v0.shape == (negm_serviceflow_toy.N_X, negm_serviceflow_toy.N_Z)
    assert bool(jnp.all(jnp.isfinite(v0)))


def test_negm_value_is_monotone_increasing_in_both_assets():
    """The period-0 value rises with both liquid wealth and the durable stock."""
    v0 = _solve_period0_alive(negm_serviceflow_toy.build_negm_model())
    assert bool(jnp.all(jnp.diff(v0, axis=0) >= -1e-6))
    assert bool(jnp.all(jnp.diff(v0, axis=1) >= -1e-6))


def test_negm_value_matches_the_dense_brute_optimum():
    """NEGM reproduces the dense grid-search optimum within the grid band.

    The grid-search twin searches the same durable grid with a dense
    consumption grid, so it is the near-exact optimum. NEGM's off-grid
    consumption tracks it: the model-wide mean absolute deviation is small and
    no cell departs by more than the coarse-grid band.
    """
    v_negm = np.asarray(_solve_period0_alive(negm_serviceflow_toy.build_negm_model()))
    v_brute = np.asarray(_solve_period0_alive(negm_serviceflow_toy.build_brute_model()))
    deviation = np.abs(v_negm - v_brute)
    assert float(deviation.mean()) < 0.06
    assert float(deviation.max()) < 0.15
