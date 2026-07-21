"""Smoke + behavior test for the stochastic-volatility example model."""

import jax
import numpy as np

import lcm
from lcm_examples.stochastic_volatility import get_model, get_params


def test_state_conditioned_is_public():
    assert lcm.StateConditioned is not None
    assert lcm.processes.StateConditioned is lcm.StateConditioned


def test_example_solves_and_uncertainty_matters():
    model = get_model(n_periods=6)
    V = model.solve(params=get_params(), log_level="warning")
    # finite value everywhere
    for leaf in jax.tree_util.tree_leaves(V):
        assert np.all(np.isfinite(np.asarray(leaf)))
    # the value depends on the uncertainty regime (distinct per-regime sigmas)
    maxdiff = 0.0
    for leaf in jax.tree_util.tree_leaves(V):
        a = np.asarray(leaf)
        if a.ndim >= 1 and 2 in a.shape:
            ax = list(a.shape).index(2)
            maxdiff = max(
                maxdiff, float(np.abs(np.take(a, 0, ax) - np.take(a, 1, ax)).max())
            )
    assert maxdiff > 1e-3
