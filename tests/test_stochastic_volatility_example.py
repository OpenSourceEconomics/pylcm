"""Smoke + behavior test for the stochastic-volatility example model."""

import numpy as np

import lcm
from lcm_examples.stochastic_volatility import get_model, get_params


def test_state_conditioned_is_public():
    assert lcm.StateConditioned is not None
    assert lcm.processes.StateConditioned is lcm.StateConditioned


def test_example_solves_and_uncertainty_matters():
    model = get_model(n_periods=6)
    V = model.solve(params=get_params(), log_level="debug").values
    # finite value everywhere
    for regime_to_value in V.values():
        for value in regime_to_value.values():
            assert np.all(np.isfinite(np.asarray(value)))
    # the value depends on the uncertainty regime (distinct per-regime sigmas)
    maxdiff = 0.0
    for regime_to_value in V.values():
        for value in regime_to_value.values():
            a = np.asarray(value)
            if a.ndim >= 1 and 2 in a.shape:
                ax = list(a.shape).index(2)
                maxdiff = max(
                    maxdiff,
                    float(np.abs(np.take(a, 0, ax) - np.take(a, 1, ax)).max()),
                )
    assert maxdiff > 1e-3
