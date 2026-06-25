"""A DC-EGM model with no analytic inverse marginal utility solves numerically.

When a regime supplies no `inverse_marginal_utility`, EGM derives one numerically
from `utility` (the iEGM path) instead of failing. On a CRRA model — where the
analytic inverse exists — dropping it and forcing the numerical inverter must
reproduce the analytic-inverse value function: the numeric root finder recovers
the same `(u')^{-1}`.
"""

import numpy as np

from lcm import AgeGrid, Model
from lcm_examples.iskhakov_et_al_2017 import dead
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_model,
    get_retirement_only_params,
)


def _numeric_retirement_model(n_periods: int) -> Model:
    """The DC-EGM retirement model with `inverse_marginal_utility` removed."""
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    functions_without_inverse = {
        name: func
        for name, func in dcegm_retirement.functions.items()
        if name != "inverse_marginal_utility"
    }
    numeric_regime = dcegm_retirement.replace(
        active=lambda age, la=last_age: age < la,
        functions=functions_without_inverse,
    )
    return Model(
        regimes={"retirement": numeric_regime, "dead": dead},
        ages=ages,
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def test_iegm_numeric_inverse_matches_analytic_value_function():
    """Forcing the numerical inverse reproduces the analytic-inverse value function."""
    n_periods = 3
    params = get_retirement_only_params(n_periods)
    analytic = get_retirement_only_model("dcegm", n_periods).solve(
        params=params, log_level="off"
    )
    numeric = _numeric_retirement_model(n_periods).solve(params=params, log_level="off")

    assert analytic.keys() == numeric.keys()
    for period in analytic:
        for regime in analytic[period]:
            np.testing.assert_allclose(
                np.asarray(numeric[period][regime]),
                np.asarray(analytic[period][regime]),
                rtol=1e-6,
                atol=1e-6,
            )
