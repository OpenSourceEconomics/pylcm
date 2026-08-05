"""The EGM solvers read preferences and the discount factor off the regime itself.

Neither `EGM` nor `TwoAssetEGM` assumes a preference family. Felicity, its
marginal, and its inverse marginal come from the regime's own `functions`, and
the discount factor from its Koopmans aggregator — so a regime that declares no
analytic `inverse_marginal_utility` is still solvable (the Euler equation is
inverted numerically), and a regime that declares an aggregator the Euler
inversion cannot represent is rejected rather than silently solved as linear.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from lcm import LinSpacedGrid
from lcm.exceptions import ModelInitializationError
from lcm.koopmans_aggregation import CESAggregator
from lcm.solvers import EGM, TwoAssetEGM
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5

_A_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=18)
_B_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=16)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=18)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)


def _solvers():
    return {
        "working": TwoAssetEGM(
            a_grid=_A_GRID, b_grid=_B_GRID, consumption_grid=_CONSUMPTION_GRID
        ),
        "retired": EGM(savings_grid=_SAVINGS_GRID),
    }


def test_numeric_inverse_reproduces_the_analytic_euler_inversion():
    """Dropping `inverse_marginal_utility` leaves every solved value unchanged.

    Without the analytic inverse the Euler equation `u'(c) = m` is solved for
    `c` by a bracketed Newton iteration on the regime's own `utility`, which
    reaches the same root the closed form gives.
    """
    analytic = get_model(n_periods=_N_PERIODS, solvers=_solvers()).solve(
        params=get_params(), log_level="debug"
    )
    numeric = get_model(
        n_periods=_N_PERIODS,
        solvers=_solvers(),
        analytic_inverse_marginal_utility=False,
    ).solve(params=get_params(), log_level="debug")

    for period, regime_to_V in analytic.items():
        for regime, V_arr in regime_to_V.items():
            expected = np.asarray(V_arr)
            got = np.asarray(numeric[period][regime])
            assert_allclose(
                got,
                expected,
                rtol=10.0 ** (-DECIMAL_PRECISION),
                err_msg=f"period {period}, regime {regime!r}",
            )


@pytest.mark.parametrize("regime_name", ["working", "retired"])
def test_custom_koopmans_aggregator_is_rejected_by_the_egm_solvers(regime_name):
    """An EGM regime whose aggregator is not the default `W` fails to build.

    The Euler inversion hard-codes `W = utility + discount_factor * CE`, so a
    CES aggregator changes the meaning of the solution rather than merely its
    parameters; the model reports that at build time.
    """
    with pytest.raises(ModelInitializationError, match="Koopmans aggregator"):
        get_model(
            n_periods=_N_PERIODS,
            solvers={regime_name: _solvers()[regime_name]},
            koopmans_aggregator=CESAggregator(),
        )
