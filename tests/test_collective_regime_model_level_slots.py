"""A collective regime fills its slots from the model level like any other regime.

A collective regime carries a body for each stakeholder its household names, and
may carry actions. Both are ordinary regime slots, so a household whose regimes
share them may declare them once on the `Model` and let the broadcast supply
every regime — completeness is a property of the regime the model runs, not of a
bare `Regime`.
"""

from dataclasses import replace

from numpy.testing import assert_array_almost_equal as aaae

from lcm import DiscreteGrid, Model
from tests.collective_fixtures import (
    AGES,
    TWO_STAKEHOLDER_V_PERIOD_0,
    CoupleRegimeId,
    Work,
    make_two_stakeholder_model,
)
from tests.conftest import DECIMAL_PRECISION


def test_collective_utilities_may_come_from_the_model_level_slot():
    """Per-stakeholder utilities declared on the model solve the collective regime.

    Model-level `functions=` accepts exactly what the regime slot accepts, so a
    couple whose regimes share `utility_f` and `utility_m` may declare the pair
    once. The household then solves to the values those utilities produce when
    each regime declares them itself.
    """
    model, params = make_two_stakeholder_model()
    # The per-stakeholder utilities the fixture declares on each regime.
    utilities = dict(model.user_regimes["couple"].functions)

    broadcast_model = Model(
        regimes={
            name: replace(regime, functions={})
            for name, regime in model.user_regimes.items()
        },
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        functions=utilities,
    )

    solution = broadcast_model.solve(params=params, log_level="debug")

    aaae(solution[0]["couple"], TWO_STAKEHOLDER_V_PERIOD_0, decimal=DECIMAL_PRECISION)


def test_collective_discrete_action_may_come_from_the_model_level_slot():
    """A discrete action declared on the model solves the collective regime.

    The household argmax runs over the discrete-action product, and the action
    it runs over may be declared once at model level. Broadcasting the couple's
    only action solves to the values that action produces when each regime
    declares it itself.
    """
    model, params = make_two_stakeholder_model()

    broadcast_model = Model(
        regimes={
            name: replace(regime, actions={})
            for name, regime in model.user_regimes.items()
        },
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        actions={"work": DiscreteGrid(Work)},
    )

    solution = broadcast_model.solve(params=params, log_level="debug")

    aaae(solution[0]["couple"], TWO_STAKEHOLDER_V_PERIOD_0, decimal=DECIMAL_PRECISION)
