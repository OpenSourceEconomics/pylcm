"""A collective regime fills its slots from the model level like any other regime.

A collective regime carries a per-stakeholder `utility_<s>` for each of its
stakeholders and at least one discrete action. Both are ordinary regime slots,
so a household whose regimes share them may declare them once on the `Model`
and let the broadcast supply every regime — completeness is a property of the
regime the model runs, not of a bare `Regime`.
"""

from dataclasses import replace

import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import DiscreteGrid, Model
from lcm.exceptions import RegimeInitializationError
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


def test_duplicate_stakeholders_are_reported_as_duplicates():
    """A stakeholder name repeated in `stakeholders` is reported as a duplicate.

    `stakeholders=("f", "f")` names one stakeholder twice. A regime taking its
    per-stakeholder utilities from the model level has nothing else wrong with
    it, so the error names the repeated stakeholder rather than describing the
    tuple in terms of a utility it would need.
    """
    couple = make_two_stakeholder_model()[0].user_regimes["couple"]

    with pytest.raises(RegimeInitializationError, match="duplicate"):
        replace(couple, stakeholders=("f", "f"), functions={})
