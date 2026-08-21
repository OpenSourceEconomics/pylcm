"""Plain `EGM` uses every role its regime declares, or says it cannot.

The envelope-free kernel inverts the Euler equation against the liquid state
directly: it reads resources off that state and never evaluates the declared
resources function. A regime whose resources differ from its state therefore
gets an answer computed from something other than what it wrote down, which is
worse than a refusal — so the solver checks the identity it relies on.
"""

import pytest

from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME
from lcm.consumption_savings_regime import LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.solvers import EGM
from tests.solution.test_egm_solver import _SAVINGS_GRID, _model


def _egm_model():
    return _model(solver=EGM(savings_grid=_SAVINGS_GRID))


def test_the_liquid_state_can_fill_the_resources_role_directly():
    """A state used as resources needs no identity DAG function."""
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )

    model = _model(
        solver=EGM(savings_grid=_SAVINGS_GRID),
        resources_func=None,
        liquid=liquid,
    )

    assert "resources" not in model.user_regimes["saving"].functions
    assert model.user_regimes["saving"].liquid.resources == "wealth"


def test_resources_reading_beyond_the_state_are_refused():
    """`resources = wealth + transfer` reaches a leaf the kernel never applies."""

    def resources(wealth, transfer=1.0):
        return wealth + transfer

    with pytest.raises(ModelInitializationError, match="must be exactly a function"):
        _model(solver=EGM(savings_grid=_SAVINGS_GRID), resources_func=resources)


def test_resources_that_rescale_the_state_are_refused():
    """`resources = 2 * wealth` reads only the state and still is not it.

    The leaf arguments match, so nothing structural separates this from the
    identity; only evaluating it does. Left standing, the solve would invert the
    Euler equation against `wealth` and quietly ignore the factor.
    """

    def resources(wealth):
        return 2.0 * wealth

    with pytest.raises(ModelInitializationError, match="must equal the liquid state"):
        _model(solver=EGM(savings_grid=_SAVINGS_GRID), resources_func=resources)


def test_the_simulate_phase_carries_the_budget_mask():
    """The forward argmax over the gridded action space gets the same mask.

    The EGM solve enforces `consumption <= resources - borrowing_limit`
    intrinsically by inverting on the savings grid; simulation recomputes the
    argmax on the action grid and needs it stated. The solve phase never sees
    the synthesized constraint.
    """
    regime = _egm_model()._regimes["saving"]

    assert DCEGM_BUDGET_CONSTRAINT_NAME in regime.simulation.constraints
    assert DCEGM_BUDGET_CONSTRAINT_NAME not in regime.solution.constraints
