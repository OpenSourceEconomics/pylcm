"""An NB-EGM continuation whose regime mass is lost publishes NaN.

A regime that sends positive transition probability to a target that is not
active next period has no continuation for that mass. The targets that remain
carry less than a distribution, so a finite blend of them would be a value that
does not depend on the missing target at all. Every continuation route answers
with NaN instead, at every log level:

- the grid-search route and simulation divide by the retained mass and poison a
  mass away from one;
- the NB-EGM linear blend and its Epstein-Zin blend poison the same mass with the
  same tolerance.

So the solve publishes NaN with `log_level="off"`, and at `log_level="warning"`
the value-function check reports the NaN rather than passing a finite value. At
`log_level="debug"` the transition validation refuses the lost mass before any
value is computed. Two specimens lose mass:

- the two-period multi-discrete toy whose survival law keeps every agent alive
  into a period where the alive regime is inactive;
- a stochastic-survival Epstein-Zin model whose living regime ends one period
  before its survival law stops sending mass to it.

Each has a correctly configured control that stays finite.
"""

import logging
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CESAggregator,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.solvers import NBEGM, GridSearch, OneMarginSolver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.test_models import nbegm_multi_discrete_toy as toy

_ALIVE = "alive"
_NBEGM = "nbegm"
_BRUTE = "brute"
_LINEAR = "linear"
_EPSTEIN_ZIN = "epstein_zin"

# The toy's survival law keeps an agent alive while `age + 1 < final_age_alive`.
# With two periods (ages 0 and 1) the alive regime is active at age 0 only, so
# `final_age_alive = 2` sends all mass from age 0 to the inactive alive regime and
# `final_age_alive = 1` sends it to the dead regime.
_TOY_LOST_FINAL_AGE_ALIVE = 2.0
_TOY_KEPT_FINAL_AGE_ALIVE = 1.0

_EZ_FIRST_AGE = 20
_EZ_LAST_LIVING_AGE = 25
_EZ_SURVIVAL = 0.9
_EZ_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=10)
_EZ_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=15.0, n_points=30)
_EZ_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=30)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _ez_utility(consumption: ContinuousAction) -> FloatND:
    """Consumption flow, in the positive units a power certainty equivalent needs."""
    return consumption


def _ez_resources(*, liquid: ContinuousState, base_income: float) -> FloatND:
    return liquid + base_income


def _ez_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _ez_feasible(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= resources


def _ez_next_liquid(*, savings: FloatND, income: ContinuousState) -> ContinuousState:
    return 1.03 * savings + 0.3 * jnp.exp(income)


def _ez_bequest(liquid: ContinuousState) -> FloatND:
    """Strictly positive terminal estate value."""
    return jnp.sqrt(liquid + 1.0)


def _ez_prob_alive(*, age: int, final_age_alive: float) -> FloatND:
    """Survive with probability `_EZ_SURVIVAL` until `final_age_alive`, then die."""
    return jnp.where(age >= final_age_alive, 0.0, _EZ_SURVIVAL)


def _ez_prob_dead(*, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 1.0 - _EZ_SURVIVAL)


def _build_ez_model(*, solver: OneMarginSolver | GridSearch, lost_mass: bool) -> Model:
    """Build the stochastic-survival Epstein-Zin model over ages 20, 25 and 30.

    The living regime is active through age 25. Its survival law stops sending
    mass to it at `final_age_alive`: at 25 when the mass is kept, and at 30 when
    it is lost, so that at age 25 the survivors' mass goes to an inactive regime.
    """
    alive = ConsumptionSavingsRegime(
        active=lambda age: age <= _EZ_LAST_LIVING_AGE,
        states={
            "liquid": _EZ_LIQUID_GRID,
            "income": NormalIIDProcess(n_points=3, gauss_hermite=True),
        },
        state_transitions={
            "liquid": {"alive": _ez_next_liquid, "dead": _ez_next_liquid}
        },
        actions={"consumption": _EZ_CONSUMPTION_GRID},
        transition={
            "alive": MarkovTransition(_ez_prob_alive),
            "dead": MarkovTransition(_ez_prob_dead),
        },
        functions={
            "utility": _ez_utility,
            "resources": _ez_resources,
            "savings": _ez_savings,
        },
        constraints=(
            {} if isinstance(solver, OneMarginSolver) else {"feasible": _ez_feasible}
        ),
        koopmans_aggregator=CESAggregator(),
        certainty_equivalent=PowerMean(),
        solver=solver,
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        transition=None,
        active=lambda age: age > _EZ_FIRST_AGE,
        states={"liquid": _EZ_LIQUID_GRID},
        functions={"utility": _ez_bequest},
    )
    final_age_alive = float(_EZ_LAST_LIVING_AGE + (5 if lost_mass else 0))
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=_EZ_FIRST_AGE, stop=_EZ_LAST_LIVING_AGE + 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


_EZ_PARAMS = {
    "alive": {
        "koopmans_aggregator": {
            "discount_factor": 0.95,
            "intertemporal_elasticity_of_substitution": 1.5,
        },
        "certainty_equivalent": {"risk_aversion": 4.0},
        "resources": {"base_income": 1.0},
        "income": {"mu": 0.0, "sigma": 0.2},
    },
    "dead": {},
}


def _model_and_params(
    *, preferences: str, solver: str, lost_mass: bool
) -> tuple[Model, Mapping[str, Any]]:
    """Build one specimen with its parameters."""
    if preferences == _LINEAR:
        model = toy.build_model(
            variant=solver,
            n_actions=3,
            n_periods=2,
            n_liquid=8,
            n_savings=10,
            n_consumption=12,
            envelope_arithmetic="ordinary",
        )
        final_age_alive = (
            _TOY_LOST_FINAL_AGE_ALIVE if lost_mass else _TOY_KEPT_FINAL_AGE_ALIVE
        )
        return model, toy.build_params(n_actions=3, final_age_alive=final_age_alive)
    ez_solver = (
        NBEGM(savings_grid=_EZ_SAVINGS_GRID, envelope_arithmetic="ordinary")
        if solver == _NBEGM
        else GridSearch()
    )
    return _build_ez_model(solver=ez_solver, lost_mass=lost_mass), _EZ_PARAMS


def _first_period_alive_value(
    *, preferences: str, solver: str, lost_mass: bool
) -> np.ndarray:
    """Solve with `log_level="off"` and return the alive value at the first age."""
    model, params = _model_and_params(
        preferences=preferences, solver=solver, lost_mass=lost_mass
    )
    values = model.solve(params=params, log_level="off").values
    return np.asarray(values[0][_ALIVE])


@pytest.mark.parametrize("solver", [_NBEGM, _BRUTE])
@pytest.mark.parametrize("preferences", [_LINEAR, _EPSTEIN_ZIN])
def test_lost_regime_mass_publishes_nan_value(*, preferences: str, solver: str) -> None:
    """Every alive value is NaN when the survivors' mass reaches an inactive regime,
    on the NB-EGM route exactly as on the grid-search route."""
    value = _first_period_alive_value(
        preferences=preferences, solver=solver, lost_mass=True
    )

    assert np.isnan(value).all()


@pytest.mark.parametrize("preferences", [_LINEAR, _EPSTEIN_ZIN])
def test_lost_regime_mass_raises_at_debug_level(*, preferences: str) -> None:
    """At `log_level="debug"` the NB-EGM solve refuses the lost mass."""
    model, params = _model_and_params(
        preferences=preferences, solver=_NBEGM, lost_mass=True
    )

    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params=params, log_level="debug")


@pytest.mark.parametrize("preferences", [_LINEAR, _EPSTEIN_ZIN])
def test_lost_regime_mass_warns_at_warning_level(
    *, preferences: str, caplog: pytest.LogCaptureFixture
) -> None:
    """At `log_level="warning"` the NB-EGM solve reports the alive regime's NaN
    value function."""
    model, params = _model_and_params(
        preferences=preferences, solver=_NBEGM, lost_mass=True
    )
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(params=params, log_level="warning")

    assert any(
        f"in regime '{_ALIVE}': all values are NaN" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("preferences", [_LINEAR, _EPSTEIN_ZIN])
def test_kept_regime_mass_publishes_finite_value(*, preferences: str) -> None:
    """With the survival law matching the active ages, every alive value is
    finite."""
    value = _first_period_alive_value(
        preferences=preferences, solver=_NBEGM, lost_mass=False
    )

    assert np.isfinite(value).all()
