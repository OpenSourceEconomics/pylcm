"""Regime mass of a stochastic-survival consumption-savings model under NB-EGM.

Preferences are either Epstein-Zin (CES aggregator, power certainty equivalent)
or additive with a linear expectation.

The living regime is solved at ages 20 and 25 and, at 25, declares only `dead`.
Its survival law sends mass to `alive` with probability 0.9 until
`final_age_alive`, then to `dead` with certainty:

- `final_age_alive = 25` keeps the mass inside the declared support, and the
  solve publishes finite values;
- `final_age_alive = 30` sends the survivors at 25 to `alive`, which that age
  does not declare, and regime selection refuses the law on every solver and
  certainty equivalent.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    CESAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    StochasticTransition,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.solvers import NBEGM, GridSearch, OneMarginSolver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.graph import with_fixture_graph
from tests.test_models.schedules import until_exit

_FIRST_AGE = 20
_LAST_LIVING_AGE = 25
_SURVIVAL = 0.9
_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=10)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=15.0, n_points=30)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=30)

# Age-20 alive value at the middle income node, at liquid nodes 0, 4 and 9,
# from the NB-EGM solve with the survival law kept inside the declared support.
_EXPECTED_KEPT_V = {
    "power": (1.6457975835281529, 3.0644089390925022, 4.226684047254188),
    "linear": (3.666423634429373, 12.33309030109604, 23.16642363442937),
}


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@pytest.mark.parametrize("certainty_equivalent", ["power", "linear"])
def test_kept_regime_mass_publishes_the_pinned_value(
    *, certainty_equivalent: str
) -> None:
    """With the survival law inside the declared support, NB-EGM solves to the pin."""
    model = _build_model(
        solver=NBEGM(savings_grid=_SAVINGS_GRID, envelope_arithmetic="ordinary"),
        certainty_equivalent=certainty_equivalent,
        lost_mass=False,
    )
    values = model.solve(
        params=_params(certainty_equivalent=certainty_equivalent), log_level="off"
    ).values
    np.testing.assert_array_almost_equal(
        np.asarray(values[0]["alive"])[1, [0, 4, 9]],
        _EXPECTED_KEPT_V[certainty_equivalent],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("log_level", ["off", "debug"])
@pytest.mark.parametrize("certainty_equivalent", ["power", "linear"])
@pytest.mark.parametrize("solver", ["grid", "nbegm"])
def test_lost_regime_mass_is_refused(
    *, solver: str, certainty_equivalent: str, log_level: LogLevel
) -> None:
    """Survivors sent to `alive` at 25, where only `dead` is declared, are refused."""
    model = _build_model(
        solver=(
            GridSearch()
            if solver == "grid"
            else NBEGM(savings_grid=_SAVINGS_GRID, envelope_arithmetic="ordinary")
        ),
        certainty_equivalent=certainty_equivalent,
        lost_mass=True,
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"^Regime transition probabilities from 'alive' between ages 25 and 30",
    ):
        model.solve(
            params=_params(certainty_equivalent=certainty_equivalent),
            log_level=log_level,
        )


def _build_model(
    *,
    solver: OneMarginSolver | GridSearch,
    certainty_equivalent: str,
    lost_mass: bool,
) -> Model:
    """Build the stochastic-survival Epstein-Zin model over ages 20, 25 and 30."""
    alive = ConsumptionSavingsRegime(
        states={
            "liquid": _LIQUID_GRID,
            "income": NormalIIDProcess(n_points=3, gauss_hermite=True),
        },
        state_transitions={"liquid": {"alive": _next_liquid, "dead": _next_liquid}},
        actions={"consumption": _CONSUMPTION_GRID},
        regime_transitions=until_exit(
            _LAST_LIVING_AGE + 5,
            law={
                "alive": StochasticTransition(func=_prob_alive),
                "dead": StochasticTransition(func=_prob_dead),
            },
            exits=("dead",),
        ),
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
        },
        constraints=(
            {} if isinstance(solver, OneMarginSolver) else {"feasible": _feasible}
        ),
        koopmans_aggregator=CESAggregator()
        if certainty_equivalent == "power"
        else None,
        certainty_equivalent=(
            PowerMean() if certainty_equivalent == "power" else LinearExpectation()
        ),
        solver=solver,
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        regime_transitions=None,
        states={"liquid": _LIQUID_GRID},
        functions={"utility": _bequest},
    )
    return with_fixture_graph(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=_FIRST_AGE, inclusive_stop=_LAST_LIVING_AGE + 5, step="5Y"),
        fixed_params={
            "final_age_alive": float(_LAST_LIVING_AGE + (5 if lost_mass else 0))
        },
        initial_nodes={_FIRST_AGE: "alive"},
    )


def _params(*, certainty_equivalent: str) -> dict:
    alive: dict = {
        "koopmans_aggregator": {"discount_factor": 0.95},
        "resources": {"base_income": 1.0},
        "income": {"mu": 0.0, "sigma": 0.2},
    }
    if certainty_equivalent == "power":
        alive["koopmans_aggregator"]["intertemporal_elasticity_of_substitution"] = 1.5
        alive["certainty_equivalent"] = {"risk_aversion": 4.0}
    return {"alive": alive, "dead": {}}


def _utility(consumption: ContinuousAction) -> FloatND:
    """Consumption flow, in the positive units a power certainty equivalent needs."""
    return consumption


def _resources(*, liquid: ContinuousState, base_income: float) -> FloatND:
    return liquid + base_income


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= resources


def _next_liquid(*, savings: FloatND, income: ContinuousState) -> ContinuousState:
    return 1.03 * savings + 0.3 * jnp.exp(income)


def _bequest(liquid: ContinuousState) -> FloatND:
    """Strictly positive terminal estate value."""
    return jnp.sqrt(liquid + 1.0)


def _prob_alive(*, age: int, final_age_alive: float) -> FloatND:
    """Survive with probability `_SURVIVAL` until `final_age_alive`, then die."""
    return jnp.where(age >= final_age_alive, 0.0, _SURVIVAL)


def _prob_dead(*, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 1.0 - _SURVIVAL)


def test_nbegm_refuses_a_ces_aggregator_under_expected_utility() -> None:
    """NB-EGM's expected-utility route solves only the additive aggregator."""
    alive = ConsumptionSavingsRegime(
        states={"liquid": _LIQUID_GRID},
        state_transitions={"liquid": {"dead": _next_liquid_certain}},
        actions={"consumption": _CONSUMPTION_GRID},
        regime_transitions={"dead": StochasticTransition(func=_certain_death)},
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
        },
        koopmans_aggregator=CESAggregator(),
        certainty_equivalent=LinearExpectation(),
        solver=NBEGM(savings_grid=_SAVINGS_GRID, envelope_arithmetic="ordinary"),
        liquid=LiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        regime_transitions=None,
        states={"liquid": _LIQUID_GRID},
        functions={"utility": _bequest},
    )
    with pytest.raises(RegimeInitializationError, match="LinearAggregator"):
        with_fixture_graph(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=_RegimeId,
            ages=AgeGrid(start=_FIRST_AGE, inclusive_stop=_LAST_LIVING_AGE, step="5Y"),
            initial_nodes={_FIRST_AGE: "alive"},
        )


def _next_liquid_certain(savings: FloatND) -> ContinuousState:
    return 1.03 * savings


def _certain_death() -> FloatND:
    return jnp.asarray(1.0)
