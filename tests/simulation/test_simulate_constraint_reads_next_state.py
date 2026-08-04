"""A constraint may read a `next_<state>` transition output in simulation.

The within-period decision resolves an auto-named `next_<state>` from the chosen
action rather than demanding it as an external input — the NEGM/DC-EGM durable
pattern, where the budget constraint cuts on the next durable stock. Everything
the simulation phase rebuilds from the regime's constraints has to resolve that
name the same way: the initial-conditions feasibility check, its per-constraint
diagnostic, and the additional-target pool.
"""

import jax.numpy as jnp
import pytest

from lcm import LinSpacedGrid, Model, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidInitialConditionsError
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.95}}, "dead": {}}
_N_PERIODS = 3


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.sqrt(consumption)


def next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return 1.05 * (wealth - consumption)


def savings_stay_above_the_floor(next_wealth: ContinuousState) -> BoolND:
    """The constraint under test: it reads the chosen next state, not the action."""
    return next_wealth >= 0.0


def next_regime(age: int) -> ScalarInt:
    return jnp.where(age + 1 >= _N_PERIODS - 1, RegimeId.dead, RegimeId.alive)


@pytest.fixture
def model() -> Model:
    """Model whose only constraint reads the auto-named `next_wealth`."""
    alive = UserRegime(
        transition=next_regime,
        active=lambda age: age < _N_PERIODS - 1,
        states={"wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=8)},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=20.0, n_points=8)},
        state_transitions={"wealth": next_wealth},
        functions={"utility": utility},
        constraints={"savings_stay_above_the_floor": savings_stay_above_the_floor},
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age >= _N_PERIODS - 1,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=_N_PERIODS, step="Y"),
        regime_id_class=RegimeId,
    )


def _initial_conditions(wealth: tuple[float, ...]) -> dict[str, object]:
    return {
        "wealth": jnp.asarray(wealth),
        "age": jnp.full(len(wealth), 0.0),
        "regime_id": jnp.full(len(wealth), RegimeId.alive, dtype=jnp.int32),
    }


def test_simulate_runs_when_a_constraint_reads_a_next_state(model):
    """Forward simulation completes and consumption stays inside the action grid."""
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions((5.0, 12.0, 20.0)),
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    consumption = result.to_dataframe().query("regime_name == 'alive'")["consumption"]
    assert consumption.between(0.5, 20.0).all()


def test_infeasible_subject_is_reported_by_constraint_name(model):
    """An unaffordable starting state names the constraint that rules it out."""
    with pytest.raises(
        InvalidInitialConditionsError, match="savings_stay_above_the_floor"
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=_initial_conditions((0.1,)),
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )


def test_the_constraint_is_available_as_an_additional_target(model):
    """`to_dataframe` can compute the next-state-reading constraint per row."""
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions((5.0, 12.0)),
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe(additional_targets=["savings_stay_above_the_floor"])
    assert df.query("regime_name == 'alive'")["savings_stay_above_the_floor"].all()
