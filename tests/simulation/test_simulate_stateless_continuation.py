"""Simulated choices price the value of a reachable regime that carries no state.

The solve phase and the simulate phase each build their own `Q`. A stateless
target contributes its value to both: the solved `V` prices it, and so must the
`Q` that selects actions during forward simulation. If only the solve phase reads
it, simulated paths are priced off a correct `V` but chosen by a `Q` that is blind
to the payoff of leaving — and the two disagree.

The oracle is a choice, not a level. Consumption at or above the threshold sends
the agent to the stateless regime; below it the agent stays. Raising the bequest
must therefore flip the chosen action from staying to leaving.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ScalarInt

_DISCOUNT = 0.95
_LEAVE_AT_CONSUMPTION = 0.7
_LAST_AGE = 22
_VALUE_OF_BEING_ALIVE = 10.0


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def _utility(consumption):
    # The flat bonus makes staying alive worth something, so a zero bequest is
    # genuinely dominated by staying. Without it the agent consumes the maximum
    # at every bequest — utility is increasing in consumption — and the model
    # could not discriminate the two rules.
    return jnp.log(consumption) + _VALUE_OF_BEING_ALIVE


def _next_wealth(wealth, consumption):
    return wealth - consumption


def _next_regime(consumption, age):
    # The action itself decides the regime, so the value of the stateless target
    # enters `Q` action by action. Everyone leaves after the last active age, so
    # no probability mass is ever sent to an inactive regime.
    leaves = (consumption >= _LEAVE_AT_CONSUMPTION) | (age >= _LAST_AGE - 1)
    return jnp.where(leaves, RegimeId.gone, RegimeId.alive)


def _simulate_with_bequest(bequest: float):
    """Simulate a two-regime model whose terminal regime carries no state."""
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
    )
    gone = Regime(transition=None, functions={"utility": lambda: jnp.array(bequest)})
    model = Model(
        regimes={"alive": alive, "gone": gone},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "H": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_regime": {},
        },
        "gone": {"utility": {}},
    }
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.array([20.0]),
            "wealth": jnp.array([5.0]),
            "regime_id": jnp.array([RegimeId.alive]),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    return result.to_dataframe()


def test_a_stateless_bequest_moves_the_simulated_action():
    """A large bequest makes the simulated agent choose the leaving action.

    With no bequest, leaving is worth nothing and the agent consumes below the
    threshold to stay alive. With a large bequest, leaving dominates and the
    chosen consumption jumps to the threshold or above.
    """
    poor = _simulate_with_bequest(0.0)
    rich = _simulate_with_bequest(1000.0)

    first_period = 0
    poor_consumption = np.atleast_1d(poor.loc[first_period, "consumption"])
    rich_consumption = np.atleast_1d(rich.loc[first_period, "consumption"])

    assert np.all(poor_consumption < _LEAVE_AT_CONSUMPTION)
    assert np.all(rich_consumption >= _LEAVE_AT_CONSUMPTION)
