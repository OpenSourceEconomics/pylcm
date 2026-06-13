"""DC-EGM splaying: `batch_size` on the Euler-state grid is a memory knob only.

A DC-EGM regime in asset-row mode solves the single-post-state pipeline per
exogenous Euler-state (asset) node. Splaying that axis — setting `batch_size`
on the Euler-state grid — processes the asset nodes in blocks rather than one
fused vmap, shedding peak working-set memory. It is a pure scheduling choice:
the solved value function is identical to the unsplayed (`batch_size=0`) solve,
whatever the block size, including block sizes that do not divide the grid.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_PERIODS = 4
N_WEALTH = 23  # deliberately prime-ish: no batch size below divides it evenly
BAND_START = 5.0
BAND_WIDTH = 40.0

CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=100.0, n_points=2000)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(110.0 * (i / 149) ** 3 for i in range(150)))


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def smoothstep(value: FloatND) -> FloatND:
    t = jnp.clip((value - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    # Reading the Euler state in the regime-transition probability switches the
    # kernel into the per-exogenous-asset-node (asset-row) solve.
    return 0.5 + 0.45 * smoothstep(wealth)


def stay_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth, age, final_age_alive)


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


@functools.cache
def _model(batch_size: int) -> Model:
    """Asset-row DC-EGM model with `batch_size` splaying on the Euler grid."""
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = UserRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(
                start=1.0, stop=100.0, n_points=N_WEALTH, batch_size=batch_size
            )
        },
        state_transitions={"wealth": next_wealth},
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=SAVINGS_GRID,
            n_constrained_points=32,
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=200)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def _params() -> dict:
    return {"discount_factor": 0.95, "final_age_alive": 40 + (N_PERIODS - 2) * 10}


def _solve(batch_size: int) -> PeriodToRegimeToVArr:
    return _model(batch_size).solve(params=_params(), log_level="debug")


@pytest.mark.parametrize("batch_size", [1, 4, 8, N_WEALTH])
def test_euler_grid_batch_size_leaves_value_function_unchanged(batch_size: int):
    """Splaying the Euler grid into blocks does not change the solved V.

    `batch_size` on the Euler-state grid only changes how the asset-row nodes
    are scheduled (blocks via `lax.map` instead of one fused vmap), so the
    value function at every period matches the unsplayed `batch_size=0` solve
    exactly — including block sizes that do not divide the grid.
    """
    reference = _solve(0)
    splayed = _solve(batch_size)
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"period={period}, regime={regime_name}",
            )
