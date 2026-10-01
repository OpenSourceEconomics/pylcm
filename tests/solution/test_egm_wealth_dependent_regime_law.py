"""A regime law that reads the liquid state keeps its mass on the one-row EGM kernel.

A deterministic next-regime ID that depends on wealth publishes the same values
as the equivalent per-target Markov law.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, ByAge, Choose, LinSpacedGrid, MarkovTransition, Model, Regime
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.solvers import EGM, GridSearch
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.solution import test_egm_solver as egm_toy

_PER_TARGET_LAW = "per_target"
_DETERMINISTIC_LAW = "deterministic"


def _wealth_keeps_saving(
    *, age: int, wealth: ContinuousState, last_age: float
) -> BoolND:
    return (age + 1 < last_age) & (wealth > 0.0)


def _prob_keep_saving(*, age: int, wealth: ContinuousState, last_age: float) -> FloatND:
    return jnp.where(
        _wealth_keeps_saving(age=age, wealth=wealth, last_age=last_age), 1.0, 0.0
    )


def _prob_stop_saving(*, age: int, wealth: ContinuousState, last_age: float) -> FloatND:
    return 1.0 - _prob_keep_saving(age=age, wealth=wealth, last_age=last_age)


def _next_regime_by_wealth(
    *, age: int, wealth: ContinuousState, last_age: float
) -> ScalarInt:
    return jnp.where(
        _wealth_keeps_saving(age=age, wealth=wealth, last_age=last_age),
        egm_toy.RegimeId.saving,
        egm_toy.RegimeId.done,
    )


def _wealth_law_saving_values(*, law: str) -> np.ndarray:
    """Solve the EGM lifecycle whose regime law reads wealth, stacking its values.

    The law sends every node to the saving regime at ages 0 and 1 and to the
    done regime at age 2, so each node's single declared target carries weight one.
    """
    wealth_grid = LinSpacedGrid(start=2.0, stop=60.0, n_points=8)
    keep, stop = (
        (
            Choose(func=_next_regime_by_wealth, targets=("saving",)),
            Choose(func=_next_regime_by_wealth, targets=("done",)),
        )
        if law == _DETERMINISTIC_LAW
        else (
            {"saving": MarkovTransition(func=_prob_keep_saving)},
            {"done": MarkovTransition(func=_prob_stop_saving)},
        )
    )
    saving = ConsumptionSavingsRegime(
        states={"wealth": wealth_grid},
        actions={"consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=40)},
        functions={"utility": egm_toy.utility, "savings": egm_toy.savings},
        state_transitions={
            "wealth": {"saving": egm_toy.next_wealth, "done": egm_toy.next_wealth}
        },
        constraints={},
        regime_transitions=ByAge.until(stop_age_exclusive=3.0, law=keep, then=stop),
        solver=EGM(savings_grid=LinSpacedGrid(start=0.0, stop=60.0, n_points=40)),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    done = Regime(
        regime_transitions=None,
        states={"wealth": wealth_grid},
        functions={"utility": egm_toy.terminal_utility},
        solver=GridSearch(),
    )
    model = Model(
        regimes={"saving": saving, "done": done},
        regime_id_class=egm_toy.RegimeId,
        ages=AgeGrid(start=0, stop=3, step="Y"),
        fixed_params={"last_age": 3.0},
        initial_regimes={0: "saving"},
    )
    law_params = {"return_liquid": 0.03, "retirement_income": 0.0}
    params = {
        "saving": {
            "utility": {"crra": 2.0},
            "koopmans_aggregator": {"discount_factor": 0.95},
            "saving": {"next_wealth": law_params},
            "done": {"next_wealth": law_params},
        },
        "done": {"utility": {"crra": 2.0}},
    }
    values = model.solve(params=params, log_level="off").values
    return np.stack([np.asarray(values[period]["saving"]) for period in (0, 1, 2)])


def test_one_row_deterministic_wealth_law_matches_per_target_law() -> None:
    """A deterministic regime law that reads the liquid state keeps its mass.

    At every node the law's single declared target carries probability one, so
    the one-row EGM kernel publishes the same finite values as the equivalent
    per-target Markov law, within 8 ULP.
    """
    np.testing.assert_array_max_ulp(
        _wealth_law_saving_values(law=_DETERMINISTIC_LAW),
        _wealth_law_saving_values(law=_PER_TARGET_LAW),
        maxulp=8,
    )
