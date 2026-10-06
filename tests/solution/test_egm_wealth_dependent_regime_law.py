"""A regime law that reads the liquid state on a one-destination EGM source.

The one-row EGM kernel needs one active target per source age, and a source
with one outgoing edge at every age takes the graph as its law. A wealth-reading
law, deterministic or per-target, is therefore not declarable on such a source.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
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


def _build_wealth_law_model(*, law: str) -> Model:
    """Build the EGM lifecycle whose single-destination source carries a wealth law.

    The law sends every node to the saving regime at ages 0 and 1 and to the
    done regime at age 2, matching the source's one outgoing edge at each age.
    """
    wealth_grid = LinSpacedGrid(start=2.0, stop=60.0, n_points=8)
    regime_law = (
        DeterministicTransition(func=_next_regime_by_wealth)
        if law == _DETERMINISTIC_LAW
        else ByAge.until(
            stop_age_exclusive=3.0,
            law={"saving": StochasticTransition(func=_prob_keep_saving)},
            then={"done": StochasticTransition(func=_prob_stop_saving)},
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
        solver=EGM(savings_grid=LinSpacedGrid(start=0.0, stop=60.0, n_points=40)),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    done = Regime(
        states={"wealth": wealth_grid},
        functions={"utility": egm_toy.terminal_utility},
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        regime_id_class=egm_toy.RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        fixed_params={"last_age": 3.0},
        initial_nodes={0: "saving"},
        edges={
            "saving": Transition(targets={"saving": (0, 1), "done": 2}, law=regime_law)
        },
    )


@pytest.mark.parametrize("law", [_DETERMINISTIC_LAW, _PER_TARGET_LAW])
def test_wealth_law_on_one_destination_egm_source_is_rejected(law: str) -> None:
    """A wealth-reading law on a source with one destination per age is rejected.

    The graph alone fixes the one-row EGM kernel's single target, so `Model`
    asks for the plain `{target: source_ages}` mapping instead.
    """
    with pytest.raises(ModelInitializationError, match="graph is its law"):
        _build_wealth_law_model(law=law)
