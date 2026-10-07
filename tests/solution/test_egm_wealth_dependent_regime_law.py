"""A regime law that reads the liquid state keeps its mass on the one-row EGM kernel.

The one-row EGM kernel needs one active target per source age. A source with one
outgoing edge at every age takes the graph as its law, so a wealth-reading law is
not declarable there. With a second edge whose probability is a fixed zero, the
same wealth-reading law is declarable, its support stays one target per age, and
it publishes the values of the graph-only lifecycle.
"""

import jax.numpy as jnp
import numpy as np
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


def _never() -> FloatND:
    return jnp.asarray(0.0)


def _per_target_wealth_law() -> ByAge:
    """Keep saving while wealth is positive before the last age, then stop."""
    return ByAge.until(
        stop_age_exclusive=3.0,
        law={
            "saving": StochasticTransition(func=_prob_keep_saving),
            "done": StochasticTransition(func=_never),
        },
        then={"done": StochasticTransition(func=_prob_stop_saving)},
    )


def _build_model(*, edges: object, reads_last_age: bool = True) -> Model:
    """Build the EGM lifecycle that saves at ages 0 and 1 and stops at age 2.

    `reads_last_age` fixes the `last_age` parameter the wealth laws read; a
    graph-only lifecycle declares no law and so reads no such parameter.
    """
    wealth_grid = LinSpacedGrid(start=2.0, stop=60.0, n_points=8)
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
        fixed_params={"last_age": 3.0} if reads_last_age else {},
        initial_nodes={0: "saving"},
        edges=edges,
    )


def _build_wealth_law_model(*, law: str) -> Model:
    """Build the lifecycle whose single-destination source carries a wealth law.

    The law sends every node to the saving regime at ages 0 and 1 and to the
    done regime at age 2, matching the source's one outgoing edge at each age.
    """
    regime_law = (
        DeterministicTransition(func=_next_regime_by_wealth)
        if law == _DETERMINISTIC_LAW
        else ByAge.until(
            stop_age_exclusive=3.0,
            law={"saving": StochasticTransition(func=_prob_keep_saving)},
            then={"done": StochasticTransition(func=_prob_stop_saving)},
        )
    )
    return _build_model(
        edges={
            "saving": Transition(targets={"saving": (0, 1), "done": 2}, law=regime_law)
        }
    )


def _saving_values(model: Model) -> np.ndarray:
    """Solve the lifecycle and stack the saving regime's values over its ages."""
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


def test_one_row_wealth_law_matches_the_graph_only_lifecycle() -> None:
    """A per-target law that reads the liquid state keeps its mass.

    The law's second edge carries a fixed zero, so each age keeps one target, and
    at every node that target carries probability one. The one-row EGM kernel
    therefore publishes the same values as the lifecycle whose graph alone is the
    law, within 8 ULP.
    """
    with_law = _build_model(
        edges={
            "saving": Transition(
                targets={"saving": (0, 1), "done": (0, 1, 2)},
                law=_per_target_wealth_law(),
            )
        }
    )
    graph_only = _build_model(
        edges={"saving": {"saving": (0, 1), "done": 2}}, reads_last_age=False
    )
    np.testing.assert_array_max_ulp(
        _saving_values(with_law), _saving_values(graph_only), maxulp=8
    )


@pytest.mark.parametrize("law", [_DETERMINISTIC_LAW, _PER_TARGET_LAW])
def test_wealth_law_on_one_destination_egm_source_is_rejected(law: str) -> None:
    """A wealth-reading law on a source with one destination per age is rejected.

    The graph alone fixes the one-row EGM kernel's single target, so `Model`
    asks for the plain `{target: source_ages}` mapping instead.
    """
    with pytest.raises(ModelInitializationError, match="graph is its law"):
        _build_wealth_law_model(law=law)
