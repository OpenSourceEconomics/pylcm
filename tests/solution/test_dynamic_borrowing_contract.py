"""Untaxed Markov borrowing floors through the public NB-EGM model API.

The floor binds by strict concavity and a negative savings derivative at the
corner. Expected values use the two terminal log payoffs directly.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solvers import NBEGM, GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON


@categorical(ordered=False)
class Income:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def savings(*, liquid: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return liquid - consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    return savings


def terminal_utility(*, liquid: ContinuousState, income: DiscreteState) -> FloatND:
    return jnp.log(liquid + jnp.where(income == Income.high, 2.0, 1.0))


def income_transition(income: DiscreteState) -> FloatND:
    high_probability = jnp.where(income == Income.high, 0.8, 0.2)
    return jnp.stack([1.0 - high_probability, high_probability])


def cell_floor(*, income: DiscreteState) -> FloatND:
    return jnp.where(income == Income.high, 0.25, 0.0)


def make_model(*, dynamic: bool) -> Model:
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="liquid",
        post_decision_state="savings",
    )
    source_grid = IrregSpacedGrid(points=np.array([0.5, 1.0, 2.0, 4.0]))
    target_grid = IrregSpacedGrid(points=np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]))
    income_grid = DiscreteGrid(category_class=Income)
    saving = ConsumptionSavingsRegime(
        states={"income": income_grid, "liquid": source_grid},
        actions={
            "consumption": IrregSpacedGrid(
                points=np.array([0.01, 0.25, 0.5, 1.0, 2.0, 4.0])
            )
        },
        functions={
            "utility": utility,
            "inverse_marginal_utility": inverse_marginal_utility,
            "savings": savings,
        },
        state_transitions={
            "liquid": {"done": next_liquid},
            "income": {"done": StochasticTransition(func=income_transition)},
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=margin,
                lower=cell_floor if dynamic else 0.0,
            )
        },
        liquid=margin,
        solver=NBEGM(
            savings_grid=IrregSpacedGrid(
                points=np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
            )
        ),
    )
    done = Regime(
        states={"liquid": target_grid, "income": income_grid},
        functions={"utility": terminal_utility},
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"saving": {"done": 0}},
        regime_id_class=RegimeId,
        initial_nodes={0: "saving"},
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize(
    "dynamic", [False, True], ids=["float_control", "callable_floor"]
)
def test_plain_markov_floor(*, dynamic: bool) -> None:
    """The income-specific floor binds at liquid resources of one half."""
    model = make_model(dynamic=dynamic)
    result = model.solve(params={"discount_factor": 1.0}, log_level="off")
    values = np.asarray(result.values[0]["saving"])
    assert tuple(model.user_regimes["saving"].states) == ("income", "liquid")
    assert tuple(model.user_regimes["done"].states) == ("liquid", "income")
    assert values.shape == (2, 4), "The source axes are income, then liquid."
    floors = (0.0, 0.25 if dynamic else 0.0)
    expected = []
    for bound, weights in zip(floors, ((0.8, 0.2), (0.2, 0.8)), strict=True):
        consumption = 0.5 - bound
        assert (
            -1 / consumption + weights[0] / (bound + 1) + weights[1] / (bound + 2) < 0
        )
        expected.append(
            math.log(consumption)
            + weights[0] * math.log(bound + 1)
            + weights[1] * math.log(bound + 2)
        )
    # Corner inputs are represented nodes; the budget applies only to this fixture.
    epsilon = np.finfo(values.dtype).eps
    np.testing.assert_allclose(
        values[:, 0], expected, rtol=1000 * epsilon, atol=1000 * epsilon
    )
