"""Solver capability boundary for transition-local joint lotteries."""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    ExecutionConfig,
    JointTransition,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import DCEGM, EGM, NBEGM, GridSearch, OneMarginSolver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _target_utility(estate: ContinuousState) -> FloatND:
    return jnp.log(estate)


def _savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _certain_target() -> FloatND:
    return jnp.asarray(1.0)


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.25, 0.75])


def _next_estate(*, savings: FloatND, match: FloatND) -> ContinuousState:
    return savings + match


def _model(solver: OneMarginSolver | GridSearch) -> Model:
    source = ConsumptionSavingsRegime(
        regime_transitions={"target": StochasticTransition(func=_certain_target)},
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=10)},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=10.0, n_points=20)},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility, "savings": _savings},
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=2,
                    support=jnp.asarray([1.0, 4.0]),
                    probabilities=_joint_probabilities,
                    outputs={"estate": _next_estate},
                )
            }
        },
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
        solver=solver,
    )
    target = Regime(
        regime_transitions=None,
        states={"estate": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        functions={"utility": _target_utility},
    )
    return Model(
        regimes={"source": source, "target": target},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        initial_nodes={0: "source"},
        edges={"source": {"target": 0}},
    )


def test_grid_search_admits_transition_local_joint_lotteries() -> None:
    """Grid search enumerates the declared joint support inside Q."""
    model = _model(GridSearch())

    assert list(
        model._regimes["source"].solution.transition_plans["target"].lotteries
    ) == ["match"]


_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=20)


@pytest.mark.parametrize(
    "solver",
    [
        EGM(savings_grid=_SAVINGS_GRID),
        DCEGM(savings_grid=_SAVINGS_GRID),
        NBEGM(savings_grid=_SAVINGS_GRID),
    ],
    ids=["EGM", "DCEGM", "NBEGM"],
)
def test_egm_family_rejects_transition_local_joint_lotteries_during_construction(
    solver: OneMarginSolver,
) -> None:
    """An EGM-family solver refuses a joint lottery rather than failing in solve."""
    with pytest.raises(
        ModelInitializationError,
        match=rf"{type(solver).__name__}.*transition-local.*JointTransition.*GridSearch",
    ):
        _model(solver)
