"""Simulation reads only the edge references of targets reachable in a period.

A regime may declare two gated edges whose targets are active over disjoint age
windows. In a period where one edge carries all the probability, the other edge's
gate references and leg fallbacks name regimes the next period never solved, and
simulation must not demand their landing values.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    near: ScalarInt
    far: ScalarInt
    far_fallback: ScalarInt


def utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def utility_state(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 1.0


def feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def keep_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def always_open(wealth: ContinuousState) -> BoolND:
    return wealth > -1.0


def p_near(age: int) -> FloatND:
    return jnp.where(age < 1, 1.0, 0.0)


def p_far(age: int) -> FloatND:
    return jnp.where(age >= 1, 1.0, 0.0)


def near_next(age: int) -> ScalarInt:
    return jnp.full(jnp.shape(age), RegimeId.far, dtype=jnp.int32)


def _edge(*, probability, fallback_regime: str) -> ValueDependentTransition:
    return ValueDependentTransition(
        probability=MarkovTransition(probability),
        gate=always_open,
        routes={
            "only": StakeholderRoute(
                target_stakeholder=None,
                fallback=ProjectedRegimeValue(
                    regime=fallback_regime, projection={"wealth": keep_wealth}
                ),
            )
        },
    )


WEALTH_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=9.0, n_points=5)


def _decision_regime(*, transition: object, active: object) -> Regime:
    return Regime(
        transition=transition,  # ty: ignore[invalid-argument-type]
        active=active,  # ty: ignore[invalid-argument-type]
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": next_wealth},
        functions={"utility": utility},
        constraints={"feasible": feasible},
    )


def _build_model(*, enable_jit: bool) -> Model:
    wealth_grid = WEALTH_GRID
    source = _decision_regime(
        transition={
            "near": _edge(probability=p_near, fallback_regime="source"),
            "far": _edge(probability=p_far, fallback_regime="far_fallback"),
        },
        active=lambda age: age < 2,
    )
    near = _decision_regime(transition=near_next, active=lambda age: age == 1)
    far = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wealth": wealth_grid},
        functions={"utility": utility_state},
    )
    far_fallback = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wealth": wealth_grid},
        functions={"utility": utility_state},
    )
    return Model(
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regimes={
            "source": source,
            "near": near,
            "far": far,
            "far_fallback": far_fallback,
        },
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [True, False])
def test_simulate_with_disjoint_edge_target_windows_lands_in_reachable_target(
    *, enable_jit: bool
) -> None:
    """A period whose only live edge leads to `near` routes every subject there."""
    model = _build_model(enable_jit=enable_jit)
    params = {
        "source": {"koopmans_aggregator": {"discount_factor": 0.95}},
        "near": {"koopmans_aggregator": {"discount_factor": 0.95}},
        "far": {},
        "far_fallback": {},
    }
    solution = model.solve(params=params, log_level="debug")
    result = model.simulate(
        params=params,
        initial_conditions={
            "regime_id": jnp.array([int(model.regime_names_to_ids["source"])] * 2),
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([5.0, 6.0]),
        },
        solution=solution,
        log_level="debug",
        seed=0,
    )
    df = result.to_dataframe()
    assert df.loc[df["period"] == 1, "regime_name"].tolist() == ["near", "near"]
