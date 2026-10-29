"""An arrival into a stateless target is decided in solve, not just in simulation."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.regime import Regime
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_BETA = 0.9
_TARGET_VALUE = 1.0
_FALLBACK_VALUE = 0.25


@categorical(ordered=False)
class RegimeId:
    src: ScalarInt
    stateless_target: ScalarInt
    stateless_fallback: ScalarInt


@categorical(ordered=False)
class Work:
    leisure: ScalarInt
    work: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _u_src(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _u_stateless_target() -> FloatND:
    return jnp.asarray(_TARGET_VALUE)


def _u_stateless_fallback() -> FloatND:
    return jnp.asarray(_FALLBACK_VALUE)


def _open_gate(V_target: FloatND) -> BoolND:
    return V_target > _TARGET_VALUE / 2


def _closed_gate(V_target: FloatND) -> BoolND:
    return V_target > 2 * _TARGET_VALUE


def _closed_by_age(age: FloatND) -> BoolND:
    return age < 0


def _build_model(*, gate, enable_jit: bool, reads_value: bool = True) -> Model:
    """Decide the landing on the target's value on arrival, or by a state-only gate."""
    routes = {
        "only": StakeholderRoute(
            fallback=ProjectedRegimeValue(regime="stateless_fallback", projection={})
        )
    }
    gates: dict[str, Gate] = {}
    arrivals: dict[str, ArrivalDecision] = {}
    if reads_value:
        arrivals["stateless_target"] = ArrivalDecision(
            predicate=gate,
            reads={
                "V_target": ArrivalRead(
                    kind="IntegratedFull", regime="stateless_target", projection={}
                )
            },
            routes=routes,
        )
    else:
        gates["stateless_target"] = Gate(predicate=gate, routes=routes)
    src = Regime(
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _identity_x},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_src},
    )
    stateless_target = Regime(
        functions={"utility": _u_stateless_target},
    )
    stateless_fallback = Regime(
        functions={"utility": _u_stateless_fallback},
    )
    return Model(
        regimes={
            "src": src,
            "stateless_target": stateless_target,
            "stateless_fallback": stateless_fallback,
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                targets={"stateless_target": 0, "stateless_fallback": 0},
                law=ByAge(
                    cases={
                        0: {"stateless_target": StochasticTransition(func=_prob_one)}
                    }
                ),
                arrivals=arrivals,
                gates=gates,
            )
        },
    )


@pytest.mark.parametrize("enable_jit", [True, False])
@pytest.mark.parametrize(
    ("gate", "reads_value", "landing_value"),
    [
        (_open_gate, True, _TARGET_VALUE),
        (_closed_gate, True, _FALLBACK_VALUE),
        (_closed_by_age, False, _FALLBACK_VALUE),
    ],
    ids=["open", "closed", "state-closed"],
)
def test_stateless_gated_target_pays_the_branch_its_gate_selects(
    *, gate, reads_value: bool, landing_value: float, enable_jit: bool
) -> None:
    """The source's value discounts the target when accepted, else the fallback.

    With zero flow utility, the source's value at every state is the discounted
    value of the branch selected at the stateless landing: the target's own
    value when the arrival or gate accepts and the projected fallback otherwise.
    """
    model = _build_model(gate=gate, enable_jit=enable_jit, reads_value=reads_value)
    params = {
        "src": {"koopmans_aggregator": {"discount_factor": _BETA}},
        "stateless_target": {},
        "stateless_fallback": {},
    }
    solution = model.solve(params=params, log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution.values[0]["src"]),
        np.full(2, _BETA * landing_value),
        decimal=DECIMAL_PRECISION,
    )
