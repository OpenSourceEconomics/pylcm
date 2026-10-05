"""A gated edge into a stateless target is gated in solve, not just in simulation."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.collective import (
    ProjectedRegimeValue,
    StakeholderRoute,
    StochasticTransition,
    ValueDependentTransition,
)
from lcm.regime import Regime
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.graph import with_fixture_graph
from tests.test_models.schedules import until_exit

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


def _build_model(*, gate, enable_jit: bool) -> Model:
    src = Regime(
        regime_transitions=until_exit(
            1,
            law={
                "stateless_target": ValueDependentTransition(
                    probability=StochasticTransition(func=_prob_one),
                    gate=gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="stateless_fallback", projection={}
                            )
                        )
                    },
                )
            },
            exits=("stateless_target",),
        ),
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _identity_x},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_src},
    )
    stateless_target = Regime(
        regime_transitions=None,
        functions={"utility": _u_stateless_target},
    )
    stateless_fallback = Regime(
        regime_transitions=None,
        functions={"utility": _u_stateless_fallback},
    )
    return with_fixture_graph(
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
    )


@pytest.mark.parametrize("enable_jit", [True, False])
@pytest.mark.parametrize(
    ("gate", "landing_value"),
    [(_open_gate, _TARGET_VALUE), (_closed_gate, _FALLBACK_VALUE)],
    ids=["open", "closed"],
)
def test_stateless_gated_target_pays_the_branch_its_gate_selects(
    *, gate, landing_value: float, enable_jit: bool
) -> None:
    """The source's value discounts the target when the gate opens, else the fallback.

    With zero flow utility, the source's value at every state is the discounted
    value of the branch the gate selects at the stateless landing: the target's
    own value when the gate is open and the projected fallback when it is closed.
    """
    model = _build_model(gate=gate, enable_jit=enable_jit)
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
