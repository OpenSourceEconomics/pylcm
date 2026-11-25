"""A gate into a stateless target is decided in solve, too."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
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


def _open_by_age(age: FloatND) -> BoolND:
    return age >= 0


def _closed_by_age(age: FloatND) -> BoolND:
    return age < 0


def _build_model(*, gate: UserFunction, enable_jit: bool) -> Model:
    """Decide the landing by a state-only gate."""
    routes = {
        "only": MemberRoute(
            fallback=ProjectedRegimeValue(regime="stateless_fallback", projection={})
        )
    }
    gates = {"stateless_target": Gate(predicate=gate, routes=routes)}
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
                gates=gates,
            )
        },
    )


@pytest.mark.parametrize("enable_jit", [True, False])
@pytest.mark.parametrize(
    ("gate", "landing_value"),
    [(_open_by_age, _TARGET_VALUE), (_closed_by_age, _FALLBACK_VALUE)],
    ids=["state-open", "state-closed"],
)
def test_stateless_gated_target_pays_the_branch_its_gate_selects(
    *,
    gate: UserFunction,
    landing_value: float,
    enable_jit: bool,
) -> None:
    """The source's value discounts the target when accepted, else the fallback.

    With zero flow utility, the source's value at every state is the discounted value of
    the branch selected at the stateless landing: the target's own value when the
    gate accepts and the projected fallback otherwise.
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
        np.full((2, 1), _BETA * landing_value),
        decimal=DECIMAL_PRECISION,
    )
