"""A gate's predicate and projections read `<state>_code` of a discrete target state.

The source and target regimes both carry a continuous `x`, a value-grid state
`zz` with values `10, 20` (codes `0, 1`) and a category-class state `kk`. Every
subject's ordinary transition lands at `x = 1`. The gate's single route falls
back into the target itself, so a closed gate lands the subject at the
projected `x` instead.

- In the projection case the gate is always closed and the fallback projects
  `x` to `0.5 * <state>_code`.
- In the predicate case the gate is open iff `<state>_code == 1` and the
  fallback projects `x` to `0.5 * x`.

`<state>_code` is bound by the engine wherever `<state>` is, so it is never a
parameter.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from dags.tree import flatten_to_qnames
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    Gate,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.transition import StochasticTransition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
)
from tests.conftest import DECIMAL_PRECISION

_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
_SAVING = IrregSpacedGrid(points=np.array([1.0, 2.0]))


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=True)
class _Kind:
    low: ScalarInt
    high: ScalarInt


def _always_true(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _always_closed(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x, dtype=bool)


def _open_if_zz_code_is_one(zz_code: DiscreteState) -> FloatND:
    return zz_code == 1


def _open_if_kk_code_is_one(kk_code: DiscreteState) -> FloatND:
    return kk_code == 1


def _half_x(x: ContinuousState) -> ContinuousState:
    return 0.5 * x


def _x_from_zz_code(zz_code: DiscreteState) -> ContinuousState:
    return 0.5 * zz_code


def _x_from_kk_code(kk_code: DiscreteState) -> ContinuousState:
    return 0.5 * kk_code


def _stay_zz(zz: DiscreteState) -> DiscreteState:
    return zz


def _stay_kk(kk: DiscreteState) -> DiscreteState:
    return kk


def _costly_saving(*, x: ContinuousState, saving: ContinuousAction) -> FloatND:
    return -saving + 0.0 * x


def _next_x(saving: ContinuousAction) -> ContinuousState:
    return saving


def _constant_utility(x: ContinuousState) -> FloatND:
    return 0.0 * x


_PREDICATE_READING_CODE = {"zz": _open_if_zz_code_is_one, "kk": _open_if_kk_code_is_one}
_PROJECTION_READING_CODE = {"zz": _x_from_zz_code, "kk": _x_from_kk_code}

# Subject `i` starts on code `i` of both discrete states.
_EXPECTED_X = {
    # closed: `x = 0.5 * code`
    "projection": [0.0, 0.5],
    # code 0 closed at `0.5 * 1`, code 1 open at the candidate `1`
    "predicate": [0.5, 1.0],
}

_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "target": {},
}

_CASES = [
    pytest.param(state, role, id=f"{role}-{state}")
    for state in ("zz", "kk")
    for role in ("projection", "predicate")
]


def _model(*, state: StateName, role: str) -> Model:
    predicate: Callable[..., FloatND] = (
        _PREDICATE_READING_CODE[state] if role == "predicate" else _always_closed
    )
    project_x: Callable[..., ContinuousState] = (
        _PROJECTION_READING_CODE[state] if role == "projection" else _half_x
    )
    states = {
        "x": _X,
        "zz": DiscreteGrid(values=np.array([10, 20])),
        "kk": DiscreteGrid(category_class=_Kind),
    }
    return Model(
        edges={
            "source": Transition(
                targets={"target": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_always_true)}}
                ),
                gates={
                    "target": Gate(
                        predicate=predicate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="target",
                                    projection={
                                        "x": project_x,
                                        "zz": _stay_zz,
                                        "kk": _stay_kk,
                                    },
                                )
                            )
                        },
                        off_grid="pointwise",
                    )
                },
            )
        },
        regimes={
            "source": Regime(
                states=states,
                state_transitions={"x": _next_x, "zz": _stay_zz, "kk": _stay_kk},
                actions={"saving": _SAVING},
                functions={"utility": _costly_saving},
            ),
            "target": Regime(states=states, functions={"utility": _constant_utility}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "source"},
    )


@pytest.mark.parametrize(("state", "role"), _CASES)
def test_code_argument_of_a_target_state_is_no_edge_parameter(
    *, state: StateName, role: str
) -> None:
    """No path of the params template ends in `<state>_code`."""
    template = _model(state=state, role=role).get_params_template()

    assert [
        qname
        for qname in flatten_to_qnames(template)
        if qname.endswith(f"{state}_code")
    ] == []


@pytest.mark.parametrize(("state", "role"), _CASES)
def test_code_argument_of_a_target_state_receives_the_code(
    *, state: StateName, role: str
) -> None:
    """Subjects on codes `0, 1` land where the gate reading those codes sends them."""
    model = _model(state=state, role=role)
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "x": jnp.zeros(2),
            "zz": jnp.array([10, 20]),
            "kk": jnp.array([0, 1]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _RegimeId.source, dtype=jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    landed = result.to_dataframe().query("period == 1").sort_values("subject_id")

    aaae(landed["x"].to_numpy(), _EXPECTED_X[role], decimal=DECIMAL_PRECISION)
