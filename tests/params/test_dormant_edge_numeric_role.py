"""An edge parameter's numeric role comes from the laws the model demands.

A `ByAge` case that no demanded age selects keeps its parameter slots, but its
annotations do not decide how the demanded case's inputs are converted.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, InitialNodes, Model, Regime, categorical
from lcm.transition import ByAge, StochasticTransition, Transition
from lcm.typing import FloatND, ScalarFloat, ScalarInt, UserParams


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    sink: ScalarInt


def _utility() -> FloatND:
    return jnp.asarray(0.0)


def _live_probability(*, hazard: ScalarFloat) -> FloatND:
    # Trace-time metadata: the callback must receive a float whatever was passed.
    assert hazard.dtype.kind == "f"
    return hazard


def _dormant_probability(*, hazard: ScalarInt) -> FloatND:  # noqa: ARG001
    msg = "A law that no demanded age selects was evaluated."
    raise AssertionError(msg)


def _model(*, include_dormant: bool) -> Model:
    cases = {0: {"sink": StochasticTransition(func=_live_probability)}}
    if include_dormant:
        cases[1] = {"sink": StochasticTransition(func=_dormant_probability)}
    return Model(
        regimes={
            name: Regime(functions={"utility": _utility}) for name in ("source", "sink")
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_age={0: "source"}),
        edges={"source": Transition(law=ByAge(cases=cases))},
        durable_identity=False,
    )


def _params(hazard: float | np.ndarray) -> UserParams:
    return {
        "discount_factor": 1.0,
        "edges": {"source": {"sink": {"hazard": hazard}}},
    }


@pytest.mark.parametrize("include_dormant", [False, True])
@pytest.mark.parametrize("hazard", [1, 1.0, np.asarray(1, dtype=np.int32)])
def test_demanded_float_law_receives_a_float(
    *, include_dormant: bool, hazard: float | np.ndarray
) -> None:
    _model(include_dormant=include_dormant).solve(
        params=_params(hazard), log_level="off"
    )
