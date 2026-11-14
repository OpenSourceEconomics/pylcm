"""Each scheduled regime law's result is checked in its own role.

A schedule turns a selector's code into an indicator and selects among
probability vectors. A float code or a Boolean probability is refused before
that, where the composed result would have hidden it.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, InitialNodes, Model, Regime, categorical
from lcm.exceptions import InvalidFunctionOutputError
from lcm.transition import (
    ByAge,
    DeterministicTransition,
    StochasticTransition,
    Transition,
)
from lcm.typing import BoolND, FloatND, ScalarInt, UserFunction


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    sink: ScalarInt


def utility() -> FloatND:
    return jnp.asarray(0.0)


def integer_selector() -> ScalarInt:
    return jnp.asarray(0, dtype=jnp.int32)


def floating_selector() -> FloatND:
    return jnp.asarray(0.0)


def certain() -> FloatND:
    return jnp.asarray(1.0)


def _mixed_model(selector: UserFunction) -> Model:
    return Model(
        regimes={
            name: Regime(functions={"utility": utility}) for name in ("source", "sink")
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_age={0: "source"}),
        edges={
            "source": Transition(
                targets={"source": 0, "sink": 1},
                law=ByAge(
                    cases={
                        0: DeterministicTransition(func=selector),
                        1: {"sink": StochasticTransition(func=certain)},
                    }
                ),
            )
        },
        fixed_params={"discount_factor": 1.0},
        durable_identity=False,
    )


def test_mixed_schedule_rejects_float_before_selector_becomes_indicator() -> None:
    _mixed_model(integer_selector).solve(params={}, log_level="off")
    with pytest.raises(InvalidFunctionOutputError):
        _mixed_model(floating_selector).solve(params={}, log_level="off")


def boolean_source_probabilities() -> BoolND:
    return jnp.asarray([True, False])


def floating_source_probabilities() -> FloatND:
    return jnp.asarray([1.0, 0.0])


def floating_sink_probabilities() -> FloatND:
    return jnp.asarray([0.0, 1.0])


def _vector_model(first: UserFunction) -> Model:
    return Model(
        regimes={
            name: Regime(functions={"utility": utility}) for name in ("source", "sink")
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_age={0: "source"}),
        edges={
            "source": Transition(
                targets={"source": 0, "sink": 1},
                law=ByAge(
                    cases={
                        # The Boolean law is the refused input under test.
                        0: StochasticTransition(func=first),  # ty: ignore[invalid-argument-type]
                        1: StochasticTransition(func=floating_sink_probabilities),
                    }
                ),
            )
        },
        fixed_params={"discount_factor": 1.0},
        durable_identity=False,
    )


def test_vector_schedule_checks_each_result_before_dtype_promotion() -> None:
    _vector_model(floating_source_probabilities).solve(params={}, log_level="off")
    with pytest.raises(InvalidFunctionOutputError):
        _vector_model(boolean_source_probabilities).solve(params={}, log_level="off")
