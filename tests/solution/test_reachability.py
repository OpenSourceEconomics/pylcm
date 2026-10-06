import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ExecutionConfig,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.asarray(0.0)


def _low_utility() -> ScalarFloat:
    return jnp.asarray(0.0)


def _high_utility() -> ScalarFloat:
    return jnp.asarray(10.0)


def _probability_low(probability_high: ScalarFloat) -> ScalarFloat:
    return 1 - probability_high


def _probability_high(probability_high: ScalarFloat) -> ScalarFloat:
    return probability_high


def test_runtime_zero_probability_keeps_static_continuation_targets() -> None:
    """Free probabilities change values without changing graph membership."""
    source_law = {
        "low": StochasticTransition(func=_probability_low),
        "high": StochasticTransition(func=_probability_high),
    }
    model = Model(
        edges={"source": Transition(targets={"low": 0, "high": 0}, law=source_law)},
        regimes={
            "source": Regime(
                functions={"utility": _zero_utility},
            ),
            "low": Regime(
                functions={"utility": _low_utility},
            ),
            "high": Regime(
                functions={"utility": _high_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        initial_nodes={0: "source"},
    )
    graph_targets = model.reachability.solution.targets(period=0, source="source")

    low_solution = model.solve(
        params={"discount_factor": 1.0, "probability_high": 0.0},
        log_level="debug",
    ).values
    high_solution = model.solve(
        params={"discount_factor": 1.0, "probability_high": 1.0},
        log_level="debug",
    ).values

    assert graph_targets == ("high", "low")
    assert (
        model.reachability.solution.targets(period=0, source="source") == graph_targets
    )
    np.testing.assert_allclose(np.asarray(low_solution[0]["source"]), 0.0)
    np.testing.assert_allclose(np.asarray(high_solution[0]["source"]), 10.0)
