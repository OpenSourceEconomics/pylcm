import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, MarkovTransition, Model, Regime, categorical
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.float32(0)


def _low_utility() -> ScalarFloat:
    return jnp.float32(0)


def _high_utility() -> ScalarFloat:
    return jnp.float32(10)


def _probability_low(probability_high: ScalarFloat) -> ScalarFloat:
    return 1 - probability_high


def _probability_high(probability_high: ScalarFloat) -> ScalarFloat:
    return probability_high


def _one_probability() -> ScalarFloat:
    return jnp.float32(1)


def _positive_dormant_probability() -> ScalarFloat:
    return jnp.float32(0.1)


def _source_is_active(age: float) -> bool:
    return age < 1


def _target_is_active(age: float) -> bool:
    return age >= 1


def test_runtime_zero_probability_keeps_static_continuation_targets() -> None:
    """Free probabilities change values without changing graph membership."""
    model = Model(
        regimes={
            "source": Regime(
                transition={
                    "low": MarkovTransition(_probability_low),
                    "high": MarkovTransition(_probability_high),
                },
                active=_source_is_active,
                functions={"utility": _zero_utility},
            ),
            "low": Regime(
                transition=None,
                active=_target_is_active,
                functions={"utility": _low_utility},
            ),
            "high": Regime(
                transition=None,
                active=_target_is_active,
                functions={"utility": _high_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    graph_targets = model.reachability.solution.targets(period=0, source="source")

    low_solution = model.solve(
        params={"discount_factor": 1.0, "probability_high": 0.0},
        log_level="debug",
    )
    high_solution = model.solve(
        params={"discount_factor": 1.0, "probability_high": 1.0},
        log_level="debug",
    )

    assert graph_targets == ("high", "low")
    assert (
        model.reachability.solution.targets(period=0, source="source") == graph_targets
    )
    np.testing.assert_allclose(np.asarray(low_solution[0]["source"]), 0.0)
    np.testing.assert_allclose(np.asarray(high_solution[0]["source"]), 10.0)


def test_positive_granular_probability_outside_graph_is_rejected() -> None:
    """A dormant declaration cannot receive positive transition probability."""

    @categorical(ordered=False)
    class _DormantRegimeId:
        source: ScalarInt
        target: ScalarInt
        dormant: ScalarInt

    model = Model(
        regimes={
            "source": Regime(
                transition={
                    "target": MarkovTransition(_one_probability),
                    "dormant": MarkovTransition(_positive_dormant_probability),
                },
                active=lambda age: age < 1,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                functions={"utility": _low_utility},
            ),
            "dormant": Regime(
                transition=None,
                active=lambda age: age < 1,
                functions={"utility": _low_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_DormantRegimeId,
        enable_jit=False,
    )

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"Regime 'dormant'.*'source'.*period 0",
    ):
        model.solve(params={"discount_factor": 1.0}, log_level="debug")
