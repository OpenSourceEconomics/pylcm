"""Solution-graph continuation values are a simulation-input contract.

Promoted from the Pro-review reachability audit: forward simulation's
decision/Q evaluation reads continuation value-function arrays for every
target the *solution* graph names for the current `(period, source)` edge.
A caller-supplied `SolutionResult` missing one of those target values must raise a
descriptive `InvalidSimulationInputError`, not a bare `KeyError` and not a silent
zero/empty fallback.
"""

from dataclasses import replace
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, MarkovTransition, Model, Regime, categorical
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.asarray(0.0)


def _low_utility() -> ScalarFloat:
    return jnp.asarray(0.0)


def _high_utility() -> ScalarFloat:
    return jnp.asarray(10.0)


def _probability_high(probability_high: ScalarFloat) -> ScalarFloat:
    return probability_high


def _probability_low(probability_high: ScalarFloat) -> ScalarFloat:
    return 1 - probability_high


def _source_is_active(age: float) -> bool:
    return age < 1


def _target_is_active(age: float) -> bool:
    return age >= 1


def _build_model() -> Model:
    return Model(
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
        regime_id_class=_RegimeId,
        enable_jit=False,
    )


def test_missing_solution_graph_value_raises_descriptive_simulation_input_error() -> (
    None
):
    """A result missing a solution-graph target value fails closed."""
    model = _build_model()
    params = {"discount_factor": 1.0, "probability_high": 0.5}
    solution = model.solve(params=params, log_level="debug")

    assert model.reachability.solution.targets(period=0, source="source") == (
        "high",
        "low",
    )

    # Drop "high"'s period-1 value array, which the period-0 decision at
    # "source" needs to compute its continuation value.
    incomplete_solution = MappingProxyType(
        {
            **solution.values,
            1: MappingProxyType(
                {k: v for k, v in solution.values[1].items() if k != "high"}
            ),
        }
    )

    initial_conditions = {
        "age": jnp.array([0.0]),
        "regime_id": jnp.array([model.regime_names_to_ids["source"]]),
    }

    incomplete_result = replace(solution, values=incomplete_solution)

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"coverage.*missing=.*high",
    ):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=incomplete_result,
            log_level="debug",
        )
